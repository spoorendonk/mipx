// Sparse supernodal Cholesky for the barrier's normal equations (#290).
//
// References: J. W. H. Liu, "The role of elimination trees in sparse
// factorization" (1990) for the elimination tree; T. A. Davis, "Direct Methods
// for Sparse Linear Systems" (SIAM 2006), sec. 4.1-4.4, for the etree,
// postorder and row-subtree column counts; E. Ng and B. Peyton, "Block sparse
// Cholesky algorithms on advanced uniprocessor computers" (1993), for
// supernodes; C. Ashcraft and R. Grimes, "The influence of relaxed supernode
// partitions on the multifrontal method" (1989), for relaxed amalgamation (the
// thresholds are CHOLMOD's defaults).

#include "mipx/sparse_cholesky.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <string>

#ifdef MIPX_BLAS_LIBRARY
#include <dlfcn.h>
#endif

namespace mipx {

namespace {

// BLAS is loaded with dlopen on first use, not linked: OpenBLAS sets up its
// thread pool when it loads, reserving per-core buffers, which would fail
// every mipx binary under an address-space limit, barrier or not. Loaded
// here, it costs only the runs that factor on it, and a missing library falls
// back to mipx's own loops. Fortran BLAS (LP64), with the hidden lengths of
// the character arguments.
using DgemmFn = void (*)(const char*, const char*, const int*, const int*, const int*,
                         const double*, const double*, const int*, const double*, const int*,
                         const double*, double*, const int*, std::size_t, std::size_t);
using DsyrkFn = void (*)(const char*, const char*, const int*, const int*, const double*,
                         const double*, const int*, const double*, double*, const int*, std::size_t,
                         std::size_t);
using DtrsmFn = void (*)(const char*, const char*, const char*, const char*, const int*, const int*,
                         const double*, const double*, const int*, double*, const int*, std::size_t,
                         std::size_t, std::size_t, std::size_t);
using SetThreadsFn = void (*)(int);

struct Blas {
    DgemmFn dgemm = nullptr;
    DsyrkFn dsyrk = nullptr;
    DtrsmFn dtrsm = nullptr;
    SetThreadsFn set_threads = nullptr;
};

std::atomic<bool> g_blas_enabled{true};
std::atomic<Int> g_blas_threads{1};

// The loaded library, or nullptr without one. Loaded once per process.
const Blas* loadedBlas() {
    static const Blas* const blas = []() -> const Blas* {
#ifdef MIPX_BLAS_LIBRARY
        // OpenBLAS sizes its pool, and the buffers it reserves, from this
        // variable when it loads, and from the core count without it, so it
        // is set for the load and removed again, leaving the environment as
        // it was for child processes and later BLAS loads. A value the user
        // set sizes the pool instead; either way the barrier's thread count
        // applies at each factorize (blas() below). setenv is not safe
        // against a concurrent getenv: this runs once per process.
        const bool set_env = std::getenv("OPENBLAS_NUM_THREADS") == nullptr;
        if (set_env) {
            const std::string threads =
                std::to_string(g_blas_threads.load(std::memory_order_relaxed));
            setenv("OPENBLAS_NUM_THREADS", threads.c_str(), 0);
        }
        // The found path first (a -dev symlink follows library upgrades),
        // then the file it resolved to at configure time (present on a host
        // with the runtime package only).
        void* handle = dlopen(MIPX_BLAS_LIBRARY, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            handle = dlopen(MIPX_BLAS_LIBRARY_REAL, RTLD_NOW | RTLD_LOCAL);
        }
        if (set_env) {
            unsetenv("OPENBLAS_NUM_THREADS");
        }
        if (handle == nullptr) {
            return nullptr;
        }
        static Blas b;
        b.dgemm = reinterpret_cast<DgemmFn>(dlsym(handle, "dgemm_"));
        b.dsyrk = reinterpret_cast<DsyrkFn>(dlsym(handle, "dsyrk_"));
        b.dtrsm = reinterpret_cast<DtrsmFn>(dlsym(handle, "dtrsm_"));
        b.set_threads = reinterpret_cast<SetThreadsFn>(dlsym(handle, "openblas_set_num_threads"));
        if (b.dgemm == nullptr || b.dsyrk == nullptr || b.dtrsm == nullptr) {
            return nullptr;
        }
        return &b;
#else
        return nullptr;
#endif
    }();
    return blas;
}

// The library to factor on now, with its thread count applied, or nullptr
// for mipx's own loops.
const Blas* blas() {
    if (!g_blas_enabled.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    const Blas* b = loadedBlas();
    if (b != nullptr && b->set_threads != nullptr) {
        // openblas_set_num_threads is process-wide; set it on each use so
        // the last caller's count, not OpenBLAS's all-cores default, holds.
        static std::mutex mutex;
        static Int applied = 0;
        const Int want = g_blas_threads.load(std::memory_order_relaxed);
        const std::lock_guard<std::mutex> lock(mutex);
        if (applied != want) {
            b->set_threads(want);
            applied = want;
        }
    }
    return b;
}

// Pattern of the permuted matrix, off-diagonal entries only, both ways:
// upper column k lists rows i < k, lower column j lists rows i > j.
struct PermutedPattern {
    std::vector<Index> up_ptr, up_idx;
    std::vector<Index> lo_ptr, lo_idx;
};

void buildPermutedPattern(Index n, std::span<const Index> col_starts,
                          std::span<const Index> row_indices, const std::vector<Index>& iperm,
                          PermutedPattern& pat) {
    pat.up_ptr.assign(static_cast<size_t>(n) + 1, 0);
    pat.lo_ptr.assign(static_cast<size_t>(n) + 1, 0);
    for (Index c = 0; c < n; ++c) {
        for (Index p = col_starts[c]; p < col_starts[c + 1]; ++p) {
            const Index a = iperm[row_indices[p]];
            const Index b = iperm[c];
            if (a == b) {
                continue;
            }
            ++pat.up_ptr[std::max(a, b) + 1];
            ++pat.lo_ptr[std::min(a, b) + 1];
        }
    }
    for (Index k = 0; k < n; ++k) {
        pat.up_ptr[k + 1] += pat.up_ptr[k];
        pat.lo_ptr[k + 1] += pat.lo_ptr[k];
    }
    pat.up_idx.resize(static_cast<size_t>(pat.up_ptr[n]));
    pat.lo_idx.resize(static_cast<size_t>(pat.lo_ptr[n]));
    std::vector<Index> up_next(pat.up_ptr.begin(), pat.up_ptr.end() - 1);
    std::vector<Index> lo_next(pat.lo_ptr.begin(), pat.lo_ptr.end() - 1);
    for (Index c = 0; c < n; ++c) {
        for (Index p = col_starts[c]; p < col_starts[c + 1]; ++p) {
            const Index a = iperm[row_indices[p]];
            const Index b = iperm[c];
            if (a == b) {
                continue;
            }
            const Index lo = std::min(a, b);
            const Index hi = std::max(a, b);
            pat.up_idx[up_next[hi]++] = lo;
            pat.lo_idx[lo_next[lo]++] = hi;
        }
    }
}

// Elimination tree from the upper pattern (Liu's algorithm with path
// compression; Davis 2006, cs_etree).
std::vector<Index> eliminationTree(Index n, const PermutedPattern& pat) {
    std::vector<Index> parent(static_cast<size_t>(n), -1);
    std::vector<Index> ancestor(static_cast<size_t>(n), -1);
    for (Index k = 0; k < n; ++k) {
        for (Index p = pat.up_ptr[k]; p < pat.up_ptr[k + 1]; ++p) {
            Index i = pat.up_idx[p];
            while (i != -1 && i < k) {
                const Index inext = ancestor[i];
                ancestor[i] = k;
                if (inext == -1) {
                    parent[i] = k;
                }
                i = inext;
            }
        }
    }
    return parent;
}

// Postorder of a forest (Davis 2006, cs_post): post[k] is the k-th node.
std::vector<Index> postorder(Index n, const std::vector<Index>& parent) {
    std::vector<Index> head(static_cast<size_t>(n), -1);
    std::vector<Index> next(static_cast<size_t>(n), -1);
    for (Index j = n - 1; j >= 0; --j) {
        if (parent[j] != -1) {
            next[j] = head[parent[j]];
            head[parent[j]] = j;
        }
    }
    std::vector<Index> post;
    post.reserve(static_cast<size_t>(n));
    std::vector<Index> stack;
    for (Index j = 0; j < n; ++j) {
        if (parent[j] != -1) {
            continue;
        }
        stack.push_back(j);
        while (!stack.empty()) {
            const Index p = stack.back();
            const Index child = head[p];
            if (child == -1) {
                stack.pop_back();
                post.push_back(p);
            } else {
                head[p] = next[child];
                stack.push_back(child);
            }
        }
    }
    return post;
}

// Column counts of L, diagonal included, from the row subtrees: row k of L
// has a nonzero in column j for every j on the etree paths from the entries of
// upper column k up to k (Davis 2006, cs_ereach). O(nnz(L)) time, O(n) space.
std::vector<Index> columnCounts(Index n, const PermutedPattern& pat,
                                const std::vector<Index>& parent) {
    std::vector<Index> count(static_cast<size_t>(n), 1);
    std::vector<Index> mark(static_cast<size_t>(n), -1);
    for (Index k = 0; k < n; ++k) {
        mark[k] = k;
        for (Index p = pat.up_ptr[k]; p < pat.up_ptr[k + 1]; ++p) {
            for (Index j = pat.up_idx[p]; mark[j] != k; j = parent[j]) {
                mark[j] = k;
                ++count[j];
            }
        }
    }
    return count;
}

// CHOLMOD's default relaxation: a supernode of up to 4 columns is always
// merged; up to 16 while under 80% explicit zeros, up to 48 under 10%, larger
// under 5%.
bool relaxedMergeAllowed(Index ncols, double zero_fraction) {
    if (ncols <= 4) {
        return true;
    }
    if (ncols <= 16) {
        return zero_fraction < 0.8;
    }
    if (ncols <= 48) {
        return zero_fraction < 0.1;
    }
    return zero_fraction < 0.05;
}

// Entries of a column-major trapezoid block with `nrows` rows and `ncols`
// columns whose column t holds nrows - t rows.
double trapezoidEntries(Index nrows, Index ncols) {
    return static_cast<double>(ncols) * static_cast<double>(nrows) -
           0.5 * static_cast<double>(ncols) * static_cast<double>(ncols - 1);
}

// Dense kernels on column-major blocks: BLAS-3 when a library is loaded,
// plain loops otherwise.

// C = C - A·Bᵀ, C m×n, A m×k, B n×k.
void gemmMinusNT(const Blas* blas, Index m, Index n, Index k, const Real* a, Index lda,
                 const Real* b, Index ldb, Real* c, Index ldc) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    if (blas != nullptr) {
        const char no = 'N';
        const char tr = 'T';
        const Real minus_one = -1.0;
        const Real one = 1.0;
        blas->dgemm(&no, &tr, &m, &n, &k, &minus_one, a, &lda, b, &ldb, &one, c, &ldc, 1, 1);
        return;
    }
    for (Index j = 0; j < n; ++j) {
        Real* cj = c + static_cast<std::ptrdiff_t>(j) * ldc;
        for (Index p = 0; p < k; ++p) {
            const Real bjp = b[j + static_cast<std::ptrdiff_t>(p) * ldb];
            if (bjp == 0.0) {
                continue;
            }
            const Real* ap = a + static_cast<std::ptrdiff_t>(p) * lda;
            for (Index i = 0; i < m; ++i) {
                cj[i] -= ap[i] * bjp;
            }
        }
    }
}

// Lower triangle of C = C - A·Aᵀ, C n×n, A n×k.
void syrkMinusLN(const Blas* blas, Index n, Index k, const Real* a, Index lda, Real* c, Index ldc) {
    if (n <= 0 || k <= 0) {
        return;
    }
    if (blas != nullptr) {
        const char lo = 'L';
        const char no = 'N';
        const Real minus_one = -1.0;
        const Real one = 1.0;
        blas->dsyrk(&lo, &no, &n, &k, &minus_one, a, &lda, &one, c, &ldc, 1, 1);
        return;
    }
    for (Index j = 0; j < n; ++j) {
        Real* cj = c + static_cast<std::ptrdiff_t>(j) * ldc;
        for (Index p = 0; p < k; ++p) {
            const Real* ap = a + static_cast<std::ptrdiff_t>(p) * lda;
            const Real ajp = ap[j];
            if (ajp == 0.0) {
                continue;
            }
            for (Index i = j; i < n; ++i) {
                cj[i] -= ap[i] * ajp;
            }
        }
    }
}

// B = B·L⁻ᵀ, B m×n, L n×n lower triangular.
void trsmRightLowerTrans(const Blas* blas, Index m, Index n, const Real* l, Index ldl, Real* b,
                         Index ldb) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (blas != nullptr) {
        const char right = 'R';
        const char lo = 'L';
        const char tr = 'T';
        const char non_unit = 'N';
        const Real one = 1.0;
        blas->dtrsm(&right, &lo, &tr, &non_unit, &m, &n, &one, l, &ldl, b, &ldb, 1, 1, 1, 1);
        return;
    }
    // Column j of B·L⁻ᵀ: (b_j - sum_{p<j} x_p l_jp) / l_jj.
    for (Index j = 0; j < n; ++j) {
        Real* bj = b + static_cast<std::ptrdiff_t>(j) * ldb;
        for (Index p = 0; p < j; ++p) {
            const Real ljp = l[j + static_cast<std::ptrdiff_t>(p) * ldl];
            if (ljp == 0.0) {
                continue;
            }
            const Real* xp = b + static_cast<std::ptrdiff_t>(p) * ldb;
            for (Index i = 0; i < m; ++i) {
                bj[i] -= xp[i] * ljp;
            }
        }
        const Real inv = 1.0 / l[j + static_cast<std::ptrdiff_t>(j) * ldl];
        for (Index i = 0; i < m; ++i) {
            bj[i] *= inv;
        }
    }
}

// Panel width of the blocked dense Cholesky of a supernode, and the cap on
// the scratch block of one update (in reals).
constexpr Index kPanel = 64;
constexpr std::size_t kMaxUpdateEntries = std::size_t{1} << 22;

}  // namespace

void SparseCholesky::setBlasThreads(Int threads) {
    g_blas_threads.store(std::max<Int>(threads, 1), std::memory_order_relaxed);
}

void SparseCholesky::setBlasEnabled(bool enabled) {
    g_blas_enabled.store(enabled, std::memory_order_relaxed);
}

bool SparseCholesky::blasAvailable() {
    return loadedBlas() != nullptr;
}

bool SparseCholesky::analyze(Index n, std::span<const Index> col_starts,
                             std::span<const Index> row_indices, std::span<const Index> perm,
                             std::size_t max_bytes) {
    error_.clear();
    try {
        return analyzeImpl(n, col_starts, row_indices, perm, max_bytes);
    } catch (const std::bad_alloc&) {
        // Release everything the analysis built; keep the stats it reached.
        const Stats reached = stats_;
        *this = SparseCholesky{};
        stats_ = reached;
        error_ = "out of memory in the sparse Cholesky analysis";
        return false;
    }
}

bool SparseCholesky::analyzeImpl(Index n, std::span<const Index> col_starts,
                                 std::span<const Index> row_indices, std::span<const Index> perm,
                                 std::size_t max_bytes) {
    const auto t0 = std::chrono::steady_clock::now();
    n_ = n;
    stats_ = Stats{};
    stats_.n = n;
    stats_.nnz_input = static_cast<std::size_t>(col_starts[n]);
    num_dependent_ = 0;

    // Fill-reducing order, then the etree's postorder on top of it, which
    // keeps the fill and makes every supernode a contiguous range of columns.
    std::vector<Index> iperm0(static_cast<size_t>(n));
    for (Index k = 0; k < n; ++k) {
        iperm0[perm[k]] = k;
    }
    PermutedPattern pat;
    buildPermutedPattern(n, col_starts, row_indices, iperm0, pat);
    const std::vector<Index> post = postorder(n, eliminationTree(n, pat));
    perm_.resize(static_cast<size_t>(n));
    iperm_.resize(static_cast<size_t>(n));
    for (Index k = 0; k < n; ++k) {
        perm_[k] = perm[post[k]];
        iperm_[perm_[k]] = k;
    }
    buildPermutedPattern(n, col_starts, row_indices, iperm_, pat);
    const std::vector<Index> parent = eliminationTree(n, pat);
    const std::vector<Index> count = columnCounts(n, pat, parent);
    for (Index j = 0; j < n; ++j) {
        stats_.nnz_l += static_cast<std::size_t>(count[j]);
        stats_.factor_flops += static_cast<double>(count[j]) * static_cast<double>(count[j]);
    }

    std::vector<Index> nchild(static_cast<size_t>(n), 0);
    for (Index j = 0; j < n; ++j) {
        if (parent[j] != -1) {
            ++nchild[parent[j]];
        }
    }

    // Fundamental supernodes (in postorder, j+1 continues j's supernode when
    // it is j's parent, its only child, and its column is j's minus one row),
    // merged with the supernode just before them where the relaxation allows.
    // A supernode is (first column, columns, rows, explicit zeros); its rows
    // are its own columns then the rows below them.
    struct Snode {
        Index first;
        Index ncols;
        Index nrows;
        double zeros;
    };
    std::vector<Snode> fund;
    for (Index j = 0; j < n; ++j) {
        if (j > 0 && parent[j - 1] == j && nchild[j] == 1 && count[j - 1] == count[j] + 1) {
            ++fund.back().ncols;
        } else {
            fund.push_back({j, 1, count[j], 0.0});
        }
    }
    std::vector<Snode> merged;
    merged.reserve(fund.size());
    for (const Snode& f : fund) {
        Snode cur = f;
        const Index last = cur.first + cur.ncols - 1;
        while (!merged.empty()) {
            const Snode& prev = merged.back();
            const Index prev_last = prev.first + prev.ncols - 1;
            const Index pp = parent[prev_last];
            if (pp < cur.first || pp > last) {
                break;  // prev is not a child of cur
            }
            const Index ncols = prev.ncols + cur.ncols;
            const Index nrows = prev.ncols + cur.nrows;
            const double zeros =
                prev.zeros + cur.zeros +
                static_cast<double>(prev.ncols) * static_cast<double>(nrows - prev.nrows);
            if (!relaxedMergeAllowed(ncols, zeros / trapezoidEntries(nrows, ncols))) {
                break;
            }
            cur = {prev.first, ncols, nrows, zeros};
            merged.pop_back();
        }
        merged.push_back(cur);
    }
    const auto ns = static_cast<Index>(merged.size());
    stats_.num_supernodes = ns;

    snode_start_.resize(static_cast<size_t>(ns) + 1);
    col_snode_.resize(static_cast<size_t>(n));
    for (Index s = 0; s < ns; ++s) {
        snode_start_[s] = merged[s].first;
        stats_.max_supernode_cols = std::max(stats_.max_supernode_cols, merged[s].ncols);
        for (Index t = 0; t < merged[s].ncols; ++t) {
            col_snode_[merged[s].first + t] = s;
        }
    }
    snode_start_[ns] = n;

    // Row structure of each supernode: its columns, the matrix's rows below
    // them, and its children's rows below them.
    std::vector<Index> sparent(static_cast<size_t>(ns), -1);
    std::vector<Index> child_head(static_cast<size_t>(ns), -1);
    std::vector<Index> child_next(static_cast<size_t>(ns), -1);
    for (Index s = ns - 1; s >= 0; --s) {
        const Index p = parent[snode_start_[s + 1] - 1];
        if (p != -1) {
            sparent[s] = col_snode_[p];
            child_next[s] = child_head[sparent[s]];
            child_head[sparent[s]] = s;
        }
    }
    snode_row_ptr_.assign(static_cast<size_t>(ns) + 1, 0);
    snode_rows_.clear();
    std::size_t total_rows = 0;
    for (Index s = 0; s < ns; ++s) {
        total_rows += static_cast<std::size_t>(merged[s].nrows);
    }
    snode_rows_.reserve(total_rows);
    std::vector<Index> mark(static_cast<size_t>(n), -1);
    std::vector<Index> below;
    for (Index s = 0; s < ns; ++s) {
        const Index first = snode_start_[s];
        const Index last = snode_start_[s + 1] - 1;
        below.clear();
        for (Index j = first; j <= last; ++j) {
            for (Index p = pat.lo_ptr[j]; p < pat.lo_ptr[j + 1]; ++p) {
                const Index i = pat.lo_idx[p];
                if (i > last && mark[i] != s) {
                    mark[i] = s;
                    below.push_back(i);
                }
            }
        }
        for (Index c = child_head[s]; c != -1; c = child_next[c]) {
            const Index c_ncols = snode_start_[c + 1] - snode_start_[c];
            for (std::size_t q = snode_row_ptr_[c] + static_cast<std::size_t>(c_ncols);
                 q < snode_row_ptr_[c + 1]; ++q) {
                const Index i = snode_rows_[q];
                if (i > last && mark[i] != s) {
                    mark[i] = s;
                    below.push_back(i);
                }
            }
        }
        std::sort(below.begin(), below.end());
        for (Index j = first; j <= last; ++j) {
            snode_rows_.push_back(j);
        }
        snode_rows_.insert(snode_rows_.end(), below.begin(), below.end());
        snode_row_ptr_[s + 1] = snode_rows_.size();
    }
    pat = PermutedPattern{};

    snode_val_ptr_.assign(static_cast<size_t>(ns) + 1, 0);
    for (Index s = 0; s < ns; ++s) {
        const auto ncols = static_cast<std::size_t>(snode_start_[s + 1] - snode_start_[s]);
        const std::size_t nrows = snode_row_ptr_[s + 1] - snode_row_ptr_[s];
        snode_val_ptr_[s + 1] = snode_val_ptr_[s] + ncols * nrows;
    }
    stats_.stored = snode_val_ptr_[ns];

    const std::size_t bytes = factorBytes();
    stats_.analyze_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (max_bytes != 0 && bytes > max_bytes) {
        error_ = "sparse Cholesky factor needs " + std::to_string(bytes >> 20) +
                 " MB, above the cap of " + std::to_string(max_bytes >> 20) + " MB";
        return false;
    }

    // Block position of each input entry: column min, row max (permuted).
    entry_pos_.resize(stats_.nnz_input);
    for (Index c = 0; c < n; ++c) {
        for (Index p = col_starts[c]; p < col_starts[c + 1]; ++p) {
            const Index a = iperm_[row_indices[p]];
            const Index b = iperm_[c];
            const Index j = std::min(a, b);
            const Index i = std::max(a, b);
            const Index s = col_snode_[j];
            const auto rows_begin =
                snode_rows_.begin() + static_cast<std::ptrdiff_t>(snode_row_ptr_[s]);
            const auto rows_end =
                snode_rows_.begin() + static_cast<std::ptrdiff_t>(snode_row_ptr_[s + 1]);
            const auto it = std::lower_bound(rows_begin, rows_end, i);
            const auto nrows = static_cast<std::size_t>(rows_end - rows_begin);
            entry_pos_[p] = snode_val_ptr_[s] +
                            static_cast<std::size_t>(j - snode_start_[s]) * nrows +
                            static_cast<std::size_t>(it - rows_begin);
        }
    }

    values_.assign(stats_.stored, 0.0);
    orig_diag_.assign(static_cast<size_t>(n), 0.0);
    relpos_.assign(static_cast<size_t>(n), 0);
    std::size_t max_rows = 0;
    for (Index s = 0; s < ns; ++s) {
        max_rows = std::max(max_rows, snode_row_ptr_[s + 1] - snode_row_ptr_[s]);
    }
    update_.assign(max_rows, 0.0);

    stats_.analyze_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

std::size_t SparseCholesky::factorBytes() const {
    return stats_.stored * sizeof(Real) + snode_rows_.size() * sizeof(Index) +
           stats_.nnz_input * sizeof(std::size_t) +
           static_cast<std::size_t>(n_) * (3 * sizeof(Index) + 2 * sizeof(Real));
}

bool SparseCholesky::factorize(std::span<const Real> values, const PivotRule& rule,
                               const Deadline& deadline) {
    error_.clear();
    std::fill(values_.begin(), values_.end(), 0.0);
    for (std::size_t p = 0; p < entry_pos_.size(); ++p) {
        values_[entry_pos_[p]] += values[p];
    }
    const auto ns = static_cast<Index>(snode_start_.size()) - 1;
    for (Index s = 0; s < ns; ++s) {
        const Index first = snode_start_[s];
        const Index nc = snode_start_[s + 1] - first;
        const std::size_t ld = snode_row_ptr_[s + 1] - snode_row_ptr_[s];
        for (Index t = 0; t < nc; ++t) {
            orig_diag_[first + t] =
                values_[snode_val_ptr_[s] + static_cast<std::size_t>(t) * (ld + 1)];
        }
    }
    num_dependent_ = 0;
    shifted_.clear();
    const Blas* kernels = blas();

    for (Index s = 0; s < ns; ++s) {
        const Index first = snode_start_[s];
        const Index nc = snode_start_[s + 1] - first;
        const auto ld = static_cast<Index>(snode_row_ptr_[s + 1] - snode_row_ptr_[s]);
        Real* blk = values_.data() + snode_val_ptr_[s];
        const Index* rows = snode_rows_.data() + snode_row_ptr_[s];
        auto at = [&](Index r, Index c) { return blk + r + static_cast<std::ptrdiff_t>(c) * ld; };

        // Blocked dense Cholesky of the supernode's columns over all its
        // rows: an unblocked panel on the diagonal (where the pivot rule
        // applies), a triangular solve for the panel's rows below, and a
        // rank-k update of the columns right of the panel.
        for (Index k0 = 0; k0 < nc; k0 += kPanel) {
            // Once per panel, so a large supernode is not one unbroken block
            // of work (#275).
            if (deadline.passed()) {
                error_ = deadline.stopped() ? "stopped in the factorization"
                                            : "time limit reached in the factorization";
                return false;
            }
            const Index k1 = std::min(nc, k0 + kPanel);
            for (Index k = k0; k < k1; ++k) {
                Real* colk = at(0, k);
                Real d = colk[k];
                const Real d0 = orig_diag_[first + k];
                const Index row = perm_[first + k];
                const Real shift = rule.dependent_shift.empty()
                                       ? 0.0
                                       : rule.dependent_shift[static_cast<size_t>(row)];
                Real bar = rule.dependent_tol * (d0 - rule.diag_offset);
                if (shift > 0.0) {
                    const Real ref = rule.shift_reference.empty()
                                         ? d0 - rule.diag_offset
                                         : rule.shift_reference[static_cast<size_t>(row)];
                    bar = std::max(bar, rule.shift_tol * ref);
                }
                if (d - rule.diag_offset <= bar || d < 1e-30) {
                    if (shift > 0.0 && std::max(d, 0.0) + shift >= 1e-30) {
                        // Replacing the pivot d by p is adding p - d to
                        // M's diagonal at this row before elimination.
                        const Real p = std::max(d, 0.0) + shift;
                        shifted_.push_back({row, p - d});
                        d = p;
                    } else {
                        d = rule.infinite_pivot;
                        ++num_dependent_;
                    }
                }
                const Real lkk = std::sqrt(d);
                if (!std::isfinite(lkk)) {
                    return false;
                }
                colk[k] = lkk;
                const Real inv = 1.0 / lkk;
                for (Index r = k + 1; r < k1; ++r) {
                    colk[r] *= inv;
                }
                for (Index j = k + 1; j < k1; ++j) {
                    const Real ljk = colk[j];
                    if (ljk == 0.0) {
                        continue;
                    }
                    Real* colj = at(0, j);
                    for (Index r = j; r < k1; ++r) {
                        colj[r] -= ljk * colk[r];
                    }
                }
            }
            trsmRightLowerTrans(kernels, ld - k1, k1 - k0, at(k0, k0), ld, at(k1, k0), ld);
            syrkMinusLN(kernels, nc - k1, k1 - k0, at(k1, k0), ld, at(k1, k1), ld);
            gemmMinusNT(kernels, ld - nc, nc - k1, k1 - k0, at(nc, k0), ld, at(k1, k0), ld,
                        at(nc, k1), ld);
        }

        // Subtract L21·L21ᵀ from the supernodes the rows below belong to,
        // one target supernode (a run of consecutive rows) at a time.
        Index r0 = nc;
        while (r0 < ld) {
            const Index tgt = col_snode_[rows[r0]];
            const Index tfirst = snode_start_[tgt];
            const Index tlast = snode_start_[tgt + 1] - 1;
            Index r1 = r0;
            while (r1 < ld && rows[r1] <= tlast) {
                ++r1;
            }
            // Position of each row r0.. in the target's row list: its own
            // columns sit at row - tfirst; rows below them are merged in.
            const auto tld =
                static_cast<std::size_t>(snode_row_ptr_[tgt + 1] - snode_row_ptr_[tgt]);
            const Index* trows = snode_rows_.data() + snode_row_ptr_[tgt];
            Index q = tlast - tfirst + 1;
            for (Index r = r0; r < ld; ++r) {
                const Index row = rows[r];
                if (row <= tlast) {
                    relpos_[r] = row - tfirst;
                } else {
                    while (trows[q] != row) {
                        ++q;
                    }
                    relpos_[r] = q;
                }
            }
            Real* tblk = values_.data() + snode_val_ptr_[tgt];
            // W = L21[c0:ld, :]·L21[c0:c1, :]ᵀ in column chunks [c0, c1) that
            // keep W within the scratch, then scatter W's lower part.
            const auto width = static_cast<Index>(
                std::max<std::size_t>(1, kMaxUpdateEntries / static_cast<std::size_t>(ld - r0)));
            for (Index c0 = r0; c0 < r1; c0 += width) {
                const Index c1 = std::min(r1, c0 + width);
                const Index wm = ld - c0;
                const Index wn = c1 - c0;
                const auto wsize = static_cast<std::size_t>(wm) * static_cast<std::size_t>(wn);
                if (update_.size() < wsize) {
                    update_.resize(wsize);
                }
                std::fill_n(update_.begin(), wsize, 0.0);
                // update_ = 0 - A·Bᵀ: the negated update, added below.
                gemmMinusNT(kernels, wm, wn, nc, at(c0, 0), ld, at(c0, 0), ld, update_.data(), wm);
                for (Index cc = c0; cc < c1; ++cc) {
                    Real* tcol = tblk + static_cast<std::size_t>(rows[cc] - tfirst) * tld;
                    const Real* w = update_.data() + static_cast<std::size_t>(cc - c0) * wm;
                    for (Index r = cc; r < ld; ++r) {
                        tcol[relpos_[r]] += w[r - c0];
                    }
                }
            }
            r0 = r1;
        }
    }
    return true;
}

void SparseCholesky::solve(std::span<Real> b) const {
    const Index n = n_;
    std::vector<Real> x(static_cast<size_t>(n));
    for (Index k = 0; k < n; ++k) {
        x[k] = b[perm_[k]];
    }
    const auto ns = static_cast<Index>(snode_start_.size()) - 1;
    for (Index s = 0; s < ns; ++s) {
        const Index first = snode_start_[s];
        const auto nc = static_cast<std::size_t>(snode_start_[s + 1] - first);
        const std::size_t ld = snode_row_ptr_[s + 1] - snode_row_ptr_[s];
        const Real* blk = values_.data() + snode_val_ptr_[s];
        const Index* rows = snode_rows_.data() + snode_row_ptr_[s];
        for (std::size_t k = 0; k < nc; ++k) {
            const Real* colk = blk + k * ld;
            const Real xk = x[rows[k]] / colk[k];
            x[rows[k]] = xk;
            for (std::size_t r = k + 1; r < ld; ++r) {
                x[rows[r]] -= colk[r] * xk;
            }
        }
    }
    for (Index s = ns - 1; s >= 0; --s) {
        const Index first = snode_start_[s];
        const auto nc = static_cast<std::size_t>(snode_start_[s + 1] - first);
        const std::size_t ld = snode_row_ptr_[s + 1] - snode_row_ptr_[s];
        const Real* blk = values_.data() + snode_val_ptr_[s];
        const Index* rows = snode_rows_.data() + snode_row_ptr_[s];
        for (std::size_t k = nc; k-- > 0;) {
            const Real* colk = blk + k * ld;
            Real sum = x[rows[k]];
            for (std::size_t r = k + 1; r < ld; ++r) {
                sum -= colk[r] * x[rows[r]];
            }
            x[rows[k]] = sum / colk[k];
        }
    }
    for (Index k = 0; k < n; ++k) {
        b[perm_[k]] = x[k];
    }
}

}  // namespace mipx
