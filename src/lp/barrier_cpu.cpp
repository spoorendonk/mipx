// CPU Newton-step backends for the barrier solver:
//   CpuCholeskySolver       — Normal Equations + dense LL'
//   CpuSparseCholeskySolver — Normal Equations + sparse supernodal LL' (MOSEK style)
//   CpuAugmentedSolver      — Augmented system + sparse LDL' (HiPO/Zanetti-Gondzio)

#include "mipx/amd_ordering.h"
#include "mipx/sparse_cholesky.h"
#include "newton_solver.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace mipx {

namespace {

// ============================================================================
// Nested Dissection ordering
// ============================================================================
// A recursive bisection approach: partition the graph into two halves with a
// small separator; recursively order each half, then number the separator last.
// This gives O(n log n) fill for 2D meshes and good fill for general graphs.
//
// We use a simple multilevel-ish BFS-based bisection (Kernighan-Lin style
// refinement omitted for simplicity; the quality is sufficient for the
// problems encountered in barrier).

namespace nd_detail {

// BFS from seed to partition vertices into two halves.
void bfsBisect(const std::vector<std::vector<Index>>& adj, const std::vector<Index>& nodes,
               std::vector<Index>& part0, std::vector<Index>& part1,
               std::vector<Index>& separator) {
    if (nodes.empty()) {
        return;
    }

    // Map global IDs → local IDs for BFS.
    std::unordered_map<Index, Index> global_to_local;
    global_to_local.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        global_to_local[nodes[i]] = static_cast<Index>(i);
    }
    Index nv = static_cast<Index>(nodes.size());

    // Build local adjacency.
    std::vector<std::vector<Index>> local_adj(static_cast<size_t>(nv));
    for (Index li = 0; li < nv; ++li) {
        Index gi = nodes[li];
        for (Index gj : adj[gi]) {
            auto it = global_to_local.find(gj);
            if (it != global_to_local.end()) {
                local_adj[li].push_back(it->second);
            }
        }
    }

    // BFS from a pseudo-peripheral node.
    auto bfs_farthest = [&](Index start) -> Index {
        std::vector<Index> dist(static_cast<size_t>(nv), -1);
        std::queue<Index> q;
        dist[start] = 0;
        q.push(start);
        Index farthest = start;
        while (!q.empty()) {
            Index u = q.front();
            q.pop();
            for (Index v : local_adj[u]) {
                if (dist[v] < 0) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                    farthest = v;
                }
            }
        }
        return farthest;
    };

    // Find pseudo-peripheral pair.
    Index start = bfs_farthest(0);
    start = bfs_farthest(start);

    // BFS from start to get level sets.
    std::vector<Index> dist(static_cast<size_t>(nv), -1);
    std::queue<Index> q;
    dist[start] = 0;
    q.push(start);
    Index max_dist = 0;
    while (!q.empty()) {
        Index u = q.front();
        q.pop();
        for (Index v : local_adj[u]) {
            if (dist[v] < 0) {
                dist[v] = dist[u] + 1;
                if (dist[v] > max_dist) {
                    max_dist = dist[v];
                }
                q.push(v);
            }
        }
    }

    // Bisect at the median level. Separator = vertices at the median level.
    Index median = max_dist / 2;
    for (Index li = 0; li < nv; ++li) {
        Index gi = nodes[li];
        if (dist[li] < 0) {
            // Disconnected component — put in part0.
            part0.push_back(gi);
        } else if (dist[li] < median) {
            part0.push_back(gi);
        } else if (dist[li] == median) {
            separator.push_back(gi);
        } else {
            part1.push_back(gi);
        }
    }

    // If bisection is very unbalanced, fall back to a simple split.
    if (part0.empty() || part1.empty()) {
        part0.clear();
        part1.clear();
        separator.clear();
        Index half = nv / 2;
        for (Index li = 0; li < nv; ++li) {
            if (li < half) {
                part0.push_back(nodes[li]);
            } else {
                part1.push_back(nodes[li]);
            }
        }
    }
}

// Recursive nested dissection.
void ndRecurse(const std::vector<std::vector<Index>>& adj, const std::vector<Index>& nodes,
               std::vector<Index>& perm, Index& next_slot) {
    if (nodes.size() <= 64) {
        // Base case: for small subgraphs, use local AMD ordering via
        // a simple minimum-degree selection.
        // Build local adjacency as CSC for AMD.
        Index nv = static_cast<Index>(nodes.size());
        std::unordered_map<Index, Index> global_to_local;
        global_to_local.reserve(nodes.size());
        for (Index li = 0; li < nv; ++li) {
            global_to_local[nodes[li]] = li;
        }
        std::vector<Index> local_cptr(static_cast<size_t>(nv + 1), 0);
        std::vector<Index> local_ridx;
        for (Index li = 0; li < nv; ++li) {
            Index gi = nodes[li];
            for (Index gj : adj[gi]) {
                auto it = global_to_local.find(gj);
                if (it != global_to_local.end() && it->second != li) {
                    local_ridx.push_back(it->second);
                }
            }
            local_cptr[li + 1] = static_cast<Index>(local_ridx.size());
        }
        std::vector<Index> local_perm, local_iperm;
        amdOrder(nv, local_cptr, local_ridx, local_perm, local_iperm);
        for (Index k = 0; k < nv; ++k) {
            perm[next_slot++] = nodes[local_perm[k]];
        }
        return;
    }

    std::vector<Index> part0, part1, separator;
    bfsBisect(adj, nodes, part0, part1, separator);

    // Recurse on each partition.
    ndRecurse(adj, part0, perm, next_slot);
    ndRecurse(adj, part1, perm, next_slot);

    // Separator is numbered last.
    for (Index v : separator) {
        perm[next_slot++] = v;
    }
}

}  // namespace nd_detail

void computeNd(Index n, const std::vector<Index>& col_starts, const std::vector<Index>& row_indices,
               std::vector<Index>& perm, std::vector<Index>& iperm) {
    perm.resize(static_cast<size_t>(n));
    iperm.resize(static_cast<size_t>(n));
    if (n == 0) {
        return;
    }

    // Build symmetric adjacency lists (no self-loops).
    std::vector<std::vector<Index>> adj(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = col_starts[j]; p < col_starts[j + 1]; ++p) {
            Index i = row_indices[p];
            if (i != j) {
                adj[j].push_back(i);
                adj[i].push_back(j);
            }
        }
    }
    // Deduplicate.
    for (Index j = 0; j < n; ++j) {
        auto& a = adj[j];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }

    // Identify connected components via BFS.
    std::vector<bool> visited(static_cast<size_t>(n), false);
    std::vector<std::vector<Index>> components;
    for (Index i = 0; i < n; ++i) {
        if (visited[i]) {
            continue;
        }
        components.emplace_back();
        auto& comp = components.back();
        std::queue<Index> q;
        q.push(i);
        visited[i] = true;
        while (!q.empty()) {
            Index u = q.front();
            q.pop();
            comp.push_back(u);
            for (Index v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
    }

    // Apply nested dissection to each connected component.
    Index next_slot = 0;
    for (auto& comp : components) {
        nd_detail::ndRecurse(adj, comp, perm, next_slot);
    }

    for (Index k = 0; k < n; ++k) {
        iperm[perm[k]] = k;
    }
}

// Auto-select ordering based on problem size.
BarrierOrdering autoSelectOrdering(Index n, Index nnz) {
    // For large problems, nested dissection tends to produce less fill.
    // Threshold: n > 5000 and relatively sparse (nnz/n < 20).
    if (n > 5000 && (nnz < 20 * n)) {
        return BarrierOrdering::Nd;
    }
    return BarrierOrdering::Amd;
}

// ============================================================================
// Symbolic factorization
// ============================================================================

struct SymbolicFact {
    Index n = 0;
    std::vector<Index> perm;   // perm[new] = old
    std::vector<Index> iperm;  // iperm[old] = new

    // L in CSC, below-diagonal only.
    std::vector<Index> l_col_ptr;
    std::vector<Index> l_row_idx;
    Index l_nnz = 0;

    // Left-looking helper: for row i, columns k < i with L[i,k] != 0.
    std::vector<std::vector<Index>> l_row_to_cols;
};

// Permute a CSC lower-triangle pattern using iperm.
static void permutePattern(Index n, const std::vector<Index>& orig_cptr,
                           const std::vector<Index>& orig_ridx, const std::vector<Index>& iperm,
                           std::vector<Index>& perm_cptr, std::vector<Index>& perm_ridx) {
    std::vector<std::vector<Index>> cols(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = orig_cptr[j]; p < orig_cptr[j + 1]; ++p) {
            Index i = orig_ridx[p];
            Index ni = iperm[i];
            Index nj = iperm[j];
            if (ni < nj) {
                std::swap(ni, nj);
            }
            cols[nj].push_back(ni);
        }
    }
    perm_cptr.resize(static_cast<size_t>(n + 1));
    perm_ridx.clear();
    perm_cptr[0] = 0;
    for (Index j = 0; j < n; ++j) {
        auto& c = cols[j];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        // Remove self-loop (diagonal) — we store below-diagonal only.
        c.erase(std::remove(c.begin(), c.end(), j), c.end());
        for (Index i : c) {
            perm_ridx.push_back(i);
        }
        perm_cptr[j + 1] = static_cast<Index>(perm_ridx.size());
    }
}

// Run ordering + symbolic Cholesky on a CSC lower-triangle pattern.
static void symbolicAnalyze(Index n, const std::vector<Index>& col_starts,
                            const std::vector<Index>& row_indices, SymbolicFact& sym,
                            BarrierOrdering ordering = BarrierOrdering::Auto,
                            bool verbose = false) {
    sym.n = n;
    if (n == 0) {
        sym.perm.clear();
        sym.iperm.clear();
        sym.l_col_ptr.assign(1, 0);
        sym.l_row_idx.clear();
        sym.l_nnz = 0;
        sym.l_row_to_cols.clear();
        return;
    }

    // Select ordering.
    BarrierOrdering effective = ordering;
    if (effective == BarrierOrdering::Auto || effective == BarrierOrdering::CuDSS) {
        Index nnz = static_cast<Index>(row_indices.size());
        effective = autoSelectOrdering(n, nnz);
    }

    if (effective == BarrierOrdering::Nd) {
        computeNd(n, col_starts, row_indices, sym.perm, sym.iperm);
    } else {
        amdOrder(n, col_starts, row_indices, sym.perm, sym.iperm);
    }

    // Permute pattern.
    std::vector<Index> pcptr, pridx;
    permutePattern(n, col_starts, row_indices, sym.iperm, pcptr, pridx);

    // Build row→column adjacency for etree computation.
    std::vector<std::vector<Index>> row_adj(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = pcptr[j]; p < pcptr[j + 1]; ++p) {
            Index i = pridx[p];
            if (i > j) {
                row_adj[i].push_back(j);
            }
        }
    }

    // Elimination tree.
    std::vector<Index> parent(static_cast<size_t>(n), -1);
    {
        std::vector<Index> ancestor(static_cast<size_t>(n), -1);
        for (Index k = 0; k < n; ++k) {
            for (Index j : row_adj[k]) {
                Index i = j;
                while (i != -1 && i != k) {
                    Index inext = ancestor[i];
                    ancestor[i] = k;
                    if (inext == -1) {
                        parent[i] = k;
                    }
                    i = inext;
                }
            }
        }
    }

    // Symbolic factorization, one row at a time: row k of L is the union of
    // the etree paths from each j with A(k,j) != 0, j < k, up towards k (the
    // row subtree; Liu 1986, cs_ereach). Walking up from a column's own
    // entries instead misses the fill a column inherits from its children,
    // and the numeric LDL' then drops it (#270).
    sym.l_row_to_cols.assign(static_cast<size_t>(n), {});
    std::vector<Index> col_count(static_cast<size_t>(n), 0);
    std::vector<Index> flag(static_cast<size_t>(n), -1);
    for (Index k = 0; k < n; ++k) {
        flag[k] = k;
        for (Index j : row_adj[k]) {
            for (Index c = j; c != -1 && flag[c] != k; c = parent[c]) {
                flag[c] = k;
                sym.l_row_to_cols[k].push_back(c);
                ++col_count[c];
            }
        }
    }

    // Columns in CSC; rows k come in increasing order, so each is sorted.
    sym.l_col_ptr.assign(static_cast<size_t>(n + 1), 0);
    for (Index c = 0; c < n; ++c) {
        sym.l_col_ptr[c + 1] = sym.l_col_ptr[c] + col_count[c];
    }
    sym.l_nnz = sym.l_col_ptr[n];
    sym.l_row_idx.assign(static_cast<size_t>(sym.l_nnz), 0);
    std::vector<Index> next(sym.l_col_ptr.begin(), sym.l_col_ptr.end() - 1);
    for (Index k = 0; k < n; ++k) {
        for (Index c : sym.l_row_to_cols[k]) {
            sym.l_row_idx[next[c]++] = k;
        }
    }

    // Report fill-in quality.
    if (verbose) {
        Index nnz_a = static_cast<Index>(row_indices.size());
        const char* ordering_name = (effective == BarrierOrdering::Nd) ? "ND" : "AMD";
        std::fprintf(stdout, "  Ordering: %s, n=%d, nnz(A)=%d, nnz(L)=%d, fill-in=%.2f\n",
                     ordering_name, n, nnz_a, sym.l_nnz,
                     nnz_a > 0 ? static_cast<double>(sym.l_nnz) / nnz_a : 0.0);
    }
}

// ============================================================================
// Augmented sparsity pattern
// ============================================================================

// Lower-triangle CSC of K = [[-Theta^{-1}, A']; [A, delta*I]]
// Dimension: (n + m) × (n + m).
static void computeAugPattern(const SparseMatrix& A, Index m, Index n,
                              std::vector<Index>& col_starts, std::vector<Index>& row_indices) {
    const Index dim = n + m;
    std::vector<std::vector<Index>> cols(static_cast<size_t>(dim));

    // Block (1,1): diagonal only, rows 0..n-1.
    for (Index j = 0; j < n; ++j) {
        cols[j].push_back(j);
    }

    // Block (2,1): A entries → row n+i, column j (lower triangle since n+i > j).
    for (Index i = 0; i < m; ++i) {
        auto rv = A.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            if (std::abs(rv.values[k]) > 0.0) {
                cols[rv.indices[k]].push_back(n + i);
            }
        }
    }

    // Block (2,2): diagonal only, rows n..n+m-1.
    for (Index i = 0; i < m; ++i) {
        cols[n + i].push_back(n + i);
    }

    // Convert to CSC.
    col_starts.resize(static_cast<size_t>(dim + 1));
    row_indices.clear();
    col_starts[0] = 0;
    for (Index j = 0; j < dim; ++j) {
        auto& c = cols[j];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        for (Index i : c) {
            row_indices.push_back(i);
        }
        col_starts[j + 1] = static_cast<Index>(row_indices.size());
    }
}

// ============================================================================
// Forward / backward solves
// ============================================================================

// LDL' forward solve: L * x = b  (L unit lower triangular).
static void forwardSolveLDL(const SymbolicFact& sym, const std::vector<Real>& l_val,
                            std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = 0; j < n; ++j) {
        Real bj = b[j];
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[sym.l_row_idx[p]] -= l_val[p] * bj;
        }
    }
}

// LDL' backward solve: L' * x = b  (L' unit upper triangular).
static void backwardSolveLDL(const SymbolicFact& sym, const std::vector<Real>& l_val,
                             std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = n - 1; j >= 0; --j) {
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[j] -= l_val[p] * b[sym.l_row_idx[p]];
        }
    }
}

// LDL' diagonal scale: x = D^{-1} * b.
static void diagScaleLDL(const std::vector<Real>& d, std::span<Real> b) {
    for (Index j = 0; j < static_cast<Index>(d.size()); ++j) {
        b[j] /= d[j];
    }
}

// Binary search for L[j,k] in a sorted column of the symbolic factorization.
// Returns index into l_row_idx/l_val, or -1 if not found.
static Index findInColumn(const SymbolicFact& sym, Index k, Index j) {
    auto beg = sym.l_row_idx.begin() + sym.l_col_ptr[k];
    auto end = sym.l_row_idx.begin() + sym.l_col_ptr[k + 1];
    auto it = std::lower_bound(beg, end, j);
    if (it != end && *it == j) {
        return static_cast<Index>(sym.l_col_ptr[k] + (it - beg));
    }
    return -1;
}

// First unsigned number in a one-line file, or nullopt (missing, or "max").
std::optional<unsigned long long> readNumberFile(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return std::nullopt;
    }
    unsigned long long v = 0;
    const bool ok = std::fscanf(f, "%llu", &v) == 1;
    std::fclose(f);
    return ok ? std::optional{v} : std::nullopt;
}

// Headroom left under this process's cgroup v2 memory limit, or nullopt
// without one: in a container or systemd slice the cgroup's OOM killer, not
// the machine's free memory, is what ends an oversized factor.
std::optional<std::size_t> cgroupHeadroom() {
    std::FILE* f = std::fopen("/proc/self/cgroup", "r");
    if (f == nullptr) {
        return std::nullopt;
    }
    char line[4096];
    std::string path;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        // cgroup v2: the single line "0::/path".
        if (std::strncmp(line, "0::", 3) == 0) {
            path = line + 3;
            while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) {
                path.pop_back();
            }
            break;
        }
    }
    std::fclose(f);
    if (path.empty()) {
        return std::nullopt;
    }
    // The limit may sit on any ancestor (a slice above the process's scope):
    // the least headroom on the way up to the root binds.
    std::optional<std::size_t> least;
    while (!path.empty() && path != "/") {
        const std::string dir = "/sys/fs/cgroup" + path;
        const auto max = readNumberFile(dir + "/memory.max");
        const auto current = readNumberFile(dir + "/memory.current");
        if (max && current) {
            const auto h = static_cast<std::size_t>(*max > *current ? *max - *current : 0);
            least = least ? std::min(*least, h) : h;
        }
        path.resize(path.rfind('/'));
    }
    return least;
}

// The cap on a normal-equations factor: BarrierOptions::ne_max_bytes, else
// the memory available now: Linux MemAvailable (which counts reclaimable
// page cache), or free physical pages, bounded by the cgroup's headroom. A
// factor above it ends with a solver error rather than a failed allocation
// or, under overcommit, the OOM killer (#280). 0 when the platform does not
// say.
std::size_t factorMemoryCap(const BarrierOptions& opts) {
    if (opts.ne_max_bytes != 0) {
        return opts.ne_max_bytes;
    }
    std::size_t cap = 0;
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof line, f) != nullptr) {
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                cap = static_cast<std::size_t>(kb) * 1024;
                break;
            }
        }
        std::fclose(f);
    }
    if (cap == 0) {
        const long pages = sysconf(_SC_AVPHYS_PAGES);
        const long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            cap = static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
        }
    }
    if (const auto headroom = cgroupHeadroom()) {
        // At least 1 byte: 0 would read as "no cap".
        const std::size_t h = std::max<std::size_t>(*headroom, 1);
        cap = cap == 0 ? h : std::min(cap, h);
    }
    return cap;
}

std::string megabytes(std::size_t bytes) {
    return std::to_string(bytes >> 20) + " MB";
}

}  // anonymous namespace

// ============================================================================
// Normal equations: M = A*Theta*A' + reg*I, Theta = Z S^{-1}
// ============================================================================
//
// The Newton system shared by both normal-equations backends; they differ in
// how M is formed and factored (factorM) and solved (solveM).

class NormalEquationsSolver : public NewtonSolver {
public:
    bool setup(const SparseMatrix& A, Index m, Index n, const BarrierOptions& opts) override {
        a_ = &A;
        m_ = m;
        n_ = n;
        ir_steps_ = opts.ir_steps;
        theta_.assign(static_cast<size_t>(n), 0.0);
        s_copy_.assign(static_cast<size_t>(n), 0.0);
        // M = sum_k theta_k a_k a_k' costs nnz(a_k)^2 per column k.
        col_nnz_.assign(static_cast<size_t>(n), 0.0);
        a_nnz_ = 0.0;
        for (Index i = 0; i < m; ++i) {
            const auto row = A.row(i);
            for (Index k = 0; k < row.size(); ++k) {
                col_nnz_[row.indices[k]] += 1.0;
            }
            a_nnz_ += static_cast<double>(row.size());
        }
        form_flops_ = 0.0;
        for (const double c : col_nnz_) {
            form_flops_ += c * c;
        }
        return setupFactor(opts);
    }

    bool factorize(std::span<const Real> z, std::span<const Real> s, Real reg) override {
        reg_ = std::max(reg, 1e-12);
        for (Index j = 0; j < n_; ++j) {
            theta_[j] = z[j] / std::max(s[j], 1e-20);
            s_copy_[j] = s[j];
        }
        return factorM();
    }

    bool solveNewton(std::span<const Real> rp, std::span<const Real> rd, std::span<const Real> rc,
                     std::span<Real> dz, std::span<Real> dy, std::span<Real> ds) override {
        // h[j] = rc[j]/s[j] - theta[j]*rd[j]
        std::vector<Real> h(static_cast<size_t>(n_));
        for (Index j = 0; j < n_; ++j) {
            Real sj = std::max(s_copy_[j], 1e-20);
            h[j] = rc[j] / sj - theta_[j] * rd[j];
        }

        // rhs = rp - A*h
        std::vector<Real> ah(static_cast<size_t>(m_), 0.0);
        a_->multiply(h, ah);
        std::vector<Real> rhs(static_cast<size_t>(m_));
        for (Index i = 0; i < m_; ++i) {
            rhs[i] = rp[i] - ah[i];
        }

        std::vector<Real> dy_vec(rhs.begin(), rhs.end());
        solveM(dy_vec);

        // Iterative refinement.
        for (Int ir = 0; ir < ir_steps_; ++ir) {
            std::vector<Real> at_dy_ir(static_cast<size_t>(n_), 0.0);
            a_->multiplyTranspose(dy_vec, at_dy_ir);
            for (Index j = 0; j < n_; ++j) {
                at_dy_ir[j] *= theta_[j];
            }
            std::vector<Real> m_dy(static_cast<size_t>(m_), 0.0);
            a_->multiply(at_dy_ir, m_dy);
            for (Index i = 0; i < m_; ++i) {
                m_dy[i] += reg_ * dy_vec[i];
            }

            std::vector<Real> corr(static_cast<size_t>(m_));
            for (Index i = 0; i < m_; ++i) {
                corr[i] = rhs[i] - m_dy[i];
            }
            solveM(corr);

            for (Index i = 0; i < m_; ++i) {
                dy_vec[i] += corr[i];
            }
        }

        for (Index i = 0; i < m_; ++i) {
            dy[i] = dy_vec[i];
        }

        // dz[j] = h[j] + theta[j] * (A'*dy)[j]
        std::vector<Real> at_dy(static_cast<size_t>(n_), 0.0);
        a_->multiplyTranspose(dy, at_dy);
        for (Index j = 0; j < n_; ++j) {
            dz[j] = h[j] + theta_[j] * at_dy[j];
        }

        // ds = rd - A'*dy
        for (Index j = 0; j < n_; ++j) {
            ds[j] = rd[j] - at_dy[j];
        }

        return true;
    }

protected:
    virtual bool setupFactor(const BarrierOptions& opts) = 0;
    // Form M from theta_ and reg_ and factor it.
    virtual bool factorM() = 0;
    // Overwrite b with M^{-1} b.
    virtual void solveM(std::span<Real> b) const = 0;
    // Flops of factoring M once formed, and of one solveM.
    [[nodiscard]] virtual double factorMFlops() const = 0;
    [[nodiscard]] virtual double solveMFlops() const = 0;

    // The cost estimates of FactorStats: forming and factoring M; and one
    // solveNewton, i.e. 1 + ir_steps solves with M and 2 + 2 ir_steps
    // products with A or A' (2 nnz(A) flops each).
    // Columns in `left_out` are not formed into M (the sparse backend's dense
    // columns).
    void addCostEstimates(FactorStats& st, std::span<const Index> left_out = {}) const {
        const double ir = static_cast<double>(std::max<Int>(ir_steps_, 0));
        double form = form_flops_;
        for (const Index j : left_out) {
            form -= col_nnz_[static_cast<size_t>(j)] * col_nnz_[static_cast<size_t>(j)];
        }
        st.factor_flops = std::max(form, 0.0) + factorMFlops();
        st.solve_flops = (1.0 + ir) * solveMFlops() + (2.0 + 2.0 * ir) * 2.0 * a_nnz_;
    }

    // A pivot that cancels to (near) zero relative to its diagonal marks a
    // row linearly dependent on the earlier ones (or a row whose columns all
    // sit at their bounds). Perturbing it to a tiny positive value makes dy
    // blow up along that row, so it is set huge instead: the solve then leaves
    // that component of dy at zero, which drops the dependent row from the
    // step (the "Cholesky-infinity" rule of PCx and Wright's primal-dual
    // book). M carries reg on its diagonal, so the test is net of reg.
    static constexpr Real kDependentPivotTol = 1e-13;
    static constexpr Real kInfinitePivot = 1e128;

    const SparseMatrix* a_ = nullptr;
    Index m_ = 0, n_ = 0;
    Int ir_steps_ = 2;
    Real reg_ = 1e-8;
    double a_nnz_ = 0.0;
    double form_flops_ = 0.0;
    std::vector<double> col_nnz_;
    std::vector<Real> theta_;
    std::vector<Real> s_copy_;
};

// ============================================================================
// CpuCholeskySolver: Normal Equations + dense LL'
// ============================================================================
//
// Forms M densely each iteration and factors it with a dense Cholesky: m^2
// memory and m^3 time per iteration however sparse A is. The fill-reducing
// orderings selected by BarrierOptions::ordering do not apply here.

class CpuCholeskySolver final : public NormalEquationsSolver {
public:
    [[nodiscard]] FactorStats factorStats() const override {
        FactorStats st;
        addCostEstimates(st);
        return st;
    }

private:
    [[nodiscard]] double factorMFlops() const override {
        const auto m = static_cast<double>(m_);
        return m * m * m / 3.0;
    }
    [[nodiscard]] double solveMFlops() const override {
        const auto m = static_cast<double>(m_);
        return 2.0 * m * m;
    }

    bool setupFactor(const BarrierOptions& opts) override {
        const std::size_t bytes =
            static_cast<std::size_t>(m_) * static_cast<std::size_t>(m_) * sizeof(Real);
        const std::size_t cap = factorMemoryCap(opts);
        if (cap != 0 && bytes > cap) {
            error_ = "dense normal equations need " + megabytes(bytes) + ", above the cap of " +
                     megabytes(cap) + "; the sparse normal-equations backend fits large LPs";
            return false;
        }
        return true;
    }

    // Reference implementation for Netlib-scale problems: forms M densely
    // and computes its Cholesky factor L stored densely (column-major, lower
    // triangle).
    bool factorM() override {
        const Index m = m_;
        ne_dense_.assign(static_cast<size_t>(m) * static_cast<size_t>(m), 0.0);

        // Build dense NE matrix M = A * Theta * A' + reg * I.
        // Column-major storage: M[i,j] = ne_dense_[i + j*m].
        std::vector<Real> tmp(static_cast<size_t>(n_), 0.0);
        std::vector<Real> ne_col(static_cast<size_t>(m_), 0.0);

        for (Index j = 0; j < m; ++j) {
            // Forming M is m products with A, O(m nnz(A)).
            if (deadline_.passed()) {
                return false;
            }
            auto rj = a_->row(j);
            for (Index k = 0; k < rj.size(); ++k) {
                tmp[rj.indices[k]] = theta_[rj.indices[k]] * rj.values[k];
            }
            std::fill(ne_col.begin(), ne_col.end(), 0.0);
            a_->multiply(tmp, ne_col);
            for (Index k = 0; k < rj.size(); ++k) {
                tmp[rj.indices[k]] = 0.0;
            }

            for (Index i = 0; i < m; ++i) {
                ne_dense_[i + j * m] = ne_col[i];
            }
            ne_dense_[j + j * m] += reg_;
        }

        // Dense Cholesky LL' factorization (in-place, lower triangle).
        // L is stored in the lower triangle of ne_dense_.
        std::vector<Real> diag_orig(static_cast<size_t>(m));
        for (Index j = 0; j < m; ++j) {
            diag_orig[j] = ne_dense_[j + j * m];
        }
        for (Index j = 0; j < m; ++j) {
            // Once per column: the factor is O(m^3), minutes at large m
            // (#275).
            if (deadline_.passed()) {
                return false;
            }
            // Diagonal: L[j,j] = sqrt(M[j,j] - sum_{k<j} L[j,k]^2)
            Real sum = 0.0;
            for (Index k = 0; k < j; ++k) {
                Real ljk = ne_dense_[j + k * m];
                sum += ljk * ljk;
            }
            Real diag = ne_dense_[j + j * m] - sum;

            if (diag - reg_ <= kDependentPivotTol * (diag_orig[j] - reg_) || diag < 1e-30) {
                diag = kInfinitePivot;
            }

            ne_dense_[j + j * m] = std::sqrt(diag);
            if (!std::isfinite(ne_dense_[j + j * m]) || ne_dense_[j + j * m] < 1e-30) {
                return false;
            }

            Real inv_diag = 1.0 / ne_dense_[j + j * m];

            // Off-diagonal: L[i,j] = (M[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
            for (Index i = j + 1; i < m; ++i) {
                Real s = 0.0;
                for (Index k = 0; k < j; ++k) {
                    s += ne_dense_[i + k * m] * ne_dense_[j + k * m];
                }
                ne_dense_[i + j * m] = (ne_dense_[i + j * m] - s) * inv_diag;
            }
        }

        return true;
    }

    void solveM(std::span<Real> b) const override {
        const Index m = m_;
        // L * x = b
        for (Index j = 0; j < m; ++j) {
            b[j] /= ne_dense_[j + j * m];
            for (Index i = j + 1; i < m; ++i) {
                b[i] -= ne_dense_[i + j * m] * b[j];
            }
        }
        // L' * x = b
        for (Index j = m - 1; j >= 0; --j) {
            for (Index i = j + 1; i < m; ++i) {
                b[j] -= ne_dense_[i + j * m] * b[i];
            }
            b[j] /= ne_dense_[j + j * m];
        }
    }

    std::vector<Real> ne_dense_;  // Dense NE matrix / Cholesky factor (col-major)
};

// ============================================================================
// CpuSparseCholeskySolver: Normal Equations + sparse supernodal LL'
// ============================================================================
//
// The pattern of M (lower triangle, by row of A) is found once in setup,
// ordered with BarrierOptions::ordering and analyzed by SparseCholesky; each
// iteration forms M's entries on that pattern and refactors (#290).
//
// Dense columns (#314; Andersen 1996, Andersen & Andersen 2000 sec. 4.2): a
// column with k nonzeros puts a k x k dense block into M. The densest columns
// A_d are left out of the factor, which is then of M_s = A_s Theta_s A_s' +
// reg I, and M = M_s + A_d Theta_d A_d' is solved by the Woodbury identity.
// M_s can be singular where M is not (a row only dense columns cover, or one
// whose sparse part depends on other rows): such a pivot of M_s is not set
// infinite, which would drop the row from M too, but shifted by the row's
// diagonal in M, and the shift is taken back out exactly by a column
// sqrt(added) e_i with sign -1 (the modified Schur complement). With
// U = [A_d Theta_d^{1/2} | sqrt(added_i) e_i] and C =
// diag(I, -I), M = M_s' + U C U' and
//   M^{-1} b = M_s'^{-1} (b - U S^{-1} U' M_s'^{-1} b),  S = C + U' M_s'^{-1} U,
// S factored densely, symmetric and without pivoting, dense columns first.
// Its dense block I + U_d' M_s'^{-1} U_d is SPD with pivots >= 1; once that
// is eliminated, the compensation block is -(I + E' M^{-1} E)^{-1} (E the
// sqrt(added_i) e_i, by Woodbury on M + E E'), negative definite, and its
// pivot on row i is -mu_i / (mu_i + added_i), mu_i row i's pivot in M (plus
// the later shifts) with i eliminated last. So a row dependent in M as well
// shows up as a compensation pivot near 0 and is dropped by the
// dependent-pivot rule's own bar, on mu_i. Forming S takes one solve with
// M_s' per column of U, one column at a time, so no m x k block is stored.
// If S would be larger than max_schur_order (many shifted pivots), the dense
// columns go back into the factor, as without this handling; if that factor
// does not fit, they stay out and S may grow to twice the cap.

class CpuSparseCholeskySolver final : public NormalEquationsSolver {
public:
    explicit CpuSparseCholeskySolver(Index max_schur_order) : max_schur_order_(max_schur_order) {}

    [[nodiscard]] FactorStats factorStats() const override {
        FactorStats st{chol_.stats().nnz_l, static_cast<Index>(dense_cols_.size()), schur_order_,
                       schur_dropped_};
        addCostEstimates(st, dense_cols_);
        return st;
    }

private:
    // The Cholesky factor of M_s; the dense columns' Schur complement adds to
    // both, and is left out of the estimate.
    [[nodiscard]] double factorMFlops() const override { return chol_.stats().factor_flops; }
    [[nodiscard]] double solveMFlops() const override {
        return 4.0 * static_cast<double>(chol_.stats().nnz_l);
    }

    // With dense columns out, a pivot of M_s at or below this fraction of the
    // row's diagonal in M (net of reg) is shifted too: a near-dependent row
    // (its pivot left at about reg) would otherwise make M_s' as
    // ill-conditioned as reg is small, and the Woodbury solve would lose those
    // digits. The bar is on M's diagonal, not M_s's: on a row the dense
    // columns carry almost alone, M_s's pivot is tiny next to M's and makes
    // S lose its digits just the same (Netlib brandy, #315).
    static constexpr Real kShiftTol = 1e-6;

    void findDenseColumns(const BarrierOptions& opts) {
        dense_cols_.clear();
        is_dense_.assign(static_cast<size_t>(n_), 0);
        if (opts.ne_max_dense_cols <= 0) {
            return;
        }
        const auto thresh = std::max<Index>(
            16, static_cast<Index>(opts.dense_col_fraction * static_cast<Real>(m_)));
        for (Index j = 0; j < n_; ++j) {
            if (a_->col(j).size() > thresh) {
                dense_cols_.push_back(j);
            }
        }
        std::stable_sort(dense_cols_.begin(), dense_cols_.end(),
                         [&](Index a, Index b) { return a_->col(a).size() > a_->col(b).size(); });
        if (dense_cols_.size() > static_cast<size_t>(opts.ne_max_dense_cols)) {
            dense_cols_.resize(static_cast<size_t>(opts.ne_max_dense_cols));
        }
        std::sort(dense_cols_.begin(), dense_cols_.end());
        for (const Index j : dense_cols_) {
            is_dense_[j] = 1;
        }
    }

    bool setupFactor(const BarrierOptions& opts) override {
        opts_ = opts;
        findDenseColumns(opts);
        return analyzePattern();
    }

    // Pattern of M_s (without the columns in dense_cols_), ordered and
    // analyzed.
    bool analyzePattern() {
        analyzed_ = false;
        const BarrierOptions& opts = opts_;
        const Index m = m_;
        // Pattern of M_s, each pair once: column k of the pattern lists k and
        // the rows i > k sharing a sparse column of A with row k.
        ne_ptr_.assign(static_cast<size_t>(m) + 1, 0);
        ne_idx_.clear();
        std::vector<Index> mark(static_cast<size_t>(m), -1);
        for (Index k = 0; k < m; ++k) {
            mark[k] = k;
            ne_idx_.push_back(k);
            const auto rk = a_->row(k);
            for (Index q = 0; q < rk.size(); ++q) {
                if (is_dense_[rk.indices[q]] != 0) {
                    continue;
                }
                const auto cj = a_->col(rk.indices[q]);
                for (Index p = 0; p < cj.size(); ++p) {
                    const Index i = cj.indices[p];
                    if (i > k && mark[i] != k) {
                        mark[i] = k;
                        ne_idx_.push_back(i);
                    }
                }
            }
            if (ne_idx_.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
                error_ = "normal-equations pattern has more than 2^31 entries";
                return false;
            }
            ne_ptr_[k + 1] = static_cast<Index>(ne_idx_.size());
        }

        BarrierOrdering ordering = opts.ordering;
        if (ordering == BarrierOrdering::Auto || ordering == BarrierOrdering::CuDSS) {
            ordering = autoSelectOrdering(m, ne_ptr_[m]);
        }
        SparseCholesky::setBlasThreads(opts.threads);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Index> perm;
        std::vector<Index> iperm;
        if (ordering == BarrierOrdering::Nd) {
            computeNd(m, ne_ptr_, ne_idx_, perm, iperm);
        } else {
            amdOrder(m, ne_ptr_, ne_idx_, perm, iperm);
        }
        const double order_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // The factor shares the cap with M's pattern and its values (an Index
        // and a Real per entry, held for the solve); 1 byte left when they
        // alone fill it, so analyze refuses rather than reading 0 as no cap.
        std::size_t cap = factorMemoryCap(opts);
        if (cap != 0) {
            const std::size_t ne_bytes =
                ne_idx_.size() * (sizeof(Index) + sizeof(Real)) +
                static_cast<std::size_t>(m) * (sizeof(Index) + sizeof(Real));
            cap = cap > ne_bytes ? cap - ne_bytes : 1;
        }
        const bool analyzed = chol_.analyze(m, ne_ptr_, ne_idx_, perm, cap);
        // Printed also when the factor is refused: its size is the finding.
        if (opts.verbose) {
            const auto& st = chol_.stats();
            std::printf(
                "  NE Cholesky: m=%d dense_cols=%zu nnz(M)=%zu nnz(L)=%zu supernodes=%d "
                "(widest %d) factor=%s ordering=%s %.2fs analysis=%.2fs blas=%s\n",
                m, dense_cols_.size(), st.nnz_input, st.nnz_l, st.num_supernodes,
                st.max_supernode_cols, megabytes(chol_.factorBytes()).c_str(),
                ordering == BarrierOrdering::Nd ? "ND" : "AMD", order_seconds, st.analyze_seconds,
                SparseCholesky::blasAvailable() ? "yes" : "no");
        }
        if (!analyzed) {
            error_ = chol_.error();
            return false;
        }
        ne_val_.assign(ne_idx_.size(), 0.0);
        work_.assign(static_cast<size_t>(m), 0.0);
        if (!dense_cols_.empty()) {
            dense_share_.assign(static_cast<size_t>(m), 0.0);
            shift_.assign(static_cast<size_t>(m), 0.0);
            full_diag_.assign(static_cast<size_t>(m), 0.0);
            ucol_.assign(static_cast<size_t>(m), 0.0);
        }
        analyzed_ = true;
        return true;
    }

    // S would be too large: put the dense columns back into the factor for
    // the rest of the solve. If that factor does not fit, keep them out and
    // let S grow past the cap instead.
    bool keepDenseColumnsIn() {
        std::vector<Index> saved;
        saved.swap(dense_cols_);
        std::fill(is_dense_.begin(), is_dense_.end(), 0);
        schur_order_ = 0;
        if (opts_.verbose) {
            std::printf(
                "  NE Cholesky: dense-column Schur complement of order > %d, %zu dense "
                "columns back into the factor\n",
                max_schur_order_, saved.size());
        }
        if (analyzePattern()) {
            return true;
        }
        if (opts_.verbose) {
            std::printf("  NE Cholesky: that factor does not fit (%s); dense columns kept out\n",
                        error_.c_str());
        }
        dense_cols_ = std::move(saved);
        for (const Index j : dense_cols_) {
            is_dense_[j] = 1;
        }
        schur_uncapped_ = true;
        if (!analyzePattern()) {
            return false;
        }
        error_.clear();
        return true;
    }

    bool factorM() override {
        const Index m = m_;
        if (!analyzed_) {
            return false;  // a failed re-analysis left the arrays out of step
        }
        error_.clear();
        // Column k of M_s: sum over the sparse columns j of row k of
        // theta_j a_kj a_j, gathered on the pattern's rows i >= k.
        for (Index k = 0; k < m; ++k) {
            const auto rk = a_->row(k);
            for (Index q = 0; q < rk.size(); ++q) {
                const Index j = rk.indices[q];
                if (is_dense_[j] != 0) {
                    continue;
                }
                const Real t = theta_[j] * rk.values[q];
                const auto cj = a_->col(j);
                for (Index p = 0; p < cj.size(); ++p) {
                    const Index i = cj.indices[p];
                    if (i >= k) {
                        work_[i] += t * cj.values[p];
                    }
                }
            }
            for (Index p = ne_ptr_[k]; p < ne_ptr_[k + 1]; ++p) {
                const Index i = ne_idx_[p];
                ne_val_[p] = work_[i];
                work_[i] = 0.0;
            }
            ne_val_[ne_ptr_[k]] += reg_;  // the diagonal comes first
        }
        SparseCholesky::PivotRule rule;
        rule.dependent_tol = kDependentPivotTol;
        rule.diag_offset = reg_;
        rule.infinite_pivot = kInfinitePivot;
        schur_order_ = 0;
        schur_dropped_ = 0;
        if (dense_cols_.empty()) {
            return chol_.factorize(ne_val_, rule, deadline_);
        }
        std::fill(dense_share_.begin(), dense_share_.end(), 0.0);
        for (const Index j : dense_cols_) {
            const auto cj = a_->col(j);
            for (Index p = 0; p < cj.size(); ++p) {
                dense_share_[cj.indices[p]] += theta_[j] * cj.values[p] * cj.values[p];
            }
        }
        // Every dependent pivot of M_s is shifted by its row's diagonal in M
        // (net of reg), not only on rows the dense columns reach: a row whose
        // sparse part repeats a reached row's is dependent in M_s alone too.
        // The compensation pivot then tells whether M drops it. Only a row
        // empty in A keeps the infinite rule here.
        for (Index i = 0; i < m; ++i) {
            full_diag_[i] = ne_val_[ne_ptr_[i]] - reg_ + dense_share_[i];
            shift_[i] = std::max(full_diag_[i], 0.0);
        }
        rule.dependent_shift = shift_;
        rule.shift_reference = shift_;  // M's diagonal, net of reg
        rule.shift_tol = kShiftTol;
        if (!chol_.factorize(ne_val_, rule, deadline_)) {
            return false;
        }
        const auto t = static_cast<Index>(dense_cols_.size() + chol_.shiftedPivots().size());
        if (schur_uncapped_ && t > 2 * max_schur_order_) {
            // Past twice the cap S is too large to form (t^2 reals, t^3 work).
            error_ = "dense-column Schur complement of order " + std::to_string(t) +
                     " and a full factor that does not fit";
            return false;
        }
        if (!schur_uncapped_ && t > max_schur_order_) {
            if (!keepDenseColumnsIn()) {
                return false;
            }
            return factorM();
        }
        return factorSchur();
    }

    // Column q of U, into ucol_ (all zero on entry).
    void scatterU(Index q) const {
        const auto k = static_cast<Index>(dense_cols_.size());
        if (q < k) {
            const Index j = dense_cols_[q];
            const Real s = std::sqrt(theta_[j]);
            const auto cj = a_->col(j);
            for (Index p = 0; p < cj.size(); ++p) {
                ucol_[cj.indices[p]] = s * cj.values[p];
            }
        } else {
            const auto& sp = chol_.shiftedPivots()[q - k];
            ucol_[sp.row] = std::sqrt(sp.added);
        }
    }

    // out = U' x.
    void applyUt(std::span<const Real> x, std::span<Real> out) const {
        const auto k = static_cast<Index>(dense_cols_.size());
        for (Index q = 0; q < k; ++q) {
            const Index j = dense_cols_[q];
            const auto cj = a_->col(j);
            Real s = 0.0;
            for (Index p = 0; p < cj.size(); ++p) {
                s += cj.values[p] * x[cj.indices[p]];
            }
            out[q] = std::sqrt(theta_[j]) * s;
        }
        const auto& shifted = chol_.shiftedPivots();
        for (size_t r = 0; r < shifted.size(); ++r) {
            out[k + static_cast<Index>(r)] = std::sqrt(shifted[r].added) * x[shifted[r].row];
        }
    }

    // Form S = C + U' M_s'^{-1} U and factor it.
    bool factorSchur() {
        const auto k = static_cast<Index>(dense_cols_.size());
        const auto& shifted = chol_.shiftedPivots();
        const auto t = k + static_cast<Index>(shifted.size());
        schur_order_ = t;
        const auto tt = static_cast<size_t>(t);
        schur_.assign(tt * tt, 0.0);  // column-major
        schur_rhs_.assign(tt, 0.0);
        // work_ is all zero outside factorM's gather: the solve's vector.
        for (Index q = 0; q < t; ++q) {
            scatterU(q);
            std::swap(ucol_, work_);
            chol_.solve(work_);
            applyUt(work_, std::span<Real>(schur_).subspan(static_cast<size_t>(q) * tt, tt));
            std::fill(work_.begin(), work_.end(), 0.0);
            schur_[static_cast<size_t>(q) * tt + static_cast<size_t>(q)] += q < k ? 1.0 : -1.0;
        }
        // Elimination in place without pivoting (see the class comment for
        // the pivots' signs). A dense pivot is >= 1 in exact arithmetic; one
        // under 1/2 means S lost all its digits to rounding, and the factor
        // fails (the caller retries with more reg). A compensation pivot
        // -mu / (mu + added) at or above the bar where mu, net of reg, is
        // dependent by kDependentPivotTol of M's diagonal drops row i as the
        // full factor's infinite pivot does: an infinite shift there is C's
        // entry -1 made infinite, so C^{-1}'s entry 0 and S's diagonal one
        // larger, which changes this pivot alone, by +1.
        auto at = [&](Index r, Index c) -> Real& {
            return schur_[static_cast<size_t>(c) * tt + static_cast<size_t>(r)];
        };
        schur_dropped_ = 0;
        for (Index c = 0; c < t; ++c) {
            Real& p = at(c, c);
            if (!std::isfinite(p) || (c < k && p < 0.5)) {
                error_ = "dense-column Schur complement lost its accuracy";
                return false;
            }
            if (c >= k) {
                const auto& sp = shifted[static_cast<size_t>(c - k)];
                const Real dep =
                    kDependentPivotTol * full_diag_[static_cast<size_t>(sp.row)] + reg_;
                if (p >= -dep / (dep + sp.added)) {
                    p += 1.0;
                    ++schur_dropped_;
                }
            }
            const Real inv = 1.0 / p;
            for (Index r = c + 1; r < t; ++r) {
                at(r, c) *= inv;
            }
            for (Index cc = c + 1; cc < t; ++cc) {
                const Real u = at(c, cc);
                if (u == 0.0) {
                    continue;
                }
                for (Index r = c + 1; r < t; ++r) {
                    at(r, cc) -= at(r, c) * u;
                }
            }
        }
        return true;
    }

    // schur_rhs_ = S^{-1} schur_rhs_.
    void solveSchur() const {
        const auto t = schur_order_;
        const auto tt = static_cast<size_t>(t);
        auto at = [&](Index r, Index c) {
            return schur_[static_cast<size_t>(c) * tt + static_cast<size_t>(r)];
        };
        auto& w = schur_rhs_;
        for (Index c = 0; c < t; ++c) {
            for (Index r = c + 1; r < t; ++r) {
                w[r] -= at(r, c) * w[c];
            }
        }
        for (Index c = t - 1; c >= 0; --c) {
            w[c] /= at(c, c);
            for (Index r = 0; r < c; ++r) {
                w[r] -= at(r, c) * w[c];
            }
        }
    }

    void solveM(std::span<Real> b) const override {
        chol_.solve(b);
        if (schur_order_ == 0) {
            return;
        }
        applyUt(b, schur_rhs_);
        solveSchur();
        // ucol_ = U w, then b -= M_s'^{-1} U w.
        const auto k = static_cast<Index>(dense_cols_.size());
        for (Index q = 0; q < schur_order_; ++q) {
            const Real w = schur_rhs_[q];
            if (q < k) {
                const Index j = dense_cols_[q];
                const Real s = std::sqrt(theta_[j]) * w;
                const auto cj = a_->col(j);
                for (Index p = 0; p < cj.size(); ++p) {
                    ucol_[cj.indices[p]] += s * cj.values[p];
                }
            } else {
                const auto& sp = chol_.shiftedPivots()[q - k];
                ucol_[sp.row] += std::sqrt(sp.added) * w;
            }
        }
        chol_.solve(ucol_);
        for (Index i = 0; i < m_; ++i) {
            b[i] -= ucol_[i];
        }
        std::fill(ucol_.begin(), ucol_.end(), 0.0);
    }

    std::vector<Index> ne_ptr_;
    std::vector<Index> ne_idx_;
    std::vector<Real> ne_val_;
    std::vector<Real> work_;
    SparseCholesky chol_;

    std::vector<Index> dense_cols_;  // the columns left out of the factor, ascending
    std::vector<char> is_dense_;
    std::vector<Real> dense_share_;  // D_i = sum over dense j of theta_j a_ij^2
    std::vector<Real> shift_;        // the dependent-pivot shift per row
    std::vector<Real> full_diag_;    // M's diagonal net of reg, per row
    BarrierOptions opts_;
    Index max_schur_order_;
    bool schur_uncapped_ = false;  // the full factor did not fit: S may pass the cap
    bool analyzed_ = false;        // pattern, ne_val_ and chol_ agree
    Index schur_order_ = 0;        // k + shifted pivots; 0: no Schur step
    Index schur_dropped_ = 0;      // compensation pivots dropped (rows dependent in M)
    std::vector<Real> schur_;      // LU of S (unit L), column-major
    mutable std::vector<Real> schur_rhs_;
    mutable std::vector<Real> ucol_;  // m-vector scratch, all zero between uses
};

// ============================================================================
// CpuAugmentedSolver: Augmented system + sparse LDL'
// ============================================================================

class CpuAugmentedSolver final : public NewtonSolver {
public:
    bool setup(const SparseMatrix& A, Index m, Index n, const BarrierOptions& opts) override {
        a_ = &A;
        m_ = m;
        n_ = n;
        ir_steps_ = opts.ir_steps;
        dim_ = n + m;

        // Augmented sparsity pattern.
        std::vector<Index> aug_cptr, aug_ridx;
        computeAugPattern(A, m, n, aug_cptr, aug_ridx);

        // Symbolic analysis with selected ordering.
        symbolicAnalyze(dim_, aug_cptr, aug_ridx, sym_, opts.ordering, opts.verbose);

        l_val_.resize(static_cast<size_t>(sym_.l_nnz), 0.0);
        d_.resize(static_cast<size_t>(dim_), 0.0);

        return true;
    }

    bool factorize(std::span<const Real> z, std::span<const Real> s, Real reg) override {
        reg_ = std::max(reg, 1e-12);

        z_copy_.assign(z.begin(), z.end());
        s_copy_.assign(s.begin(), s.end());

        return numericLDL();
    }

    bool solveNewton(std::span<const Real> rp, std::span<const Real> rd, std::span<const Real> rc,
                     std::span<Real> dz, std::span<Real> dy, std::span<Real> ds) override {
        // Form augmented RHS:
        //   rhs_x[j] = rd[j] - rc[j]/z[j]   for j = 0..n-1
        //   rhs_y[i] = rp[i]                  for i = 0..m-1
        std::vector<Real> rhs(static_cast<size_t>(dim_));
        for (Index j = 0; j < n_; ++j) {
            rhs[j] = rd[j] - rc[j] / std::max(z_copy_[j], 1e-20);
        }
        for (Index i = 0; i < m_; ++i) {
            rhs[n_ + i] = rp[i];
        }

        // Permute RHS.
        std::vector<Real> rhs_perm(static_cast<size_t>(dim_));
        for (Index i = 0; i < dim_; ++i) {
            rhs_perm[sym_.iperm[i]] = rhs[i];
        }

        // Solve K * sol = rhs  via  L * D * L' * sol = rhs.
        forwardSolveLDL(sym_, l_val_, rhs_perm);
        diagScaleLDL(d_, rhs_perm);
        backwardSolveLDL(sym_, l_val_, rhs_perm);

        // Iterative refinement.
        for (Int ir = 0; ir < ir_steps_; ++ir) {
            // Unpermute current solution: the inverse of the permute above,
            // so iperm, not perm (perm read a different entry unless the
            // ordering was its own inverse, #270).
            std::vector<Real> sol(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) {
                sol[i] = rhs_perm[sym_.iperm[i]];
            }

            // Compute K * sol.
            std::vector<Real> ksol(static_cast<size_t>(dim_), 0.0);
            // Block (1,1): -Theta^{-1} * sol_x
            for (Index j = 0; j < n_; ++j) {
                ksol[j] = -(s_copy_[j] / std::max(z_copy_[j], 1e-20)) * sol[j];
            }
            // Block (1,2): A' * sol_y
            std::span<const Real> sol_y(sol.data() + n_, static_cast<size_t>(m_));
            std::vector<Real> at_sol_y(static_cast<size_t>(n_), 0.0);
            a_->multiplyTranspose(sol_y, at_sol_y);
            for (Index j = 0; j < n_; ++j) {
                ksol[j] += at_sol_y[j];
            }

            // Block (2,1): A * sol_x
            std::span<const Real> sol_x(sol.data(), static_cast<size_t>(n_));
            std::vector<Real> a_sol_x(static_cast<size_t>(m_), 0.0);
            a_->multiply(sol_x, a_sol_x);
            for (Index i = 0; i < m_; ++i) {
                ksol[n_ + i] = a_sol_x[i] + reg_ * sol[n_ + i];
            }

            // Residual = rhs - K*sol.
            std::vector<Real> resid(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) {
                resid[i] = rhs[i] - ksol[i];
            }

            // Solve correction.
            std::vector<Real> corr(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) {
                corr[sym_.iperm[i]] = resid[i];
            }
            forwardSolveLDL(sym_, l_val_, corr);
            diagScaleLDL(d_, corr);
            backwardSolveLDL(sym_, l_val_, corr);

            for (Index i = 0; i < dim_; ++i) {
                rhs_perm[i] += corr[i];
            }
        }

        // Unpermute and recover.
        std::vector<Real> sol(static_cast<size_t>(dim_));
        for (Index i = 0; i < dim_; ++i) {
            sol[i] = rhs_perm[sym_.iperm[i]];
        }

        // dz = sol[0..n-1], dy = sol[n..n+m-1]
        for (Index j = 0; j < n_; ++j) {
            dz[j] = sol[j];
        }
        for (Index i = 0; i < m_; ++i) {
            dy[i] = sol[n_ + i];
        }

        // ds = (rc - s*dz) / z
        for (Index j = 0; j < n_; ++j) {
            ds[j] = (rc[j] - s_copy_[j] * dz[j]) / std::max(z_copy_[j], 1e-20);
        }

        return true;
    }

private:
    // Left-looking numeric LDL' (unit diagonal L, diagonal D can be negative).
    bool numericLDL() {
        std::fill(l_val_.begin(), l_val_.end(), 0.0);
        std::fill(d_.begin(), d_.end(), 0.0);

        std::vector<Real> w(static_cast<size_t>(dim_), 0.0);

        for (Index j = 0; j < dim_; ++j) {
            // Once per column: fill-in can make this minutes long (#275).
            if (deadline_.passed()) {
                return false;
            }
            // Scatter augmented matrix column j into w.
            scatterAugColumn(j, w);

            // Column modifications: for each k < j with L[j,k] != 0.
            for (Index k : sym_.l_row_to_cols[j]) {
                // Find L[j,k] via binary search (l_row_idx is sorted).
                Index pk = findInColumn(sym_, k, j);
                Real ljk = (pk >= 0) ? l_val_[pk] : 0.0;

                Real ljk_dk = ljk * d_[k];

                // Diagonal: w[j] -= ljk * D[k] * ljk
                w[j] -= ljk_dk * ljk;

                // Off-diagonal: w[i] -= L[i,k] * D[k] * ljk
                for (Index p = sym_.l_col_ptr[k]; p < sym_.l_col_ptr[k + 1]; ++p) {
                    if (sym_.l_row_idx[p] > j) {
                        w[sym_.l_row_idx[p]] -= l_val_[p] * ljk_dk;
                    }
                }
            }

            // Pivot (D[j] = w[j], can be negative for indefinite systems).
            d_[j] = w[j];

            // Per-pivot regularization (HiPO-style).
            constexpr Real eps_s = 1e-12;  // static
            constexpr Real eps_d = 1e-7;   // dynamic boost
            if (std::abs(d_[j]) < eps_d) {
                // Determine sign: variables should be negative, constraints positive.
                Index orig = sym_.perm[j];
                Real sign = (orig < n_) ? -1.0 : 1.0;
                d_[j] = sign * std::max(std::abs(d_[j]) + eps_d, eps_d);
            }
            d_[j] += (d_[j] >= 0 ? eps_s : -eps_s);

            if (std::abs(d_[j]) < 1e-30) {
                return false;
            }

            // Scale off-diagonal.
            Real inv_d = 1.0 / d_[j];
            for (Index p = sym_.l_col_ptr[j]; p < sym_.l_col_ptr[j + 1]; ++p) {
                Index i = sym_.l_row_idx[p];
                l_val_[p] = w[i] * inv_d;
                w[i] = 0.0;
            }
            w[j] = 0.0;
        }

        return true;
    }

    // Scatter one column of the augmented matrix into dense workspace.
    void scatterAugColumn(Index j, std::vector<Real>& w) const {
        Index orig_j = sym_.perm[j];

        if (orig_j < n_) {
            // Variable column: diagonal = -s/z  (i.e., -Theta^{-1}).
            w[j] = -(s_copy_[orig_j] / std::max(z_copy_[orig_j], 1e-20));

            // A entries: for each row i with A[i, orig_j] != 0,
            // augmented row = n_ + i, permuted row = iperm[n_ + i].
            auto cv = a_->col(orig_j);
            for (Index k = 0; k < cv.size(); ++k) {
                Index aug_row = n_ + cv.indices[k];
                Index perm_row = sym_.iperm[aug_row];
                if (perm_row > j) {
                    w[perm_row] = cv.values[k];
                }
                // If perm_row < j, this entry was already handled as an
                // above-diagonal entry in column perm_row.
            }
        } else {
            // Constraint column orig_j = n_ + i: diagonal = reg (δI block),
            // plus A-coupling entries whose variable column was permuted *after*
            // this constraint (perm_row > j). The variable-column branch only
            // scatters couplings with the constraint permuted after the variable
            // (perm_row > j there); the opposite orientation belongs here, and
            // without it the entry is silently dropped from the factorization.
            Index i = orig_j - n_;
            w[j] = reg_;
            auto rv = a_->row(i);
            for (Index k = 0; k < rv.size(); ++k) {
                Index perm_row = sym_.iperm[rv.indices[k]];
                if (perm_row > j) {
                    w[perm_row] = rv.values[k];
                }
            }
        }
    }

    const SparseMatrix* a_ = nullptr;
    Index m_ = 0, n_ = 0, dim_ = 0;
    Int ir_steps_ = 2;
    Real reg_ = 1e-8;

    SymbolicFact sym_;
    std::vector<Real> l_val_;
    std::vector<Real> d_;

    std::vector<Real> z_copy_;
    std::vector<Real> s_copy_;
};

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<NewtonSolver> createCpuCholeskySolver() {
    return std::make_unique<CpuCholeskySolver>();
}

std::unique_ptr<NewtonSolver> createCpuSparseCholeskySolver(Index max_schur_order) {
    return std::make_unique<CpuSparseCholeskySolver>(max_schur_order);
}

std::unique_ptr<NewtonSolver> createCpuAugmentedSolver() {
    return std::make_unique<CpuAugmentedSolver>();
}

}  // namespace mipx
