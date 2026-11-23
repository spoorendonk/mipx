#include "mipx/amd_ordering.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

// Approximate minimum degree on the quotient graph (Amestoy, Davis & Duff,
// SIAM J. Matrix Anal. Appl. 17(4), 1996), from the paper's algorithm. The
// dense-node rule is SuiteSparse AMD's (its C code; the paper and MC47 have
// none): a node with more than max(16, α·√n) neighbours is ordered last.
//
// The quotient graph lives in one workspace `iw`. Node i's list starts at
// pe[i] and holds len[i] entries: for a variable, its elen[i] adjacent
// elements first, then its adjacent variables; for an element e, the
// variables of Le. Absorbed nodes (elements absorbed into a newer element,
// variables merged into a supervariable or mass-eliminated) carry
// pe = flip(the node that absorbed them). nv[i] is a supervariable's weight
// (0 for a non-principal variable; negative while it sits in the element
// being formed). degree[i] is a variable's approximate external degree, or
// an element's weighted size |Le|. w[e] - wflg is |Le \ Lme| during a pivot
// step, and w[e] = 0 marks an absorbed element.
//
// Each pivot step: (1) take a variable of least approximate degree; (2) form
// the new element Lme from its elements' variables and its own, absorbing
// those elements; (3) for every element adjacent to Lme compute
// |Le \ Lme|; (4) bound each Lme variable's external degree, prune its list,
// absorb elements with Le ⊆ Lme (aggressive absorption) and eliminate those
// adjacent to nothing but me (mass elimination); (5) merge indistinguishable
// variables of Lme (hashing, then comparing lists); (6) put the variables
// back on the degree lists. A supervariable's members are numbered together,
// right after its principal.

namespace mipx {

namespace {

using Pos = std::int64_t;
using Mark = std::int64_t;
constexpr Index kEmpty = -1;

// flip(j) <= -2 for j >= 0, and flip is its own inverse.
constexpr Index flip(Index j) {
    return -j - 2;
}

}  // namespace

// The algorithm is one loop of six steps over shared state; it is kept in
// one function, as in the paper, rather than split along artificial seams.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void amdOrder(Index n, std::span<const Index> col_starts, std::span<const Index> row_indices,
              std::vector<Index>& perm, std::vector<Index>& iperm, const AmdOptions& opts,
              AmdInfo* info) {
    const auto nn = static_cast<std::size_t>(std::max<Index>(n, 0));
    perm.assign(nn, 0);
    iperm.assign(nn, 0);
    AmdInfo stats;
    if (n <= 0) {
        if (info != nullptr) {
            *info = stats;
        }
        return;
    }
    assert(col_starts.size() >= nn + 1);

    // --- Symmetric pattern without the diagonal or duplicates ---
    // Both triangles go into the workspace itself, duplicates are dropped
    // in place, and the workspace then grows into capacity reserved up
    // front: the pattern is never held twice.
    std::vector<Pos> tstart(nn + 1, 0);
    for (Index j = 0; j < n; ++j) {
        for (Index p = col_starts[j]; p < col_starts[j + 1]; ++p) {
            const Index i = row_indices[p];
            assert(i >= 0 && i < n);
            if (i != j) {
                ++tstart[static_cast<std::size_t>(i) + 1];
                ++tstart[static_cast<std::size_t>(j) + 1];
            }
        }
    }
    for (std::size_t i = 0; i < nn; ++i) {
        tstart[i + 1] += tstart[i];
    }
    const double elbow_room = std::max(0.0, opts.elbow_room);
    const auto elbowOf = [&](std::size_t size) {
        return static_cast<std::size_t>(elbow_room * static_cast<double>(size));
    };
    const auto raw = static_cast<std::size_t>(tstart[nn]);
    std::vector<Index> iw;
    iw.reserve(raw + elbowOf(raw) + nn);
    iw.resize(raw);
    {
        std::vector<Pos> fill(tstart.begin(), tstart.end() - 1);
        for (Index j = 0; j < n; ++j) {
            for (Index p = col_starts[j]; p < col_starts[j + 1]; ++p) {
                const Index i = row_indices[p];
                if (i != j) {
                    iw[static_cast<std::size_t>(fill[i]++)] = j;
                    iw[static_cast<std::size_t>(fill[j]++)] = i;
                }
            }
        }
    }

    std::vector<Pos> pe(nn);
    std::vector<Index> len(nn);
    Pos pfree = 0;
    {
        // Lists only shrink, so the write position never passes the read.
        std::vector<Index> seen(nn, kEmpty);
        for (Index i = 0; i < n; ++i) {
            pe[i] = pfree;
            for (Pos p = tstart[i]; p < tstart[i + 1]; ++p) {
                const Index j = iw[static_cast<std::size_t>(p)];
                if (seen[j] != i) {
                    seen[j] = i;
                    iw[static_cast<std::size_t>(pfree++)] = j;
                }
            }
            len[i] = static_cast<Index>(pfree - pe[i]);
        }
    }
    tstart = {};
    // The pattern, n entries (the most one new element needs once the
    // workspace is compacted) and elbow room against frequent compaction
    // (20% by default, as in the paper); within the reserved capacity. The
    // workspace grows only if a compaction ever leaves too little.
    const auto nz = static_cast<std::size_t>(pfree);
    iw.resize(nz + elbowOf(nz) + nn);
    auto iwlen = static_cast<Pos>(iw.size());

    std::vector<Index> elen(nn, 0);
    std::vector<Index> nv(nn, 1);
    std::vector<Index> degree(nn);
    std::vector<Mark> w(nn, 1);
    // Degree lists.
    std::vector<Index> head(nn, kEmpty);
    std::vector<Index> dnext(nn, kEmpty);
    std::vector<Index> dlast(nn, kEmpty);
    // Hash buckets of supervariable detection.
    std::vector<Index> hhead(nn, kEmpty);
    std::vector<Index> hnext(nn, kEmpty);
    std::vector<Index> hval(nn, 0);
    // Members of each supervariable, principal first.
    std::vector<Index> mnext(nn, kEmpty);
    std::vector<Index> mtail(nn);

    const Mark wbig = std::numeric_limits<Mark>::max() - static_cast<Mark>(n);
    auto clearFlag = [&](Mark wflg) {
        if (wflg < 2 || wflg >= wbig) {
            for (Mark& x : w) {
                if (x != 0) {
                    x = 1;
                }
            }
            wflg = 2;
        }
        return wflg;
    };
    auto dinsert = [&](Index i, Index deg) {
        const Index inext = head[deg];
        if (inext != kEmpty) {
            dlast[inext] = i;
        }
        dnext[i] = inext;
        dlast[i] = kEmpty;
        head[deg] = i;
    };
    auto dremove = [&](Index i) {
        const Index ilast = dlast[i];
        const Index inext = dnext[i];
        if (inext != kEmpty) {
            dlast[inext] = ilast;
        }
        if (ilast != kEmpty) {
            dnext[ilast] = inext;
        } else {
            head[degree[i]] = inext;
        }
    };
    auto absorbMembers = [&](Index into, Index from) {
        mnext[mtail[into]] = from;
        mtail[into] = mtail[from];
    };

    // --- Initialization: empty nodes first, dense nodes last ---
    Index dense = n;
    if (opts.dense_alpha >= 0.0) {
        const double d = std::max(16.0, opts.dense_alpha * std::sqrt(static_cast<double>(n)));
        dense = static_cast<Index>(std::min(d, static_cast<double>(n)));
    }
    std::vector<Index> dense_nodes;
    Index nel = 0;
    Index k = 0;
    for (Index i = 0; i < n; ++i) {
        mtail[i] = i;
        degree[i] = len[i];
        if (len[i] == 0) {
            perm[k++] = i;
            elen[i] = flip(1);
            nv[i] = 0;
            ++nel;
            pe[i] = kEmpty;
            w[i] = 0;
        } else if (len[i] > dense) {
            dense_nodes.push_back(i);
            nv[i] = 0;
            elen[i] = kEmpty;
            ++nel;
            pe[i] = kEmpty;
        } else {
            dinsert(i, len[i]);
        }
    }

    Index mindeg = 0;
    Index lemax = 0;
    Mark wflg = clearFlag(0);

    while (nel < n) {
        // --- 1. Pivot of least approximate degree ---
        Index deg = mindeg;
        while (deg < n && head[deg] == kEmpty) {
            ++deg;
        }
        mindeg = deg;
        const Index me = head[deg];
        {
            const Index inext = dnext[me];
            if (inext != kEmpty) {
                dlast[inext] = kEmpty;
            }
            head[deg] = inext;
        }
        const Index elenme = elen[me];
        Index nvpiv = nv[me];
        nel += nvpiv;

        // --- 2. Form the new element Lme ---
        nv[me] = -nvpiv;
        Index degme = 0;
        Pos pme1 = 0;
        Pos pme2 = 0;
        if (elenme == 0) {
            // No elements adjacent: Lme is built in place in me's list.
            pme1 = pe[me];
            pme2 = pme1 - 1;
            for (Pos p = pme1; p <= pme1 + len[me] - 1; ++p) {
                const Index i = iw[static_cast<std::size_t>(p)];
                const Index nvi = nv[i];
                if (nvi > 0) {
                    degme += nvi;
                    nv[i] = -nvi;
                    iw[static_cast<std::size_t>(++pme2)] = i;
                    dremove(i);
                }
            }
        } else {
            // Lme is built at the end of the workspace from the variables of
            // me's elements and of me itself; those elements are absorbed.
            Pos p = pe[me];
            pme1 = pfree;
            const Index slenme = len[me] - elenme;
            for (Index knt1 = 1; knt1 <= elenme + 1; ++knt1) {
                Index e = 0;
                Pos pj = 0;
                Index ln = 0;
                if (knt1 > elenme) {
                    e = me;
                    pj = p;
                    ln = slenme;
                } else {
                    e = iw[static_cast<std::size_t>(p++)];
                    pj = pe[e];
                    ln = len[e];
                }
                for (Index knt2 = 1; knt2 <= ln; ++knt2) {
                    const Index i = iw[static_cast<std::size_t>(pj++)];
                    const Index nvi = nv[i];
                    if (nvi <= 0) {
                        continue;
                    }
                    if (pfree >= iwlen) {
                        // Compact the workspace: save where me and e
                        // resume, tag each live list's first entry with its
                        // owner, slide the live lists down, then the part
                        // of Lme formed so far.
                        ++stats.compactions;
                        pe[me] = p;
                        len[me] -= knt1;
                        if (len[me] == 0) {
                            pe[me] = kEmpty;
                        }
                        pe[e] = pj;
                        len[e] = ln - knt2;
                        if (len[e] == 0) {
                            pe[e] = kEmpty;
                        }
                        for (Index j = 0; j < n; ++j) {
                            const Pos pn = pe[j];
                            if (pn >= 0) {
                                pe[j] = iw[static_cast<std::size_t>(pn)];
                                iw[static_cast<std::size_t>(pn)] = flip(j);
                            }
                        }
                        Pos psrc = 0;
                        Pos pdst = 0;
                        const Pos pend = pme1 - 1;
                        while (psrc <= pend) {
                            const Index j = flip(iw[static_cast<std::size_t>(psrc++)]);
                            if (j >= 0) {
                                iw[static_cast<std::size_t>(pdst)] = static_cast<Index>(pe[j]);
                                pe[j] = pdst++;
                                for (Index knt3 = 0; knt3 <= len[j] - 2; ++knt3) {
                                    iw[static_cast<std::size_t>(pdst++)] =
                                        iw[static_cast<std::size_t>(psrc++)];
                                }
                            }
                        }
                        const Pos p1 = pdst;
                        for (psrc = pme1; psrc <= pfree - 1; ++psrc) {
                            iw[static_cast<std::size_t>(pdst++)] =
                                iw[static_cast<std::size_t>(psrc)];
                        }
                        pme1 = p1;
                        pfree = pdst;
                        pj = pe[e];
                        p = pe[me];
                        if (pfree >= iwlen) {
                            // Not reached: the n entries beyond the pattern
                            // hold any one element after compaction, with or
                            // without elbow room. Kept so a broken invariant
                            // costs memory, not a wrong ordering.
                            iw.resize(iw.size() + iw.size() / 2 + nn);
                            iwlen = static_cast<Pos>(iw.size());
                        }
                    }
                    degme += nvi;
                    nv[i] = -nvi;
                    iw[static_cast<std::size_t>(pfree++)] = i;
                    dremove(i);
                }
                if (e != me) {
                    pe[e] = flip(me);
                    w[e] = 0;
                }
            }
            pme2 = pfree - 1;
        }
        degree[me] = degme;
        pe[me] = pme1;
        len[me] = static_cast<Index>(pme2 - pme1 + 1);
        elen[me] = flip(nvpiv + degme);
        wflg = clearFlag(wflg);

        // --- 3. |Le \ Lme| for every element adjacent to Lme ---
        for (Pos pme = pme1; pme <= pme2; ++pme) {
            const Index i = iw[static_cast<std::size_t>(pme)];
            const Index eln = elen[i];
            if (eln <= 0) {
                continue;
            }
            const Index nvi = -nv[i];
            const Mark wnvi = wflg - nvi;
            for (Pos p = pe[i]; p <= pe[i] + eln - 1; ++p) {
                const Index e = iw[static_cast<std::size_t>(p)];
                Mark we = w[e];
                if (we >= wflg) {
                    we -= nvi;
                } else if (we != 0) {
                    we = degree[e] + wnvi;
                }
                w[e] = we;
            }
        }

        // --- 4. Degree bounds, list pruning, absorption ---
        for (Pos pme = pme1; pme <= pme2; ++pme) {
            const Index i = iw[static_cast<std::size_t>(pme)];
            const Pos p1 = pe[i];
            const Pos p2 = p1 + elen[i] - 1;
            Pos pn = p1;
            std::uint64_t hash = 0;
            std::int64_t ideg = 0;
            for (Pos p = p1; p <= p2; ++p) {
                const Index e = iw[static_cast<std::size_t>(p)];
                const Mark we = w[e];
                if (we == 0) {
                    continue;  // absorbed
                }
                const Mark dext = we - wflg;
                if (dext > 0 || !opts.aggressive) {
                    ideg += dext;
                    iw[static_cast<std::size_t>(pn++)] = e;
                    hash += static_cast<std::uint64_t>(e);
                } else {
                    // Le ⊆ Lme: e adds nothing beyond me.
                    pe[e] = flip(me);
                    w[e] = 0;
                }
            }
            elen[i] = static_cast<Index>(pn - p1 + 1);  // + me, put first below
            const Pos p3 = pn;
            const Pos p4 = p1 + len[i];
            for (Pos p = p2 + 1; p < p4; ++p) {
                const Index j = iw[static_cast<std::size_t>(p)];
                const Index nvj = nv[j];
                if (nvj > 0) {
                    // Principal, not in Lme: still a direct neighbour.
                    ideg += nvj;
                    iw[static_cast<std::size_t>(pn++)] = j;
                    hash += static_cast<std::uint64_t>(j);
                }
            }
            if (elen[i] == 1 && p3 == pn) {
                // Adjacent to me alone: eliminated together with it.
                pe[i] = flip(me);
                const Index nvi = -nv[i];
                degme -= nvi;
                nvpiv += nvi;
                nel += nvi;
                nv[i] = 0;
                elen[i] = kEmpty;
                absorbMembers(me, i);
            } else {
                degree[i] = static_cast<Index>(std::min<std::int64_t>(degree[i], ideg));
                // me goes first: the first variable moves to the end, the
                // first element to the end of the elements.
                iw[static_cast<std::size_t>(pn)] = iw[static_cast<std::size_t>(p3)];
                iw[static_cast<std::size_t>(p3)] = iw[static_cast<std::size_t>(p1)];
                iw[static_cast<std::size_t>(p1)] = me;
                len[i] = static_cast<Index>(pn - p1 + 1);
                const auto bucket = static_cast<Index>(hash % static_cast<std::uint64_t>(n));
                hnext[i] = hhead[bucket];
                hhead[bucket] = i;
                hval[i] = bucket;
            }
        }
        degree[me] = degme;
        lemax = std::max(lemax, degme);
        wflg += lemax;
        wflg = clearFlag(wflg);

        // --- 5. Supervariables: equal lists within a hash bucket ---
        for (Pos pme = pme1; pme <= pme2; ++pme) {
            Index i = iw[static_cast<std::size_t>(pme)];
            if (nv[i] >= 0) {
                continue;  // mass-eliminated, or merged already
            }
            const Index bucket = hval[i];
            i = hhead[bucket];
            hhead[bucket] = kEmpty;
            while (i != kEmpty && hnext[i] != kEmpty) {
                const Index ln = len[i];
                const Index eln = elen[i];
                for (Pos p = pe[i] + 1; p <= pe[i] + ln - 1; ++p) {
                    w[iw[static_cast<std::size_t>(p)]] = wflg;
                }
                Index jlast = i;
                Index j = hnext[i];
                while (j != kEmpty) {
                    bool ok = len[j] == ln && elen[j] == eln;
                    for (Pos p = pe[j] + 1; ok && p <= pe[j] + ln - 1; ++p) {
                        if (w[iw[static_cast<std::size_t>(p)]] != wflg) {
                            ok = false;
                        }
                    }
                    if (ok) {
                        pe[j] = flip(i);
                        nv[i] += nv[j];
                        nv[j] = 0;
                        elen[j] = kEmpty;
                        absorbMembers(i, j);
                        ++stats.merges;
                        j = hnext[j];
                        hnext[jlast] = j;
                    } else {
                        jlast = j;
                        j = hnext[j];
                    }
                }
                ++wflg;
                i = hnext[i];
            }
        }

        // --- 6. Back on the degree lists; Lme keeps its principals ---
        Pos p = pme1;
        const Index nleft = n - nel;
        for (Pos pme = pme1; pme <= pme2; ++pme) {
            const Index i = iw[static_cast<std::size_t>(pme)];
            const Index nvi = -nv[i];
            if (nvi <= 0) {
                continue;
            }
            nv[i] = nvi;
            const Index d = std::min(degree[i] + degme - nvi, nleft - nvi);
            dinsert(i, d);
            mindeg = std::min(mindeg, d);
            degree[i] = d;
            iw[static_cast<std::size_t>(p++)] = i;
        }
        nv[me] = nvpiv;
        len[me] = static_cast<Index>(p - pe[me]);
        if (len[me] == 0) {
            pe[me] = kEmpty;
            w[me] = 0;
        }
        if (elenme != 0) {
            pfree = p;
        }
        for (Index v = me; v != kEmpty; v = mnext[v]) {
            perm[k++] = v;
        }
    }

    for (const Index i : dense_nodes) {
        perm[k++] = i;
    }
    assert(k == n);
    iperm.assign(nn, kEmpty);
    for (Index q = 0; q < n; ++q) {
        assert(iperm[perm[q]] == kEmpty);  // each node numbered once
        iperm[perm[q]] = q;
    }
    stats.dense = static_cast<Index>(dense_nodes.size());
    if (info != nullptr) {
        *info = stats;
    }
}

}  // namespace mipx
