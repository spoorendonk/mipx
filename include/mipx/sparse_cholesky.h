#pragma once

#include "mipx/core.h"
#include "mipx/deadline.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mipx {

/// Sparse supernodal Cholesky factorization P·M·Pᵀ = L·Lᵀ of a symmetric
/// positive (semi)definite matrix M, for the barrier's normal equations (#290).
///
/// analyze() runs once per pattern: it takes a fill-reducing permutation from
/// the caller, composes it with a postorder of the elimination tree, computes
/// column counts (row subtrees, Davis 2006 sec. 4.4), forms fundamental
/// supernodes with relaxed amalgamation, and lays out each supernode as a
/// dense column-major block of its rows × its columns. factorize() then runs
/// per numeric matrix: it scatters M into the blocks and factors them supernode
/// by supernode, right-looking: factor a block's columns (blocked, BLAS-3 when
/// a BLAS loads), then subtract its update from the blocks of the
/// supernodes its rows belong to.
///
/// The pattern is the symmetric matrix's entries (row, col) in its original
/// numbering, each unordered pair once, from either triangle; the values given
/// to factorize() are aligned with that entry list. Repeated entries are summed.
///
/// A pivot that cancels to (near) zero relative to its diagonal in M is a row
/// dependent on earlier ones. It is set huge rather than perturbed small, so a
/// solve leaves that component at zero (the "Cholesky-infinity" rule of PCx and
/// Wright's primal-dual book).
class SparseCholesky {
public:
    struct Stats {
        Index n = 0;
        std::size_t nnz_input = 0;  // entries of the pattern (each pair once)
        std::size_t nnz_l = 0;      // nonzeros of L counted by column counts, diagonal included
        double factor_flops = 0.0;  // sum of squared column counts: the factorization's flops
        std::size_t stored = 0;     // reals in the supernode blocks (nnz_l plus relaxation zeros
                                    // and the upper halves of the diagonal blocks)
        Index num_supernodes = 0;
        Index max_supernode_cols = 0;
        double analyze_seconds = 0.0;
    };

    struct PivotRule {
        /// A pivot at or below dependent_tol times its original diagonal, both
        /// net of diag_offset, is dependent.
        Real dependent_tol = 1e-13;
        /// Regularization already on M's diagonal, taken out of the test.
        Real diag_offset = 0.0;
        Real infinite_pivot = 1e128;
        /// Per row, original numbering; empty: none. A dependent pivot on a
        /// row with dependent_shift > 0 becomes max(pivot, 0) + shift rather
        /// than infinite_pivot, and is listed in shiftedPivots(): the caller
        /// that compensates for it (a Schur complement) gets M's solve back.
        std::span<const Real> dependent_shift;
        /// On a row with dependent_shift > 0, the dependent test uses
        /// max(dependent_tol, shift_tol): the caller takes the shift back out
        /// exactly, so a small but nonzero pivot is shifted too rather than
        /// left to make the factor ill-conditioned (Andersen 1996).
        Real shift_tol = 0.0;
        /// Per row, original numbering, net of diag_offset; empty: the
        /// original diagonal. On a row with dependent_shift > 0, shift_tol is
        /// measured against this rather than the pivot's own diagonal: with
        /// columns left out of M, a row they carry almost alone has a small
        /// diagonal here but a large one in the full matrix, and its pivot
        /// must be shifted for the compensated solve to stay accurate.
        std::span<const Real> shift_reference;
    };

    /// A dependent pivot moved by PivotRule::dependent_shift: the factor is
    /// that of M + added·e_row·e_rowᵀ.
    struct ShiftedPivot {
        Index row = 0;
        Real added = 0.0;
    };

    /// Symbolic analysis. col_starts/row_indices give the pattern as above
    /// (CSC-style: entries col_starts[j]..col_starts[j+1]-1 have column j).
    /// perm[new] = old is the fill-reducing ordering. Returns false, with
    /// error() set, when the factor would exceed max_bytes (0: no cap) or
    /// cannot be allocated, or when `deadline` passes or its stop flag is
    /// raised (checked between its passes and inside the long ones, #347).
    bool analyze(Index n, std::span<const Index> col_starts, std::span<const Index> row_indices,
                 std::span<const Index> perm, std::size_t max_bytes = 0,
                 const Deadline& deadline = {});

    /// Build the symbolic layout and estimate cost and storage before
    /// allocating numeric entries. A trial over max_bytes fails.
    bool analyzeSymbolic(Index n, std::span<const Index> col_starts,
                         std::span<const Index> row_indices, std::span<const Index> perm,
                         std::size_t max_bytes, const Deadline& deadline);

    /// Finish a successful symbolic trial once, using the same input pattern.
    bool prepareNumeric(std::span<const Index> col_starts, std::span<const Index> row_indices,
                        const Deadline& deadline);

    /// Numeric factorization of the matrix whose entries are `values`, aligned
    /// with the analyzed pattern. Returns false when a pivot is not finite, or
    /// when `deadline` passes or its stop flag is raised (checked before each
    /// panel, and between update kernels and before returning with
    /// lower-triangle BLAS updates enabled; error() says which).
    bool factorize(std::span<const Real> values, const PivotRule& rule,
                   const Deadline& deadline = {});

    /// Solve M·x = b in place (original numbering).
    void solve(std::span<Real> b) const;

    /// Compute only the lower part of wide inter-supernode updates with BLAS.
    void setLowerTriangleUpdates(bool enabled) { lower_triangle_updates_ = enabled; }

    /// Threads of the BLAS that runs the supernode blocks, process-wide, applied
    /// at the next factorize() (OpenBLAS only; a no-op for other BLAS and for
    /// builds without one). Default 1.
    static void setBlasThreads(Int threads);

    /// Whether factorize() may use BLAS (default true). False runs the blocks
    /// on mipx's own loops, as a build without BLAS does. Process-wide.
    static void setBlasEnabled(bool enabled);

    /// Whether the BLAS library the build found loads in this process. The
    /// library is loaded on the first call to this or to factorize(), not at
    /// program start (OpenBLAS reserves its thread pool when it loads).
    static bool blasAvailable();

    [[nodiscard]] const Stats& stats() const { return stats_; }
    [[nodiscard]] Index numDependentPivots() const { return num_dependent_; }
    /// The pivots of the last factorize() moved by dependent_shift; not
    /// counted in numDependentPivots().
    [[nodiscard]] const std::vector<ShiftedPivot>& shiftedPivots() const { return shifted_; }
    [[nodiscard]] const std::string& error() const { return error_; }

    /// Estimated bytes for the numeric factor and its index arrays.
    [[nodiscard]] std::size_t factorBytes() const;

    /// Supernode partition, for tests: supernode s holds the permuted columns
    /// snodeStart()[s] .. snodeStart()[s+1]-1.
    [[nodiscard]] const std::vector<Index>& snodeStart() const { return snode_start_; }

private:
    bool analyzeSymbolicImpl(Index n, std::span<const Index> col_starts,
                             std::span<const Index> row_indices, std::span<const Index> perm,
                             std::size_t max_bytes, const Deadline& deadline);
    bool prepareNumericImpl(std::span<const Index> col_starts, std::span<const Index> row_indices,
                            const Deadline& deadline);
    bool analysisInterrupted(const Deadline& deadline);
    void discardFailedAnalysis();

    bool lower_triangle_updates_ = false;
    Index n_ = 0;
    std::vector<Index> perm_;   // perm_[new] = old (fill-reducing then postorder)
    std::vector<Index> iperm_;  // iperm_[old] = new

    // Supernodes: columns [snode_start_[s], snode_start_[s+1]); rows
    // snode_rows_[snode_row_ptr_[s] .. snode_row_ptr_[s+1]), sorted, the
    // supernode's own columns first; block values at snode_val_ptr_[s],
    // column-major with leading dimension = its row count.
    std::vector<Index> snode_start_;
    std::vector<std::size_t> snode_row_ptr_;
    std::vector<Index> snode_rows_;
    std::vector<std::size_t> snode_val_ptr_;
    std::vector<Index> col_snode_;  // supernode of each permuted column
    std::size_t row_index_count_ = 0;

    std::vector<std::size_t> entry_pos_;  // block position of each input entry
    // The factor's blocks, stored entries long: allocated uninitialized by
    // analyze() and zeroed by factorize() supernode by supernode, inside its
    // deadline checks, so a many-GB factor is not one unbroken fill (#347).
    std::unique_ptr<Real[]> values_;
    std::size_t values_size_ = 0;
    std::vector<Real> orig_diag_;  // M's diagonal before elimination, permuted
    // Numeric preparation requires a successful symbolic analysis.
    bool symbolic_ready_ = false;
    // Factorization requires a successfully prepared numeric layout.
    bool analyzed_ = false;

    // Numeric scratch.
    std::vector<Index> relpos_;
    std::vector<Real> update_;

    Stats stats_{};
    Index num_dependent_ = 0;
    std::vector<ShiftedPivot> shifted_;
    std::string error_;
};

}  // namespace mipx
