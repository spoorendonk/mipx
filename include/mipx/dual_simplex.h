#pragma once

#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/lp_solver.h"
#include "mipx/lu.h"
#include "mipx/sparse_matrix.h"

namespace mipx {

class DualSimplexSolver : public LpSolver {
public:
    DualSimplexSolver() = default;

    void load(const LpProblem& problem) override;
    LpResult solve() override;

    Status getStatus() const override { return status_; }
    Real getObjective() const override;

    std::vector<Real> getPrimalValues() const override;
    std::vector<Real> getDualValues() const override;
    std::vector<Real> getReducedCosts() const override;

    std::vector<BasisStatus> getBasis() const override;
    void setBasis(std::span<const BasisStatus> basis) override;

    void addRows(std::span<const Index> starts,
                 std::span<const Index> indices,
                 std::span<const Real> values,
                 std::span<const Real> lower,
                 std::span<const Real> upper) override;
    void removeRows(std::span<const Index> rows) override;

    void setColBounds(Index col, Real lower, Real upper) override;
    void setObjective(std::span<const Real> obj) override;

    void setIterationLimit(Int limit) { iter_limit_ = limit; }

private:
    // Total number of variables = num_cols (structural) + num_rows (slacks).
    Index numVars() const { return num_cols_ + num_rows_; }

    // Build the augmented matrix [A | I].
    void buildAugmentedMatrix();

    // Set up initial basis (all slacks basic).
    void setupInitialBasis();

    // Compute primal values for basic variables: x_B = B^{-1} * b.
    void computePrimals();

    // Compute dual values and reduced costs.
    void computeDuals();

    // Refactorize the basis.
    void refactorize();

    // Scaling.
    void computeScaling();
    void applyScaling();
    void unscaleResults();

    // Variable bounds (structural + slack).
    Real varLower(Index k) const;
    Real varUpper(Index k) const;
    Real varCost(Index k) const;

    // Problem data (internal, after sign flip for maximize).
    Index num_cols_ = 0;
    Index num_rows_ = 0;
    Sense sense_ = Sense::Minimize;
    Real obj_offset_ = 0.0;
    std::vector<Real> obj_;           // size num_cols
    std::vector<Real> col_lower_;     // size num_cols
    std::vector<Real> col_upper_;     // size num_cols
    std::vector<Real> row_lower_;     // size num_rows
    std::vector<Real> row_upper_;     // size num_rows
    SparseMatrix matrix_{0, 0};       // original constraint matrix
    SparseMatrix augmented_{0, 0};    // [A | I]

    // Scaling factors.
    std::vector<Real> row_scale_;     // size num_rows
    std::vector<Real> col_scale_;     // size num_cols
    bool scaled_ = false;

    // Basis representation.
    // basis_[i] = variable index (0..numVars-1) in basis position i.
    std::vector<Index> basis_;
    // nonbasic_ = list of nonbasic variable indices.
    std::vector<Index> nonbasic_;
    // basis_pos_[k] = position in basis_ if basic, -1 otherwise.
    std::vector<Index> basis_pos_;
    // var_status_[k] = status of variable k.
    std::vector<BasisStatus> var_status_;

    // Solution data.
    std::vector<Real> primal_;        // size numVars
    std::vector<Real> dual_;          // size num_rows
    std::vector<Real> reduced_cost_;  // size numVars

    // LU factorization.
    SparseLU lu_;

    // Solver state.
    Status status_ = Status::Error;
    Int iterations_ = 0;
    Int iter_limit_ = 1000000;
    bool loaded_ = false;

    // Tolerances.
    static constexpr Real kPrimalTol = 1e-7;
    static constexpr Real kDualTol = 1e-7;
    static constexpr Real kPivotTol = 1e-7;
    static constexpr Real kZeroTol = 1e-13;
    static constexpr Int kLogFrequency = 200;
};

}  // namespace mipx
