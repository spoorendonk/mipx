#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>

#include "mipx/lp_solver.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// MockLpSolver — trivial implementation of the abstract interface.
// ---------------------------------------------------------------------------
class MockLpSolver : public LpSolver {
public:
    void load(const LpProblem& problem) override {
        num_cols_ = problem.num_cols;
        num_rows_ = problem.num_rows;
        status_ = Status::Error;  // not yet solved
        objective_ = 0.0;
        loaded_ = true;
    }

    LpResult solve() override {
        status_ = Status::Optimal;
        objective_ = 42.0;
        iterations_ = 7;
        return {status_, objective_, iterations_};
    }

    Status getStatus() const override { return status_; }
    Real getObjective() const override { return objective_; }

    std::vector<Real> getPrimalValues() const override {
        return std::vector<Real>(static_cast<std::size_t>(num_cols_), 1.0);
    }

    std::vector<Real> getDualValues() const override {
        return std::vector<Real>(static_cast<std::size_t>(num_rows_), 0.5);
    }

    std::vector<Real> getReducedCosts() const override {
        return std::vector<Real>(static_cast<std::size_t>(num_cols_), -0.25);
    }

    std::vector<BasisStatus> getBasis() const override {
        return basis_;
    }

    void setBasis(std::span<const BasisStatus> basis) override {
        basis_.assign(basis.begin(), basis.end());
    }

    void addRows(std::span<const Index> /*starts*/,
                 std::span<const Index> /*indices*/,
                 std::span<const Real> /*values*/,
                 std::span<const Real> lower,
                 std::span<const Real> /*upper*/) override {
        // Just track how many rows were added.
        rows_added_ += static_cast<Index>(lower.size());
        num_rows_ += static_cast<Index>(lower.size());
    }

    void removeRows(std::span<const Index> rows) override {
        rows_removed_ += static_cast<Index>(rows.size());
        num_rows_ -= static_cast<Index>(rows.size());
    }

    void setColBounds(Index col, Real lower, Real upper) override {
        last_bound_col_ = col;
        last_lower_ = lower;
        last_upper_ = upper;
    }

    void setObjective(std::span<const Real> obj) override {
        obj_.assign(obj.begin(), obj.end());
    }

    // Accessors for test inspection.
    bool loaded() const { return loaded_; }
    Index rowsAdded() const { return rows_added_; }
    Index rowsRemoved() const { return rows_removed_; }
    Index lastBoundCol() const { return last_bound_col_; }
    Real lastLower() const { return last_lower_; }
    Real lastUpper() const { return last_upper_; }
    std::span<const Real> storedObjective() const { return obj_; }

private:
    bool loaded_ = false;
    Status status_ = Status::Error;
    Real objective_ = 0.0;
    Int iterations_ = 0;
    Index num_cols_ = 0;
    Index num_rows_ = 0;
    std::vector<BasisStatus> basis_;
    Index rows_added_ = 0;
    Index rows_removed_ = 0;
    Index last_bound_col_ = -1;
    Real last_lower_ = 0.0;
    Real last_upper_ = 0.0;
    std::vector<Real> obj_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("LpSolver: interface can be instantiated via mock", "[lp_solver]") {
    std::unique_ptr<LpSolver> solver = std::make_unique<MockLpSolver>();
    REQUIRE(solver != nullptr);
}

TEST_CASE("LpSolver: load and solve", "[lp_solver]") {
    auto mock = std::make_unique<MockLpSolver>();

    LpProblem prob;
    prob.num_cols = 3;
    prob.num_rows = 2;
    mock->load(prob);
    CHECK(mock->loaded());
    CHECK(mock->getStatus() == Status::Error);  // not yet solved

    LpResult result = mock->solve();
    CHECK(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(42.0, 1e-12));
    CHECK(result.iterations == 7);

    CHECK(mock->getStatus() == Status::Optimal);
    CHECK_THAT(mock->getObjective(), WithinAbs(42.0, 1e-12));
}

TEST_CASE("LpSolver: solution vectors", "[lp_solver]") {
    MockLpSolver mock;

    LpProblem prob;
    prob.num_cols = 4;
    prob.num_rows = 3;
    mock.load(prob);
    mock.solve();

    auto primal = mock.getPrimalValues();
    REQUIRE(primal.size() == 4);
    CHECK_THAT(primal[0], WithinAbs(1.0, 1e-12));

    auto dual = mock.getDualValues();
    REQUIRE(dual.size() == 3);
    CHECK_THAT(dual[0], WithinAbs(0.5, 1e-12));

    auto rc = mock.getReducedCosts();
    REQUIRE(rc.size() == 4);
    CHECK_THAT(rc[0], WithinAbs(-0.25, 1e-12));
}

TEST_CASE("LpSolver: basis operations", "[lp_solver]") {
    MockLpSolver mock;

    std::vector<BasisStatus> basis = {
        BasisStatus::Basic,
        BasisStatus::AtLower,
        BasisStatus::AtUpper,
        BasisStatus::Fixed,
        BasisStatus::Free,
    };
    mock.setBasis(basis);

    auto retrieved = mock.getBasis();
    REQUIRE(retrieved.size() == 5);
    CHECK(retrieved[0] == BasisStatus::Basic);
    CHECK(retrieved[1] == BasisStatus::AtLower);
    CHECK(retrieved[2] == BasisStatus::AtUpper);
    CHECK(retrieved[3] == BasisStatus::Fixed);
    CHECK(retrieved[4] == BasisStatus::Free);
}

TEST_CASE("LpSolver: addRows and removeRows", "[lp_solver]") {
    MockLpSolver mock;

    LpProblem prob;
    prob.num_cols = 2;
    prob.num_rows = 1;
    mock.load(prob);

    // Add 2 rows (CSR style: starts defines row boundaries).
    std::vector<Index> starts = {0, 1, 3};
    std::vector<Index> indices = {0, 0, 1};
    std::vector<Real> values = {1.0, 2.0, 3.0};
    std::vector<Real> lower = {0.0, -kInf};
    std::vector<Real> upper = {10.0, 5.0};

    mock.addRows(starts, indices, values, lower, upper);
    CHECK(mock.rowsAdded() == 2);

    // Remove one row.
    std::vector<Index> to_remove = {0};
    mock.removeRows(to_remove);
    CHECK(mock.rowsRemoved() == 1);
}

TEST_CASE("LpSolver: setColBounds", "[lp_solver]") {
    MockLpSolver mock;

    mock.setColBounds(5, -1.0, 10.0);
    CHECK(mock.lastBoundCol() == 5);
    CHECK_THAT(mock.lastLower(), WithinAbs(-1.0, 1e-12));
    CHECK_THAT(mock.lastUpper(), WithinAbs(10.0, 1e-12));
}

TEST_CASE("LpSolver: setObjective", "[lp_solver]") {
    MockLpSolver mock;

    std::vector<Real> obj = {1.0, -2.0, 3.5};
    mock.setObjective(obj);

    auto stored = mock.storedObjective();
    REQUIRE(stored.size() == 3);
    CHECK_THAT(stored[0], WithinAbs(1.0, 1e-12));
    CHECK_THAT(stored[1], WithinAbs(-2.0, 1e-12));
    CHECK_THAT(stored[2], WithinAbs(3.5, 1e-12));
}

TEST_CASE("LpResult: default construction", "[lp_solver]") {
    LpResult result;
    CHECK(result.status == Status::Error);
    CHECK_THAT(result.objective, WithinAbs(0.0, 1e-12));
    CHECK(result.iterations == 0);
}
