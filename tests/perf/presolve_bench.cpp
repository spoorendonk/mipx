// mipx-presolve-bench: mipx's presolve alone, without the solve, on one LP
// file (#287). Reports its time, the sizes before and after, and what each
// reduction did, as one CSV row (presolve_bench_common.h has the leading
// columns); mipx-presolve-bench-highs writes the same for HiGHS's presolve.
//
// The model is read as an LP (integer columns relaxed; readOrExit). Presolve
// runs with the options mipx-solve's LP path uses (PresolveOptions'
// defaults), --repeats times on a fresh Presolver; `seconds` is the fastest
// call and `seconds_median` the median, reading excluded. The per-reduction
// columns, <rule>_changes and <rule>_seconds, are PresolveStats' counters and
// times of the last call; `loop_seconds` is its time_seconds (its setup and
// reduction loop), the rest of that call's time building the reduced LP.
//
// --check-postsolve solves the reduced LP, postsolves the
// point, and checks it against the model as read: every bound and row to
// 1e-6 (pointHoldsModel) and its objective within --rel-tol (1e-6, relative to
// max(1, |reference|)) of the reference: the instance's =opt= line in --solu
// FILE (an LP optimum), else a solve of the model as read.
// mipx-presolve-check-highs (MIPX_BENCH_HIGHS) solves both LPs with HiGHS
// on one thread.
// A presolve verdict of infeasible, or a reduced LP that is not optimal,
// must match the original solve's status. A solve that hits --time-limit
// is reported as "limit"; a solver error on either side is a fail.
//
// Status: reduced, reduced_to_empty, not_reduced (sizes unchanged, though
// bounds may have been tightened; HiGHS's not_reduced means no reduction at
// all), infeasible (sizes after -1, as HiGHS's), or error (the presolver
// threw; every column after the common ones empty). Exit 0, 1 when the check
// fails or presolve threw, 2 on a usage or read error.

#include "mipx/presolve.h"
#include "presolve_bench_common.h"

#if defined(MIPX_PRESOLVE_BENCH_USE_HIGHS)
#include "presolve_bench_highs_model.h"

#include <simplex/SimplexConst.h>
#else
#include "mipx/dual_simplex.h"
#endif

#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace mipx;
namespace pb = mipx::presolve_bench;

namespace {

const std::string kUsage =
#if defined(MIPX_PRESOLVE_BENCH_USE_HIGHS)
    "Usage: mipx-presolve-check-highs"
#else
    "Usage: mipx-presolve-bench"
#endif
    " <model> [--name NAME] [--format auto|mps|lp] [--repeats N]\n"
    "       [--csv FILE] [--check-postsolve [--solu FILE] [--rel-tol T] [--time-limit S]]\n"
    "       [--parallel-cols] [--free-col-substitution] [--doubleton-fill-in] [--aggregator]\n"
    "       [--zero-cost-singletons] [--extended-dual-fixing] [--single-equation-dual-fixing]\n"
    "       [--dominated-columns] [--dependent-equations]\n";

struct Options {
    pb::CommonOptions common;
    bool check = false;
    bool parallel_cols = false;  ///< PresolveOptions::enable_parallel_cols (#308), off by default.
    /// PresolveOptions::enable_free_col_substitution (#324), off by default.
    bool free_col_substitution = false;
    /// PresolveOptions::enable_doubleton_fill_in (#343), off by default.
    bool doubleton_fill_in = false;
    /// PresolveOptions::enable_aggregator (#348), off by default.
    bool aggregator = false;
    /// PresolveOptions::enable_zero_cost_singletons (#359), off by default.
    bool zero_cost_singletons = false;
    /// PresolveOptions::enable_extended_dual_fixing (#366), off by default.
    bool extended_dual_fixing = false;
    /// PresolveOptions::enable_single_equation_dual_fixing (#367), off by default.
    bool single_equation_dual_fixing = false;
    /// PresolveOptions::enable_dominated_columns (#361), off by default.
    bool dominated_columns = false;
    /// PresolveOptions::enable_dependent_equations (#393), off by default.
    bool dependent_equations = false;
    std::string solu;
    double rel_tol = 1e-6;
    double time_limit = -1.0;
};

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (pb::parseCommon(argc, argv, i, opt.common, kUsage)) {
            continue;
        }
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                pb::usageError(kUsage, arg + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "--check-postsolve") {
            opt.check = true;
        } else if (arg == "--parallel-cols") {
            opt.parallel_cols = true;
        } else if (arg == "--free-col-substitution") {
            opt.free_col_substitution = true;
        } else if (arg == "--doubleton-fill-in") {
            opt.doubleton_fill_in = true;
        } else if (arg == "--aggregator") {
            opt.aggregator = true;
        } else if (arg == "--zero-cost-singletons") {
            opt.zero_cost_singletons = true;
        } else if (arg == "--extended-dual-fixing") {
            opt.extended_dual_fixing = true;
        } else if (arg == "--single-equation-dual-fixing") {
            opt.single_equation_dual_fixing = true;
        } else if (arg == "--dominated-columns") {
            opt.dominated_columns = true;
        } else if (arg == "--dependent-equations") {
            opt.dependent_equations = true;
        } else if (arg == "--solu") {
            opt.solu = value();
        } else if (arg == "--rel-tol") {
            opt.rel_tol = std::atof(value().c_str());
        } else if (arg == "--time-limit") {
            opt.time_limit = std::atof(value().c_str());
        } else {
            pb::usageError(kUsage, "unknown argument " + arg);
        }
    }
    if (opt.common.path.empty()) {
        pb::usageError(kUsage, "no model file");
    }
    if (!opt.solu.empty() && !opt.check) {
        pb::usageError(kUsage, "--solu needs --check-postsolve");
    }
    return opt;
}

/// The instance's value on an "=opt= NAME VALUE" line of a .solu file, if
/// it has one.
std::optional<double> soluOptimum(const std::string& path, const std::string& instance) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "Error: cannot read %s\n", path.c_str());
        std::exit(pb::kExitUsage);
    }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string tag;
        std::string name;
        double value = 0.0;
        if (fields >> tag >> name >> value && tag == "=opt=" && name == instance) {
            return value;
        }
    }
    return std::nullopt;
}

const char* statusName(Status s) {
    switch (s) {
        case Status::Optimal:
            return "optimal";
        case Status::Infeasible:
            return "infeasible";
        case Status::Unbounded:
            return "unbounded";
        case Status::IterLimit:
        case Status::TimeLimit:
        case Status::NodeLimit:
            return "limit";
        case Status::Error:
            break;
    }
    return "error";
}

struct Solve {
    Status status = Status::Error;
    std::vector<Real> x;
};

#if !defined(MIPX_PRESOLVE_BENCH_USE_HIGHS)
Solve solveForCheck(const LpProblem& lp, double time_limit) {
    Solve s;
    if (lp.num_cols == 0 && lp.num_rows == 0) {
        s.status = Status::Optimal;
        return s;
    }
    DualSimplexSolver solver;
    solver.setVerbose(false);
    DualSimplexOptions dopts = solver.getOptions();
    dopts.max_solve_seconds = time_limit;
    solver.setOptions(dopts);
    solver.load(lp);
    s.status = solver.solve().status;
    s.x = solver.getPrimalValues();
    return s;
}

#else
Solve solveForCheck(const LpProblem& lp, double time_limit) {
    Highs solver;
    // HiGHS reports a model without columns as empty without checking its rows.
    if (lp.num_cols == 0) {
        double tolerance = 0.0;
        if (solver.getOptionValue("primal_feasibility_tolerance", tolerance) != HighsStatus::kOk) {
            return {};
        }
        return {pointViolation(lp, {}) <= tolerance ? Status::Optimal : Status::Infeasible, {}};
    }
    if (!pb::widenHighsLimits(solver) ||
        solver.setOptionValue("output_flag", false) != HighsStatus::kOk ||
        solver.setOptionValue("threads", 1) != HighsStatus::kOk ||
        solver.setOptionValue("allow_unbounded_or_infeasible", true) != HighsStatus::kOk ||
        (time_limit >= 0.0 &&
         solver.setOptionValue("time_limit", time_limit) != HighsStatus::kOk)) {
        std::fprintf(stderr, "Error: HiGHS refused an option\n");
        return {};
    }
    if (solver.passModel(pb::toHighs(lp)) == HighsStatus::kError) {
        std::fprintf(stderr, "Error: HiGHS refused the model\n");
        return {};
    }
    HighsStatus run_status = solver.run();
    if (solver.getModelStatus() == HighsModelStatus::kUnboundedOrInfeasible) {
        // HiGHS 1.16 can mask a time limit while resolving this presolve verdict.
        // Resolve it explicitly with primal simplex under the same cumulative clock.
        if (time_limit >= 0.0 && solver.getRunTime() >= time_limit) {
            return {Status::TimeLimit, {}};
        }
        if (solver.setOptionValue("presolve", "off") != HighsStatus::kOk ||
            solver.setOptionValue("solver", "simplex") != HighsStatus::kOk ||
            solver.setOptionValue("simplex_strategy",
                                  static_cast<HighsInt>(kSimplexStrategyPrimal)) !=
                HighsStatus::kOk) {
            std::fprintf(stderr, "Error: HiGHS refused the primal simplex fallback\n");
            return {};
        }
        run_status = solver.run();
    }
    if (run_status == HighsStatus::kError) {
        std::fprintf(stderr, "Error: HiGHS could not solve the model\n");
        return {};
    }
    switch (solver.getModelStatus()) {
        case HighsModelStatus::kOptimal: {
            const HighsSolution& solution = solver.getSolution();
            if (solution.value_valid &&
                solution.col_value.size() == static_cast<std::size_t>(lp.num_cols)) {
                return {Status::Optimal, solution.col_value};
            }
            return {};
        }
        case HighsModelStatus::kInfeasible:
            return {Status::Infeasible, {}};
        case HighsModelStatus::kUnbounded:
            return {Status::Unbounded, {}};
        case HighsModelStatus::kTimeLimit:
            return {Status::TimeLimit, {}};
        case HighsModelStatus::kIterationLimit:
            return {Status::IterLimit, {}};
        default:
            return {};
    }
}
#endif

bool isLimit(Status s) {
    return s == Status::IterLimit || s == Status::TimeLimit || s == Status::NodeLimit;
}

Real objectiveOf(const LpProblem& lp, const std::vector<Real>& x) {
    Real obj = lp.obj_offset;
    for (Index j = 0; j < lp.num_cols; ++j) {
        obj += lp.obj[j] * x[j];
    }
    return obj;
}

/// The check's columns: check (pass, fail, limit), reduced_status,
/// original_status (when the original was solved), violation, objective,
/// reference, rel_err; empty where they do not apply.
struct Check {
    std::string verdict;
    std::string reduced_status;
    std::string original_status;
    std::optional<Real> violation;
    std::optional<Real> objective;
    std::optional<Real> reference;
    std::optional<Real> rel_err;
};

Check checkPostsolve(const Options& opt, const std::string& instance, const LpProblem& lp,
                     const Presolver& presolver, const LpProblem& reduced) {
    Check c;
    std::optional<Solve> original;
    const auto solveOriginal = [&]() -> const Solve& {
        if (!original) {
            original = solveForCheck(lp, opt.time_limit);
            c.original_status = statusName(original->status);
        }
        return *original;
    };
    // A verdict without a point to postsolve must match the original's. A
    // solver error on either side proves nothing, so it fails.
    const auto sameVerdict = [&](Status reduced_status) {
        const Status s = solveOriginal().status;
        if (s == Status::Error || reduced_status == Status::Error) {
            c.verdict = "fail";
        } else if (isLimit(s) || isLimit(reduced_status)) {
            c.verdict = "limit";
        } else {
            c.verdict = s == reduced_status ? "pass" : "fail";
        }
    };

    if (presolver.isInfeasible()) {
        c.reduced_status = "infeasible";
        sameVerdict(Status::Infeasible);
        return c;
    }
    const Solve r = solveForCheck(reduced, opt.time_limit);
    c.reduced_status = statusName(r.status);
    if (r.status != Status::Optimal) {
        sameVerdict(r.status);
        return c;
    }
    const std::vector<Real> x = presolver.postsolve(r.x);
    c.violation = pointViolation(lp, x);
    c.objective = objectiveOf(lp, x);
    std::optional<Real> reference;
    if (!opt.solu.empty()) {
        reference = soluOptimum(opt.solu, instance);
    }
    if (!reference) {
        const Solve& o = solveOriginal();
        if (o.status != Status::Optimal) {
            c.verdict = isLimit(o.status) ? "limit" : "fail";
            return c;
        }
        reference = objectiveOf(lp, o.x);
    }
    c.reference = reference;
    c.rel_err = std::abs(*c.objective - *reference) / std::max<Real>(1.0, std::abs(*reference));
    c.verdict = pointHoldsModel(lp, x) && *c.rel_err <= opt.rel_tol ? "pass" : "fail";
    return c;
}

void addOptional(pb::Row& row, const std::string& column, const std::optional<Real>& v) {
    if (v) {
        row.addReal(column, *v);
    } else {
        row.add(column, std::string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    const std::string instance =
        opt.common.name.empty() ? pb::instanceName(opt.common.path) : opt.common.name;
    bool relaxed = false;
    const LpProblem lp = pb::readOrExit(opt.common, &relaxed);
    if (relaxed && !opt.solu.empty()) {
        // A .solu value of a model with integers is its MIP optimum, not the
        // optimum of the relaxation this checks.
        pb::usageError(kUsage, "--solu with a model that has integer columns");
    }

    std::vector<double> times;
    Presolver presolver;
    LpProblem reduced;
    std::string error;
    for (int rep = 0; rep < opt.common.repeats && error.empty(); ++rep) {
        presolver = Presolver();
        PresolveOptions popts = presolver.options();
        popts.enable_parallel_cols = opt.parallel_cols;
        popts.enable_free_col_substitution = opt.free_col_substitution;
        popts.enable_doubleton_fill_in = opt.doubleton_fill_in;
        popts.enable_aggregator = opt.aggregator;
        popts.enable_zero_cost_singletons = opt.zero_cost_singletons;
        popts.enable_extended_dual_fixing = opt.extended_dual_fixing;
        popts.enable_single_equation_dual_fixing = opt.single_equation_dual_fixing;
        popts.enable_dominated_columns = opt.dominated_columns;
        popts.enable_dependent_equations = opt.dependent_equations;
        presolver.setOptions(popts);
        try {
            // Into a fresh LP, so freeing the last repeat's is not timed.
            LpProblem out;
            times.push_back(pb::secondsOf([&] { out = presolver.presolve(lp); }));
            reduced = std::move(out);
        } catch (const std::exception& e) {
            error = e.what();
        }
    }

    std::string status = "reduced";
    if (!error.empty()) {
        std::fprintf(stderr, "Error: presolve threw: %s\n", error.c_str());
        status = "error";
        times = {std::numeric_limits<double>::quiet_NaN()};
    } else if (presolver.isInfeasible()) {
        status = "infeasible";
    } else if (reduced.num_rows == 0 && reduced.num_cols == 0) {
        status = "reduced_to_empty";
    } else if (reduced.num_rows == lp.num_rows && reduced.num_cols == lp.num_cols &&
               reduced.matrix.numNonzeros() == lp.matrix.numNonzeros()) {
        status = "not_reduced";
    }

    pb::Row row;
    pb::addCommon(row, instance, "mipx", status, times, lp);
    const std::size_t common_columns = row.size();
    if (status == "infeasible") {
        // Half reduced: its sizes are no reduction to compare.
        row.addSizes("after", -1, -1, -1);
    } else {
        row.addSizes("after", reduced);
    }
    const PresolveStats& st = presolver.stats();
    row.addReal("loop_seconds", st.time_seconds);
    row.addInt("rounds", st.rounds);
    row.addInt("vars_removed", st.vars_removed);
    row.addInt("rows_removed", st.rows_removed);
    row.addInt("bounds_tightened", st.bounds_tightened);
    row.addInt("doubleton_fill_in", st.doubleton_fill_in);
    row.addInt("aggregator_fill_in", st.aggregator_fill_in);
    row.addInt("extended_dual_fixing", st.extended_dual_fixing_changes);
    row.addInt("single_equation_dual_fixing", st.single_equation_dual_fixing_changes);
    row.addInt("implied_row_dual_bounds", st.implied_row_dual_bounds);
    row.addReal("dual_bounds_seconds", st.dual_bounds_seconds);
    row.addInt("weakly_dominated_col", st.weakly_dominated_col_changes);
    row.addInt("forcing_col", st.forcing_col_changes);
    row.addInt("dependent_equations_fictitious", st.dependent_equation_fictitious);
    row.addInt("dependent_equations_timeouts", st.dependent_equation_timeouts);
    for (const PresolveRuleStat& rule : presolveRuleStats()) {
        row.addInt(std::string(rule.name) + "_changes", st.*rule.changes);
    }
    for (const PresolveRuleStat& rule : presolveRuleStats()) {
        row.addReal(std::string(rule.name) + "_seconds", st.*rule.seconds);
    }

    int exit_code = pb::kExitOk;
    if (opt.check) {
        const Check c =
            error.empty() ? checkPostsolve(opt, instance, lp, presolver, reduced) : Check{};
        row.add("check", c.verdict);
        row.add("reduced_status", c.reduced_status);
        row.add("original_status", c.original_status);
        addOptional(row, "violation", c.violation);
        addOptional(row, "objective", c.objective);
        addOptional(row, "reference", c.reference);
        addOptional(row, "rel_err", c.rel_err);
        if (c.verdict == "fail") {
            exit_code = pb::kExitCheckFailed;
        }
    }
    if (!error.empty()) {
        // The same columns as any other run's, so it joins one CSV file.
        row.blankFrom(common_columns);
        exit_code = pb::kExitCheckFailed;
    }
    row.emit(opt.common.csv);
    return exit_code;
}
