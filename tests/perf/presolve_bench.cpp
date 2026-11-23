// mipx-presolve-bench: mipx's presolve alone, without the solve, on one LP
// file (#287). Reports its time, the sizes before and after, and what each
// reduction did, as one CSV row (presolve_bench_common.h has the leading
// columns); mipx-presolve-bench-highs writes the same for HiGHS's presolve.
//
// Presolve runs with the options mipx-solve's LP path uses (PresolveOptions'
// defaults), --repeats times on a fresh Presolver; `seconds` is the fastest
// call and `seconds_median` the median, reading excluded. The per-reduction
// columns, <rule>_changes and <rule>_seconds, are PresolveStats' counters and
// times of the last call; `loop_seconds` is its reduction loop, the rest of
// `seconds` building the reduced LP.
//
// --check-postsolve solves the reduced LP with dual simplex, postsolves the
// point, and checks it against the model as read: every bound and row to
// 1e-6 (pointHoldsModel) and its objective within --rel-tol (1e-6, relative to
// max(1, |reference|)) of the reference: the instance's =opt= line in --solu
// FILE, else a dual simplex solve of the model as read. A presolve verdict of
// infeasible, or a reduced LP that is not optimal, must match that solve's
// status. A solve that hits --time-limit is reported as "limit", not a fail.
//
// Status: reduced, reduced_to_empty, not_reduced, infeasible, or error (the
// presolver threw). Exit 0, 1 when the check fails or presolve threw, 2 on a
// usage or read error.

#include "mipx/dual_simplex.h"
#include "mipx/presolve.h"
#include "presolve_bench_common.h"

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
    "Usage: mipx-presolve-bench <model> [--name NAME] [--format auto|mps|lp] [--repeats N]\n"
    "       [--csv FILE] [--check-postsolve [--solu FILE] [--rel-tol T] [--time-limit S]]\n";

struct Options {
    pb::CommonOptions common;
    bool check = false;
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

Solve solveDual(const LpProblem& lp, double time_limit) {
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
            original = solveDual(lp, opt.time_limit);
            c.original_status = statusName(original->status);
        }
        return *original;
    };
    // A verdict without a point to postsolve must match the original's.
    const auto sameVerdict = [&](Status reduced_status) {
        const Status s = solveOriginal().status;
        if (isLimit(s) || isLimit(reduced_status)) {
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
    const Solve r = solveDual(reduced, opt.time_limit);
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

struct Rule {
    const char* name;
    Index PresolveStats::* changes;
    Real PresolveStats::* seconds;
};

// In the order presolve runs them.
constexpr Rule kRules[] = {
    {"fixed_var", &PresolveStats::fixed_var_changes, &PresolveStats::fixed_var_seconds},
    {"singleton_row", &PresolveStats::singleton_row_changes, &PresolveStats::singleton_row_seconds},
    {"singleton_col", &PresolveStats::singleton_col_changes, &PresolveStats::singleton_col_seconds},
    {"doubleton_eq", &PresolveStats::doubleton_eq_changes, &PresolveStats::doubleton_eq_seconds},
    {"forcing_row", &PresolveStats::forcing_row_changes, &PresolveStats::forcing_row_seconds},
    {"dominated_row", &PresolveStats::dominated_row_changes, &PresolveStats::dominated_row_seconds},
    {"duplicate_row", &PresolveStats::duplicate_row_changes, &PresolveStats::duplicate_row_seconds},
    {"parallel_row", &PresolveStats::parallel_row_changes, &PresolveStats::parallel_row_seconds},
    {"implied_equation", &PresolveStats::implied_equation_changes,
     &PresolveStats::implied_equation_seconds},
    {"activity_bound_tightening", &PresolveStats::activity_bound_tightening_changes,
     &PresolveStats::activity_bound_tightening_seconds},
    {"dual_fixing", &PresolveStats::dual_fixing_changes, &PresolveStats::dual_fixing_seconds},
    {"empty_col", &PresolveStats::empty_col_changes, &PresolveStats::empty_col_seconds},
    {"coeff_tightening", &PresolveStats::coeff_tightening_changes,
     &PresolveStats::coeff_tightening_seconds},
    {"probing", &PresolveStats::probing_changes, &PresolveStats::probing_seconds},
};

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
    const LpProblem lp = pb::readOrExit(opt.common);

    std::vector<double> times;
    Presolver presolver;
    LpProblem reduced;
    std::string error;
    for (int rep = 0; rep < opt.common.repeats && error.empty(); ++rep) {
        presolver = Presolver();
        try {
            times.push_back(pb::secondsOf([&] { reduced = presolver.presolve(lp); }));
        } catch (const std::exception& e) {
            error = e.what();
        }
    }

    pb::Row row;
    if (!error.empty()) {
        std::fprintf(stderr, "Error: presolve threw: %s\n", error.c_str());
        pb::addCommon(row, instance, "mipx", "error", {std::numeric_limits<double>::quiet_NaN()},
                      lp);
        row.emit(opt.common.csv);
        return pb::kExitCheckFailed;
    }

    std::string status = "reduced";
    if (presolver.isInfeasible()) {
        status = "infeasible";
    } else if (reduced.num_rows == 0 && reduced.num_cols == 0) {
        status = "reduced_to_empty";
    } else if (reduced.num_rows == lp.num_rows && reduced.num_cols == lp.num_cols &&
               reduced.matrix.numNonzeros() == lp.matrix.numNonzeros()) {
        status = "not_reduced";
    }

    const PresolveStats& st = presolver.stats();
    pb::addCommon(row, instance, "mipx", status, times, lp);
    row.addSizes("after", reduced);
    row.addReal("loop_seconds", st.time_seconds);
    row.addInt("rounds", st.rounds);
    row.addInt("vars_removed", st.vars_removed);
    row.addInt("rows_removed", st.rows_removed);
    row.addInt("bounds_tightened", st.bounds_tightened);
    for (const Rule& rule : kRules) {
        row.addInt(std::string(rule.name) + "_changes", st.*rule.changes);
    }
    for (const Rule& rule : kRules) {
        row.addReal(std::string(rule.name) + "_seconds", st.*rule.seconds);
    }

    int exit_code = pb::kExitOk;
    if (opt.check) {
        const Check c = checkPostsolve(opt, instance, lp, presolver, reduced);
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
    row.emit(opt.common.csv);
    return exit_code;
}
