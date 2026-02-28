#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mipx/heuristic_runtime.h"
#include "mipx/io.h"
#include "mipx/mip_solver.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    using mipx::HeuristicRuntimeMode;
    using mipx::LpProblem;
    using mipx::MipPreRootStats;
    using mipx::MipResult;
    using mipx::MipSolver;
    using mipx::ParallelMode;
    using mipx::RootLpPolicy;
    using mipx::SearchProfile;
    using mipx::Sense;
    using mipx::Status;
    using mipx::VarType;

    m.doc() = "Python bindings for the mipx branch-and-cut solver";

    nb::enum_<Status>(m, "Status")
        .value("Optimal", Status::Optimal)
        .value("Infeasible", Status::Infeasible)
        .value("Unbounded", Status::Unbounded)
        .value("IterLimit", Status::IterLimit)
        .value("TimeLimit", Status::TimeLimit)
        .value("NodeLimit", Status::NodeLimit)
        .value("Error", Status::Error);

    nb::enum_<Sense>(m, "Sense")
        .value("Minimize", Sense::Minimize)
        .value("Maximize", Sense::Maximize);

    nb::enum_<VarType>(m, "VarType")
        .value("Continuous", VarType::Continuous)
        .value("Integer", VarType::Integer)
        .value("Binary", VarType::Binary)
        .value("SemiContinuous", VarType::SemiContinuous)
        .value("SemiInteger", VarType::SemiInteger);

    nb::enum_<RootLpPolicy>(m, "RootLpPolicy")
        .value("DualDefault", RootLpPolicy::DualDefault)
        .value("BarrierRoot", RootLpPolicy::BarrierRoot)
        .value("PdlpRoot", RootLpPolicy::PdlpRoot)
        .value("ConcurrentRootExperimental", RootLpPolicy::ConcurrentRootExperimental);

    nb::enum_<SearchProfile>(m, "SearchProfile")
        .value("Stable", SearchProfile::Stable)
        .value("Default", SearchProfile::Default)
        .value("Aggressive", SearchProfile::Aggressive);

    nb::enum_<ParallelMode>(m, "ParallelMode")
        .value("Deterministic", ParallelMode::Deterministic)
        .value("Opportunistic", ParallelMode::Opportunistic);

    nb::enum_<HeuristicRuntimeMode>(m, "HeuristicRuntimeMode")
        .value("Deterministic", HeuristicRuntimeMode::Deterministic)
        .value("Opportunistic", HeuristicRuntimeMode::Opportunistic);

    nb::class_<LpProblem>(m, "LpProblem")
        .def(nb::init<>())
        .def_rw("name", &LpProblem::name)
        .def_rw("sense", &LpProblem::sense)
        .def_rw("num_cols", &LpProblem::num_cols)
        .def_rw("obj", &LpProblem::obj)
        .def_rw("col_lower", &LpProblem::col_lower)
        .def_rw("col_upper", &LpProblem::col_upper)
        .def_rw("col_type", &LpProblem::col_type)
        .def_rw("col_names", &LpProblem::col_names)
        .def_rw("num_rows", &LpProblem::num_rows)
        .def_rw("row_lower", &LpProblem::row_lower)
        .def_rw("row_upper", &LpProblem::row_upper)
        .def_rw("row_names", &LpProblem::row_names)
        .def_rw("obj_offset", &LpProblem::obj_offset)
        .def("has_integers", &LpProblem::hasIntegers);

    m.def("read_mps", &mipx::readMps, "filename"_a,
          "Read an MPS model from disk.");
    m.def("read_lp", &mipx::readLp, "filename"_a,
          "Read a CPLEX LP model from disk.");
    m.def("write_mps", &mipx::writeMps, "filename"_a, "problem"_a,
          "Write a model to MPS format.");

    nb::class_<MipPreRootStats>(m, "MipPreRootStats")
        .def_ro("enabled", &MipPreRootStats::enabled)
        .def_ro("lp_light_enabled", &MipPreRootStats::lp_light_enabled)
        .def_ro("lp_light_available", &MipPreRootStats::lp_light_available)
        .def_ro("portfolio_enabled", &MipPreRootStats::portfolio_enabled)
        .def_ro("rounds", &MipPreRootStats::rounds)
        .def_ro("calls", &MipPreRootStats::calls)
        .def_ro("improvements", &MipPreRootStats::improvements)
        .def_ro("feasible_found", &MipPreRootStats::feasible_found)
        .def_ro("early_stops", &MipPreRootStats::early_stops)
        .def_ro("portfolio_epochs", &MipPreRootStats::portfolio_epochs)
        .def_ro("portfolio_wins", &MipPreRootStats::portfolio_wins)
        .def_ro("portfolio_stagnant", &MipPreRootStats::portfolio_stagnant)
        .def_ro("fj_calls", &MipPreRootStats::fj_calls)
        .def_ro("fpr_calls", &MipPreRootStats::fpr_calls)
        .def_ro("local_mip_calls", &MipPreRootStats::local_mip_calls)
        .def_ro("lp_light_calls", &MipPreRootStats::lp_light_calls)
        .def_ro("lp_light_fpr_calls", &MipPreRootStats::lp_light_fpr_calls)
        .def_ro("lp_light_diving_calls", &MipPreRootStats::lp_light_diving_calls)
        .def_ro("fj_improvements", &MipPreRootStats::fj_improvements)
        .def_ro("fpr_improvements", &MipPreRootStats::fpr_improvements)
        .def_ro("local_mip_improvements", &MipPreRootStats::local_mip_improvements)
        .def_ro("lp_light_fpr_improvements", &MipPreRootStats::lp_light_fpr_improvements)
        .def_ro("lp_light_diving_improvements", &MipPreRootStats::lp_light_diving_improvements)
        .def_ro("lp_light_lp_solves", &MipPreRootStats::lp_light_lp_solves)
        .def_ro("lp_light_lp_iterations", &MipPreRootStats::lp_light_lp_iterations)
        .def_ro("work_units", &MipPreRootStats::work_units)
        .def_ro("lp_light_lp_work", &MipPreRootStats::lp_light_lp_work)
        .def_ro("fj_reward", &MipPreRootStats::fj_reward)
        .def_ro("fpr_reward", &MipPreRootStats::fpr_reward)
        .def_ro("local_mip_reward", &MipPreRootStats::local_mip_reward)
        .def_ro("lp_light_fpr_reward", &MipPreRootStats::lp_light_fpr_reward)
        .def_ro("lp_light_diving_reward", &MipPreRootStats::lp_light_diving_reward)
        .def_ro("effort_scale_final", &MipPreRootStats::effort_scale_final)
        .def_ro("time_seconds", &MipPreRootStats::time_seconds)
        .def_ro("time_to_first_feasible", &MipPreRootStats::time_to_first_feasible)
        .def_ro("incumbent_at_root", &MipPreRootStats::incumbent_at_root);

    nb::class_<MipResult>(m, "MipResult")
        .def_ro("status", &MipResult::status)
        .def_ro("objective", &MipResult::objective)
        .def_ro("best_bound", &MipResult::best_bound)
        .def_ro("gap", &MipResult::gap)
        .def_ro("nodes", &MipResult::nodes)
        .def_ro("lp_iterations", &MipResult::lp_iterations)
        .def_ro("work_units", &MipResult::work_units)
        .def_ro("time_seconds", &MipResult::time_seconds)
        .def_ro("solution", &MipResult::solution);

    nb::class_<MipSolver>(m, "MipSolver")
        .def(nb::init<>())
        .def("load", &MipSolver::load, "problem"_a)
        .def("solve", &MipSolver::solve)
        .def("set_node_limit", &MipSolver::setNodeLimit, "limit"_a)
        .def("set_time_limit", &MipSolver::setTimeLimit, "seconds"_a)
        .def("set_gap_tolerance", &MipSolver::setGapTolerance, "tol"_a)
        .def("set_verbose", &MipSolver::setVerbose, "verbose"_a)
        .def("set_presolve", &MipSolver::setPresolve, "enabled"_a)
        .def("set_cuts_enabled", &MipSolver::setCutsEnabled, "enabled"_a)
        .def("set_num_threads", &MipSolver::setNumThreads, "threads"_a)
        .def("set_root_lp_policy", &MipSolver::setRootLpPolicy, "policy"_a)
        .def("set_parallel_mode", &MipSolver::setParallelMode, "mode"_a)
        .def("set_heuristic_mode", &MipSolver::setHeuristicMode, "mode"_a)
        .def("set_heuristic_seed", &MipSolver::setHeuristicSeed, "seed"_a)
        .def("set_pre_root_lpfree_enabled", &MipSolver::setPreRootLpFreeEnabled,
             "enabled"_a)
        .def("set_pre_root_lplight_enabled", &MipSolver::setPreRootLpLightEnabled,
             "enabled"_a)
        .def("set_pre_root_portfolio_enabled", &MipSolver::setPreRootPortfolioEnabled,
             "enabled"_a)
        .def("set_search_profile", &MipSolver::setSearchProfile, "profile"_a)
        .def("has_lplight_capability", &MipSolver::hasLpLightCapability)
        .def("get_pre_root_stats", [](const MipSolver& solver) {
            return solver.getPreRootStats();
        });

    m.def(
        "solve_mps",
        [](const std::string& filename,
           mipx::Int node_limit,
           double time_limit,
           double gap_tolerance,
           bool verbose,
           bool presolve,
           bool cuts,
           mipx::Int threads,
           HeuristicRuntimeMode heuristic_mode,
           std::uint64_t heuristic_seed) {
            auto model = mipx::readMps(filename);
            MipSolver solver;
            solver.setNodeLimit(node_limit);
            solver.setTimeLimit(time_limit);
            solver.setGapTolerance(gap_tolerance);
            solver.setVerbose(verbose);
            solver.setPresolve(presolve);
            solver.setCutsEnabled(cuts);
            solver.setNumThreads(threads);
            solver.setHeuristicMode(heuristic_mode);
            solver.setHeuristicSeed(heuristic_seed);
            solver.load(model);
            return solver.solve();
        },
        "filename"_a,
        "node_limit"_a = 1000000,
        "time_limit"_a = 3600.0,
        "gap_tolerance"_a = 1e-4,
        "verbose"_a = false,
        "presolve"_a = true,
        "cuts"_a = true,
        "threads"_a = 1,
        "heuristic_mode"_a = HeuristicRuntimeMode::Deterministic,
        "heuristic_seed"_a = 1,
        "Load an MPS model and solve it with default MIP settings.");
}
