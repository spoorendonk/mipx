#include "../io/file_suffix.h"
#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/logger.h"
#include "mipx/mip_solver.h"
#include "mipx/pdlp.h"
#include "mipx/presolve.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace {

void printUsage(FILE* out) {
    std::fprintf(out,
                 "Usage: mipx-solve <model-file> [--format auto|mps|lp] "
                 "[--threads N] [--time-limit S] "
                 "[--node-limit N] [--gap-tol G] [--no-cuts|--cuts] "
                 "[--no-presolve|--presolve] [--barrier|--pdlp|--dual|--concurrent-root] "
                 "[--conflicts|--no-conflicts] "
                 "[--conflict-watches|--no-conflict-watches] "
                 "[--conflict-activity-branching|--no-conflict-activity-branching] "
                 "[--conflict-restart-threshold N] "
                 "[--geometric-restarts|--no-geometric-restarts] "
                 "[--bound-conflict-learning|--no-bound-conflict-learning] "
                 "[--reconvergence-cuts|--no-reconvergence-cuts] "
                 "[--conflict-ray-shortening|--no-conflict-ray-shortening] "
                 "[--conflict-lp-proofs|--no-conflict-lp-proofs] "
                 "[--print-conflict-bench-stats (serial)] "
                 "[--pre-lp-propagation|--no-pre-lp-propagation] "
                 "[--pre-lp-propagation-bounds|--no-pre-lp-propagation-bounds] "
                 "[--tree-presolve|--no-tree-presolve] "
                 "[--deterministic-tree-presolve (deterministic TBB tree)] "
                 "[--tree-presolve-auto|--no-tree-presolve-auto] "
                 "[--tree-presolve-max-depth N] "
                 "[--tree-presolve-min-frac N] "
                 "[--tree-presolve-depth-frequency N] "
                 "[--tree-cuts|--no-tree-cuts] "
                 "[--objective-propagation|--no-objective-propagation] "
                 "[--objective-integrality|--no-objective-integrality] "
                 "[--tree-pseudocosts|--no-tree-pseudocosts] "
                 "[--balanced-branch-score|--no-balanced-branch-score] "
                 "[--directional-inference (serial tree)] "
                 "[--inference-branch-coverage (serial tree)] "
                 "[--inference-branch-score (serial tree)] "
                 "[--branch-cutoff-stats (serial tree)] "
                 "[--serial-node-policy best-first|best-estimate|depth-biased] "
                 "[--cut-strengthening|--no-cut-strengthening] "
                 "[--root-cut-recheck] [--root-cut-propagation] [--root-cut-inference] "
                 "[--root-cut-propagation-row-budget N] [--root-flow-cover] "
                 "[--root-cutoff-feedback] "
                 "[--sos2-adjacent-intervals|--sos2-native-branching] "
                 "[--sos2-propagation] "
                 "[--probing|--no-probing] [--probing-rounds N] [--probing-probes N] "
                 "[--probing-time S] "
                 "[--systematic-probing|--no-systematic-probing] "
                 "[--systematic-probing-pairs N] [--systematic-probing-rows N] "
                 "[--root-rapid-learning|--no-root-rapid-learning] "
                 "[--root-rapid-learning-nodes N] [--root-rapid-learning-rows N] "
                 "[--presolve-probing|--no-presolve-probing] [--presolve-probing-rounds N] "
                 "[--presolve-probing-probes N] [--presolve-probing-time S] "
                 "[--clique-table|--no-clique-table] "
                 "[--plunge-depth N|--no-plunge] [--plunge-bound-quotient Q] "
                 "[--presolve-forcing-rows|--no-presolve-forcing-rows] "
                 "[--presolve-dual-fixing|--no-presolve-dual-fixing] "
                 "[--presolve-coeff-tightening|--no-presolve-coeff-tightening] "
                 "[--parallel-mode deterministic|opportunistic] [--seed N] "
                 "[--persistent-tree-workers (opportunistic TBB tree)] "
                 "[--deterministic-batch-branching (deterministic TBB tree)] "
                 "[--heur-deterministic|--heur-opportunistic] "
                 "[--root-feasjump-restarts N] "
                 "[--pre-root-lpfree|--no-pre-root-lpfree] [--pre-root-work W] "
                 "[--pre-root-rounds N] [--pre-root-no-early-stop] "
                 "[--pre-root-lplight|--no-pre-root-lplight] "
                 "[--pre-root-portfolio|--pre-root-fixed] [--no-symmetry] "
                 "[--exact-refine-off|--exact-refine-auto|--exact-refine-on] "
                 "[--exact-rational-check|--exact-no-rational-check] "
                 "[--exact-warning-tol T] [--exact-cert-tol T] "
                 "[--exact-max-rounds N] [--exact-repair-passes N] "
                 "[--exact-rational-scale S] "
                 "[--dual-idiot-crash|--dual-no-idiot-crash] "
                 "[--dual-primal-devex|--dual-no-primal-devex] "
                 "[--dual-bfrt|--dual-no-bfrt] "
                 "[--dual-lu-update-limit N] "
                 "[--dual-lu-ft-drop-tol T] [--dual-dense-row-price] "
                 "[--dual-row-price-actual-work] "
                 "[--dual-retain-dse-weights] "
                 "[--search-stable|--search-default|--search-aggressive] "
                 "[--root-heur-max-int-inf N] [--root-heur-max-int-vars N] "
                 "[--pdlp-opt-norm l2|linf] [--pdlp-restart-gap-decay F] "
                 "[--pdlp-restart-kkt-decay F] [--pdlp-adaptive-eval-frequency] "
                 "[--pdlp-restart-first-step-anchor] "
                 "[--barrier-ordering auto|amd|nd|cudss|screened] [--barrier-ne dense|sparse] "
                 "[--barrier-augmented] [--barrier-ne-lower-updates] "
                 "[--barrier-dualize off|on|auto] "
                 "[--barrier-dense-cols N] [--barrier-implicit-bounds] [--barrier-gondzio N] "
                 "[--barrier-gpu-dense-cols N] "
                 "[--barrier-hsd] [--barrier-initial-point auto|mehrotra|hipo] "
                 "[--barrier-crossover-pushes] "
                 "[--gpu|--no-gpu] [--gpu-min-rows N] [--gpu-min-nnz N] "
                 "[--relax-integrality] "
                 "[--verbose|--quiet] [--print-backend] [--print-det-heur-gate]\n"
                 "\n"
                 "The model file is read as MPS unless it ends in .lp, "
                 "ignoring any .gz/.bz2\n"
                 "compression suffix. --format overrides that choice.\n");
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(stdout);
        return 1;
    }
    if (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") {
        printUsage(stdout);
        return 0;
    }

    std::string filename = argv[1];
    mipx::ModelFormat model_format = mipx::ModelFormat::Auto;
    int num_threads = 1;
    double time_limit = -1.0;  // negative = use solver default
    mipx::Int node_limit = 1000000;
    double gap_tol = 1e-4;
    bool verbose = true;
    bool print_backend = false;
    bool print_det_heur_gate = false;
    bool presolve = true;
    bool presolve_forcing_rows = true;
    bool presolve_dual_fixing = true;
    bool presolve_coefficient_tightening = false;
    bool cuts_enabled = true;
    enum class LpMode { Dual, Barrier, Pdlp, Concurrent };
    LpMode lp_mode = LpMode::Dual;
    bool barrier_gpu = true;
    mipx::BarrierOrdering barrier_ordering = mipx::BarrierOrdering::Auto;
    bool barrier_crossover_pushes = false;
    mipx::BarrierDualize barrier_dualize = mipx::BarrierDualize::Off;
    bool barrier_sparse_ne = false;
    bool barrier_ne_lower_updates = false;
    bool barrier_augmented = false;
    mipx::Int barrier_dense_cols = mipx::BarrierOptions{}.ne_max_dense_cols;
    mipx::Int barrier_gpu_dense_cols = mipx::BarrierOptions{}.gpu_ne_max_dense_cols;
    bool barrier_implicit_bounds = false;
    bool barrier_hsd = false;
    auto barrier_initial_point = mipx::BarrierInitialPoint::Auto;
    mipx::Int barrier_gondzio = 0;
    mipx::PdlpOptions pdlp_options;
    bool relax_integrality = false;
    mipx::Int barrier_gpu_min_rows = 512;
    mipx::Int barrier_gpu_min_nnz = 10000;
    mipx::ParallelMode parallel_mode = mipx::ParallelMode::Deterministic;
    bool persistent_tree_workers = false;
    bool deterministic_batch_branching = false;
    uint64_t heuristic_seed = 1;
    mipx::Int root_heur_max_int_inf = 12;
    mipx::Int root_heur_max_int_vars = 96;
    std::optional<mipx::Int> root_feasjump_restarts;
    bool pre_root_lpfree = false;
    double pre_root_work = 5.0e4;
    mipx::Int pre_root_rounds = 24;
    bool pre_root_early_stop = true;
    bool pre_root_lplight = false;
    bool pre_root_portfolio = true;
    bool conflicts_enabled = true;
    bool conflict_watches = false;
    bool conflict_activity_branching = false;
    mipx::Int conflict_restart_threshold = 0;
    bool geometric_restarts = false;
    bool bound_conflict_learning = false;
    bool reconvergence_cuts = false;
    bool conflict_ray_shortening = false;
    bool conflict_lp_proofs = false;
    bool print_conflict_bench_stats = false;
    bool pre_lp_propagation = false;
    bool pre_lp_propagation_bounds = false;
    bool tree_presolve_enabled = true;
    bool deterministic_tree_presolve = false;
    bool tree_presolve_auto = true;
    mipx::Int tree_presolve_max_depth = 24;
    mipx::Int tree_presolve_min_frac = 8;
    mipx::Int tree_presolve_depth_frequency = 3;
    bool tree_cuts_enabled = false;
    bool objective_propagation = false;
    bool objective_integrality = false;
    bool tree_pseudocosts = true;
    bool balanced_branch_score = false;
    bool directional_inference = false;
    bool inference_branch_coverage = false;
    bool inference_branch_score = false;
    bool branch_cutoff_stats = false;
    std::optional<mipx::NodePolicy> serial_node_policy;
    const char* serial_node_policy_name = nullptr;
    bool cut_strengthening = false;
    bool root_cut_recheck = false;
    bool root_cut_propagation = false;
    bool root_cut_inference = false;
    bool root_cutoff_feedback = false;
    mipx::Int root_cut_propagation_row_budget = 0;
    bool root_flow_cover = false;
    bool sos2_adjacent_intervals = false;
    bool sos2_native_branching = false;
    bool sos2_propagation = false;
    // Off by default in the solver; a numeric knob left unset keeps the
    // solver's own default, so a run without these flags is unchanged.
    bool probing_enabled = false;
    std::optional<mipx::Int> probing_rounds;
    std::optional<mipx::Int> probing_probes;
    std::optional<double> probing_time;
    bool systematic_probing = false;
    std::optional<mipx::Int> systematic_probing_pairs;
    std::optional<mipx::Int> systematic_probing_rows;
    bool root_rapid_learning = false;
    std::optional<mipx::Int> rapid_learning_nodes;
    std::optional<mipx::Int> rapid_learning_rows;
    bool presolve_probing = false;
    std::optional<mipx::Int> presolve_probing_rounds;
    std::optional<mipx::Int> presolve_probing_probes;
    std::optional<double> presolve_probing_time;
    bool clique_table_enabled = false;
    std::optional<mipx::Int> plunge_depth;
    std::optional<double> plunge_bound_quotient;
    mipx::SearchProfile search_profile = mipx::SearchProfile::Default;
    bool search_profile_explicit = false;
    bool symmetry_enabled = true;
    mipx::ExactRefinementMode exact_refine_mode = mipx::ExactRefinementMode::Off;
    bool exact_rational_check = false;
    double exact_warning_tol = 1e-7;
    double exact_cert_tol = 1e-8;
    mipx::Int exact_max_rounds = 2;
    mipx::Int exact_repair_passes = 2;
    double exact_rational_scale = 1.0e6;
    bool dual_idiot_crash = false;
    int dual_primal_devex_override = -1;  // -1: keep the solver's default
    int dual_bfrt_override = -1;
    mipx::Int dual_lu_update_limit = std::numeric_limits<mipx::Int>::min();
    mipx::Real dual_lu_ft_drop_tolerance = std::numeric_limits<mipx::Real>::quiet_NaN();
    // Forces the dense pivot-row assembly (DualSimplexOptions::force_dense_pivot_row),
    // so the hyper-sparse row price can be benchmarked against it (#178).
    bool dual_dense_row_price = false;
    // Charges the row price its actual work (DualSimplexOptions::row_price_actual_work, #379).
    bool dual_row_price_actual_work = false;
    bool dual_retain_dse_weights = false;

    // Parse optional arguments.
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--time-limit" && i + 1 < argc) {
            time_limit = std::atof(argv[++i]);
        } else if (arg == "--node-limit" && i + 1 < argc) {
            node_limit = static_cast<mipx::Int>(std::atoll(argv[++i]));
        } else if (arg == "--gap-tol" && i + 1 < argc) {
            gap_tol = std::atof(argv[++i]);
        } else if (arg == "--no-cuts") {
            cuts_enabled = false;
        } else if (arg == "--cuts") {
            cuts_enabled = true;
        } else if (arg == "--no-presolve") {
            presolve = false;
        } else if (arg == "--presolve") {
            presolve = true;
        } else if (arg == "--conflicts") {
            conflicts_enabled = true;
        } else if (arg == "--no-conflicts") {
            conflicts_enabled = false;
        } else if (arg == "--conflict-watches") {
            conflict_watches = true;
        } else if (arg == "--no-conflict-watches") {
            conflict_watches = false;
        } else if (arg == "--conflict-activity-branching") {
            conflict_activity_branching = true;
        } else if (arg == "--no-conflict-activity-branching") {
            conflict_activity_branching = false;
        } else if (arg == "--conflict-restart-threshold" && i + 1 < argc) {
            const std::string_view value = argv[++i];
            const char* end = value.data() + value.size();
            const auto [parsed, ec] =
                std::from_chars(value.data(), end, conflict_restart_threshold);
            if (ec != std::errc{} || parsed != end || conflict_restart_threshold < 0) {
                std::fprintf(stderr,
                             "--conflict-restart-threshold requires a nonnegative integer\n");
                return 1;
            }
        } else if (arg == "--geometric-restarts") {
            geometric_restarts = true;
        } else if (arg == "--no-geometric-restarts") {
            geometric_restarts = false;
        } else if (arg == "--bound-conflict-learning") {
            bound_conflict_learning = true;
        } else if (arg == "--no-bound-conflict-learning") {
            bound_conflict_learning = false;
        } else if (arg == "--reconvergence-cuts") {
            reconvergence_cuts = true;
        } else if (arg == "--no-reconvergence-cuts") {
            reconvergence_cuts = false;
        } else if (arg == "--conflict-ray-shortening") {
            conflict_ray_shortening = true;
        } else if (arg == "--no-conflict-ray-shortening") {
            conflict_ray_shortening = false;
        } else if (arg == "--conflict-lp-proofs") {
            conflict_lp_proofs = true;
        } else if (arg == "--no-conflict-lp-proofs") {
            conflict_lp_proofs = false;
        } else if (arg == "--print-conflict-bench-stats") {
            print_conflict_bench_stats = true;
        } else if (arg == "--pre-lp-propagation") {
            pre_lp_propagation = true;
        } else if (arg == "--no-pre-lp-propagation") {
            pre_lp_propagation = false;
        } else if (arg == "--pre-lp-propagation-bounds") {
            pre_lp_propagation_bounds = true;
        } else if (arg == "--no-pre-lp-propagation-bounds") {
            pre_lp_propagation_bounds = false;
        } else if (arg == "--tree-presolve") {
            tree_presolve_enabled = true;
        } else if (arg == "--no-tree-presolve") {
            tree_presolve_enabled = false;
        } else if (arg == "--deterministic-tree-presolve") {
            deterministic_tree_presolve = true;
        } else if (arg == "--tree-presolve-auto") {
            tree_presolve_auto = true;
        } else if (arg == "--no-tree-presolve-auto") {
            tree_presolve_auto = false;
        } else if (arg == "--tree-presolve-max-depth" && i + 1 < argc) {
            tree_presolve_max_depth =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--tree-presolve-min-frac" && i + 1 < argc) {
            tree_presolve_min_frac =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--tree-presolve-depth-frequency" && i + 1 < argc) {
            tree_presolve_depth_frequency =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--tree-cuts") {
            tree_cuts_enabled = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(stdout);
            return 0;
        } else if (arg == "--probing") {
            probing_enabled = true;
        } else if (arg == "--no-probing") {
            probing_enabled = false;
        } else if (arg == "--probing-rounds" && i + 1 < argc) {
            probing_rounds = static_cast<mipx::Int>(std::atoll(argv[++i]));
        } else if (arg == "--probing-probes" && i + 1 < argc) {
            probing_probes = static_cast<mipx::Int>(std::atoll(argv[++i]));
        } else if (arg == "--probing-time" && i + 1 < argc) {
            probing_time = std::atof(argv[++i]);
        } else if (arg == "--systematic-probing") {
            systematic_probing = true;
        } else if (arg == "--no-systematic-probing") {
            systematic_probing = false;
        } else if ((arg == "--systematic-probing-pairs" || arg == "--systematic-probing-rows") &&
                   i + 1 < argc) {
            const std::string_view value = argv[++i];
            const auto end = value.data() + value.size();
            mipx::Int count = 0;
            const auto [parsed, ec] = std::from_chars(value.data(), end, count);
            if (ec != std::errc{} || parsed != end || count <= 0) {
                std::fprintf(stderr, "%s requires a positive integer\n", arg.c_str());
                return 1;
            }
            (arg == "--systematic-probing-pairs" ? systematic_probing_pairs
                                                 : systematic_probing_rows) = count;
        } else if (arg == "--root-rapid-learning") {
            root_rapid_learning = true;
        } else if (arg == "--no-root-rapid-learning") {
            root_rapid_learning = false;
        } else if ((arg == "--root-rapid-learning-nodes" || arg == "--root-rapid-learning-rows") &&
                   i + 1 < argc) {
            const std::string_view value = argv[++i];
            const auto end = value.data() + value.size();
            mipx::Int count = 0;
            const auto [parsed, ec] = std::from_chars(value.data(), end, count);
            if (ec != std::errc{} || parsed != end || count < 0) {
                std::fprintf(stderr, "%s requires a nonnegative integer\n", arg.c_str());
                return 1;
            }
            (arg == "--root-rapid-learning-nodes" ? rapid_learning_nodes : rapid_learning_rows) =
                count;
        } else if (arg == "--presolve-probing") {
            presolve_probing = true;
        } else if (arg == "--no-presolve-probing") {
            presolve_probing = false;
        } else if (arg == "--presolve-probing-rounds" && i + 1 < argc) {
            presolve_probing_rounds =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--presolve-probing-probes" && i + 1 < argc) {
            presolve_probing_probes =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--presolve-probing-time" && i + 1 < argc) {
            presolve_probing_time = std::max(0.0, std::atof(argv[++i]));
        } else if (arg == "--clique-table") {
            clique_table_enabled = true;
        } else if (arg == "--no-clique-table") {
            clique_table_enabled = false;
        } else if (arg == "--plunge-depth" && i + 1 < argc) {
            plunge_depth = static_cast<mipx::Int>(std::atoll(argv[++i]));
        } else if (arg == "--no-plunge") {
            plunge_depth = 0;
        } else if (arg == "--plunge-bound-quotient" && i + 1 < argc) {
            plunge_bound_quotient = std::atof(argv[++i]);
        } else if (arg == "--no-tree-cuts") {
            tree_cuts_enabled = false;
        } else if (arg == "--objective-propagation") {
            objective_propagation = true;
        } else if (arg == "--no-objective-propagation") {
            objective_propagation = false;
        } else if (arg == "--objective-integrality") {
            objective_integrality = true;
        } else if (arg == "--no-objective-integrality") {
            objective_integrality = false;
        } else if (arg == "--tree-pseudocosts") {
            tree_pseudocosts = true;
        } else if (arg == "--no-tree-pseudocosts") {
            tree_pseudocosts = false;
        } else if (arg == "--balanced-branch-score") {
            balanced_branch_score = true;
        } else if (arg == "--no-balanced-branch-score") {
            balanced_branch_score = false;
        } else if (arg == "--directional-inference") {
            directional_inference = true;
        } else if (arg == "--inference-branch-coverage") {
            inference_branch_coverage = true;
        } else if (arg == "--inference-branch-score") {
            inference_branch_score = true;
        } else if (arg == "--branch-cutoff-stats") {
            branch_cutoff_stats = true;
        } else if (arg == "--serial-node-policy" && i + 1 < argc) {
            serial_node_policy_name = argv[++i];
            const std::string_view policy = serial_node_policy_name;
            if (policy == "best-first") {
                serial_node_policy = mipx::NodePolicy::BestFirst;
            } else if (policy == "best-estimate") {
                serial_node_policy = mipx::NodePolicy::BestEstimate;
            } else if (policy == "depth-biased") {
                serial_node_policy = mipx::NodePolicy::DepthBiased;
            } else {
                std::fprintf(stderr, "Invalid serial node policy: %s\n", serial_node_policy_name);
                return 2;
            }
        } else if (arg == "--cut-strengthening") {
            cut_strengthening = true;
        } else if (arg == "--no-cut-strengthening") {
            cut_strengthening = false;
        } else if (arg == "--root-cut-recheck") {
            root_cut_recheck = true;
        } else if (arg == "--root-cut-propagation") {
            root_cut_propagation = true;
        } else if (arg == "--root-cut-inference") {
            root_cut_inference = true;
        } else if (arg == "--root-cutoff-feedback") {
            root_cutoff_feedback = true;
        } else if (arg == "--root-cut-propagation-row-budget" && i + 1 < argc) {
            const std::string_view value = argv[++i];
            const auto end = value.data() + value.size();
            const auto [parsed, ec] =
                std::from_chars(value.data(), end, root_cut_propagation_row_budget);
            if (ec != std::errc{} || parsed != end || root_cut_propagation_row_budget < 0) {
                std::fprintf(stderr,
                             "--root-cut-propagation-row-budget requires a nonnegative integer\n");
                return 1;
            }
        } else if (arg == "--root-flow-cover") {
            root_flow_cover = true;
        } else if (arg == "--sos2-adjacent-intervals") {
            sos2_adjacent_intervals = true;
        } else if (arg == "--sos2-native-branching") {
            sos2_native_branching = true;
        } else if (arg == "--sos2-propagation") {
            sos2_propagation = true;
        } else if (arg == "--presolve-forcing-rows") {
            presolve_forcing_rows = true;
        } else if (arg == "--no-presolve-forcing-rows") {
            presolve_forcing_rows = false;
        } else if (arg == "--presolve-dual-fixing") {
            presolve_dual_fixing = true;
        } else if (arg == "--no-presolve-dual-fixing") {
            presolve_dual_fixing = false;
        } else if (arg == "--presolve-coeff-tightening") {
            presolve_coefficient_tightening = true;
        } else if (arg == "--no-presolve-coeff-tightening") {
            presolve_coefficient_tightening = false;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--quiet") {
            verbose = false;
        } else if (arg == "--print-backend") {
            print_backend = true;
        } else if (arg == "--print-det-heur-gate") {
            print_det_heur_gate = true;
        } else if (arg == "--barrier") {
            lp_mode = LpMode::Barrier;
        } else if (arg == "--pdlp") {
            lp_mode = LpMode::Pdlp;
        } else if (arg == "--dual") {
            lp_mode = LpMode::Dual;
        } else if (arg == "--concurrent-root") {
            lp_mode = LpMode::Concurrent;
        } else if (arg == "--parallel-mode" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "deterministic") {
                parallel_mode = mipx::ParallelMode::Deterministic;
            } else if (mode == "opportunistic") {
                parallel_mode = mipx::ParallelMode::Opportunistic;
            } else {
                std::fprintf(stderr,
                             "Invalid --parallel-mode value: %s "
                             "(expected deterministic or opportunistic)\n",
                             mode.c_str());
                return 1;
            }
        } else if (arg == "--heur-deterministic") {
            parallel_mode = mipx::ParallelMode::Deterministic;
        } else if (arg == "--heur-opportunistic") {
            parallel_mode = mipx::ParallelMode::Opportunistic;
        } else if (arg == "--persistent-tree-workers") {
            persistent_tree_workers = true;
        } else if (arg == "--deterministic-batch-branching") {
            deterministic_batch_branching = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            heuristic_seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--root-heur-max-int-inf" && i + 1 < argc) {
            root_heur_max_int_inf =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--root-heur-max-int-vars" && i + 1 < argc) {
            root_heur_max_int_vars =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--root-feasjump-restarts" && i + 1 < argc) {
            const std::string_view value = argv[++i];
            mipx::Int restarts = 0;
            const auto end = value.data() + value.size();
            const auto [parsed, ec] = std::from_chars(value.data(), end, restarts);
            if (ec != std::errc{} || parsed != end || restarts < 0 ||
                restarts > mipx::HeuristicRuntimeConfig::kDefaultRootFeasibilityJumpRestarts) {
                std::fprintf(stderr, "--root-feasjump-restarts requires an integer from 0 to %d\n",
                             mipx::HeuristicRuntimeConfig::kDefaultRootFeasibilityJumpRestarts);
                return 1;
            }
            root_feasjump_restarts = restarts;
        } else if (arg == "--pre-root-lpfree") {
            pre_root_lpfree = true;
        } else if (arg == "--no-pre-root-lpfree") {
            pre_root_lpfree = false;
        } else if (arg == "--pre-root-work" && i + 1 < argc) {
            pre_root_work = std::max(1.0, std::atof(argv[++i]));
        } else if (arg == "--pre-root-rounds" && i + 1 < argc) {
            pre_root_rounds = std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--pre-root-no-early-stop") {
            pre_root_early_stop = false;
        } else if (arg == "--pre-root-lplight") {
            pre_root_lplight = true;
        } else if (arg == "--no-pre-root-lplight") {
            pre_root_lplight = false;
        } else if (arg == "--pre-root-portfolio") {
            pre_root_portfolio = true;
        } else if (arg == "--pre-root-fixed") {
            pre_root_portfolio = false;
        } else if (arg == "--no-symmetry") {
            symmetry_enabled = false;
        } else if (arg == "--exact-refine-off") {
            exact_refine_mode = mipx::ExactRefinementMode::Off;
        } else if (arg == "--exact-refine-auto") {
            exact_refine_mode = mipx::ExactRefinementMode::Auto;
        } else if (arg == "--exact-refine-on") {
            exact_refine_mode = mipx::ExactRefinementMode::On;
        } else if (arg == "--exact-rational-check") {
            exact_rational_check = true;
        } else if (arg == "--exact-no-rational-check") {
            exact_rational_check = false;
        } else if (arg == "--exact-warning-tol" && i + 1 < argc) {
            exact_warning_tol = std::max(1e-12, std::atof(argv[++i]));
        } else if (arg == "--exact-cert-tol" && i + 1 < argc) {
            exact_cert_tol = std::max(1e-12, std::atof(argv[++i]));
        } else if (arg == "--exact-max-rounds" && i + 1 < argc) {
            exact_max_rounds =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--exact-repair-passes" && i + 1 < argc) {
            exact_repair_passes =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--exact-rational-scale" && i + 1 < argc) {
            exact_rational_scale = std::max(1.0, std::atof(argv[++i]));
        } else if (arg == "--dual-idiot-crash") {
            dual_idiot_crash = true;
        } else if (arg == "--dual-no-idiot-crash") {
            dual_idiot_crash = false;
        } else if (arg == "--dual-primal-devex") {
            dual_primal_devex_override = 1;
        } else if (arg == "--dual-no-primal-devex") {
            dual_primal_devex_override = 0;
        } else if (arg == "--dual-bfrt") {
            dual_bfrt_override = 1;
        } else if (arg == "--dual-no-bfrt") {
            dual_bfrt_override = 0;
        } else if (arg == "--dual-lu-update-limit" && i + 1 < argc) {
            dual_lu_update_limit = static_cast<mipx::Int>(std::atoll(argv[++i]));
        } else if (arg == "--dual-lu-ft-drop-tol" && i + 1 < argc) {
            dual_lu_ft_drop_tolerance = std::atof(argv[++i]);
        } else if (arg == "--dual-dense-row-price") {
            dual_dense_row_price = true;
        } else if (arg == "--dual-row-price-actual-work") {
            dual_row_price_actual_work = true;
        } else if (arg == "--dual-retain-dse-weights") {
            dual_retain_dse_weights = true;
        } else if (arg == "--search-stable") {
            search_profile = mipx::SearchProfile::Stable;
            search_profile_explicit = true;
        } else if (arg == "--search-default") {
            search_profile = mipx::SearchProfile::Default;
            search_profile_explicit = true;
        } else if (arg == "--search-aggressive") {
            search_profile = mipx::SearchProfile::Aggressive;
            search_profile_explicit = true;
        } else if (arg == "--pdlp-opt-norm" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "l2") {
                pdlp_options.optimality_norm = mipx::PdlpOptimalityNorm::L2;
            } else if (mode == "linf") {
                pdlp_options.optimality_norm = mipx::PdlpOptimalityNorm::LInf;
            } else {
                std::fprintf(stderr,
                             "Invalid --pdlp-opt-norm value: %s "
                             "(expected l2 or linf)\n",
                             mode.c_str());
                return 1;
            }
        } else if ((arg == "--pdlp-restart-gap-decay" || arg == "--pdlp-restart-kkt-decay") &&
                   i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            const double fraction = std::strtod(value, &end);
            if (end == value || *end != '\0' || !(fraction >= 0.0 && fraction < 1.0)) {
                std::fprintf(stderr, "Invalid %s value: %s (expected a fraction in [0, 1))\n",
                             argv[i - 1], value);
                return 1;
            }
            if (arg == "--pdlp-restart-gap-decay")
                pdlp_options.restart_gap_decay = fraction;
            else
                pdlp_options.restart_kkt_decay = fraction;
        } else if (arg == "--pdlp-adaptive-eval-frequency") {
            pdlp_options.adaptive_eval_frequency = true;
        } else if (arg == "--pdlp-restart-first-step-anchor") {
            pdlp_options.restart_first_step_anchor = true;
        } else if (arg == "--gpu") {
            barrier_gpu = true;
        } else if (arg == "--no-gpu") {
            barrier_gpu = false;
        } else if (arg == "--gpu-min-rows" && i + 1 < argc) {
            barrier_gpu_min_rows =
                std::max<mipx::Int>(0, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--gpu-min-nnz" && i + 1 < argc) {
            barrier_gpu_min_nnz =
                std::max<mipx::Int>(0, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--barrier-ordering" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "auto") {
                barrier_ordering = mipx::BarrierOrdering::Auto;
            } else if (mode == "amd") {
                barrier_ordering = mipx::BarrierOrdering::Amd;
            } else if (mode == "nd") {
                barrier_ordering = mipx::BarrierOrdering::Nd;
            } else if (mode == "cudss") {
                barrier_ordering = mipx::BarrierOrdering::CuDSS;
            } else if (mode == "screened") {
                barrier_ordering = mipx::BarrierOrdering::Screened;
            } else {
                std::fprintf(stderr, "Unknown barrier ordering: %s\n", mode.c_str());
                return 1;
            }
        } else if (arg == "--barrier-dualize" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "off") {
                barrier_dualize = mipx::BarrierDualize::Off;
            } else if (mode == "on") {
                barrier_dualize = mipx::BarrierDualize::On;
            } else if (mode == "auto") {
                barrier_dualize = mipx::BarrierDualize::Auto;
            } else {
                std::fprintf(stderr, "Unknown barrier dualization mode: %s\n", mode.c_str());
                return 1;
            }
        } else if (arg == "--barrier-ne" && i + 1 < argc) {
            // Normal-equations factor of the CPU barrier: dense LL' (the
            // default) or the opt-in sparse supernodal LL' of #290.
            const std::string mode = argv[++i];
            if (mode == "dense") {
                barrier_sparse_ne = false;
            } else if (mode == "sparse") {
                barrier_sparse_ne = true;
            } else {
                std::fprintf(stderr, "Unknown barrier normal-equations factor: %s\n", mode.c_str());
                return 1;
            }
        } else if (arg == "--barrier-augmented") {
            // The CPU augmented system with its sparse LDL' (#140), whatever
            // Auto would choose; it overrides --barrier-ne.
            barrier_augmented = true;
        } else if (arg == "--barrier-ne-lower-updates") {
            barrier_ne_lower_updates = true;
        } else if (arg == "--barrier-dense-cols" && i + 1 < argc) {
            // Most dense columns the sparse NE factor leaves out and solves by
            // Schur complement (#314); 0 keeps every column in the factor.
            barrier_dense_cols =
                std::max<mipx::Int>(0, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--barrier-gpu-dense-cols" && i + 1 < argc) {
            barrier_gpu_dense_cols = static_cast<mipx::Int>(std::clamp<long long>(
                std::atoll(argv[++i]), 0, std::numeric_limits<mipx::Int>::max()));
        } else if (arg == "--barrier-implicit-bounds") {
            // Upper bounds inside the CPU IPM instead of as rows (#299).
            barrier_implicit_bounds = true;
        } else if (arg == "--barrier-crossover-pushes") {
            barrier_crossover_pushes = true;
        } else if (arg == "--barrier-gondzio" && i + 1 < argc) {
            // At most N Gondzio centrality correctors per CPU IPM iteration
            // (#293); 0 (the default) is off.
            barrier_gondzio =
                static_cast<mipx::Int>(std::clamp<long long>(std::atoll(argv[++i]), 0, 1000));
        } else if (arg == "--barrier-hsd") {
            // The CPU IPM on the homogeneous self-dual model (#370).
            barrier_hsd = true;
        } else if (arg == "--barrier-initial-point" && i + 1 < argc) {
            // Starting point of the CPU Mehrotra loop (#144).
            const std::string mode = argv[++i];
            if (mode == "auto") {
                barrier_initial_point = mipx::BarrierInitialPoint::Auto;
            } else if (mode == "mehrotra") {
                barrier_initial_point = mipx::BarrierInitialPoint::Mehrotra;
            } else if (mode == "hipo") {
                barrier_initial_point = mipx::BarrierInitialPoint::Hipo;
            } else {
                std::fprintf(stderr, "Unknown barrier initial point: %s\n", mode.c_str());
                return 1;
            }
        } else if (arg == "--format" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "auto") {
                model_format = mipx::ModelFormat::Auto;
            } else if (mode == "mps") {
                model_format = mipx::ModelFormat::Mps;
            } else if (mode == "lp") {
                model_format = mipx::ModelFormat::Lp;
            } else {
                std::fprintf(stderr,
                             "Invalid --format value: %s "
                             "(expected auto, mps or lp)\n",
                             mode.c_str());
                return 1;
            }
        } else if (arg == "--relax-integrality") {
            relax_integrality = true;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (!search_profile_explicit && num_threads > 1) {
        search_profile = mipx::SearchProfile::Aggressive;
    }

    mipx::LpProblem lp;
    try {
        lp = mipx::readModel(filename, model_format);
    } catch (const mipx::ModelFormatError& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        // A compressed LP file is the one format error --format cannot
        // fix: neither reader takes it, and the error already says to
        // decompress the file.
        const bool compressed_lp =
            mipx::detectModelFormat(filename) == mipx::ModelFormat::Lp &&
            mipx::io_detail::stripCompressionSuffix(filename).size() != filename.size();
        if (model_format == mipx::ModelFormat::Auto && !compressed_lp) {
            // Only worth saying when the reader was guessed from the
            // extension. Spelled out in full because the model file has to be
            // the first argument -- "--format lp model.lp" is rejected with
            // "Unknown argument: lp", so a bare flag name would send the user
            // straight into that.
            std::fprintf(stderr,
                         "Hint: force a reader with "
                         "'mipx-solve %s --format mps|lp' "
                         "(the model file must come first).\n",
                         filename.c_str());
        }
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    try {
        if (relax_integrality) {
            for (auto& t : lp.col_type) {
                t = mipx::VarType::Continuous;
            }
            // Dropping SOS and indicator constraints relaxes the model as
            // well; kept, they would send it back to the MIP solver.
            lp.sos_constraints.clear();
            lp.indicator_constraints.clear();
        }
        mipx::Logger log;
        mipx::PresolveOptions presolve_opts;
        presolve_opts.enable_forcing_rows = presolve_forcing_rows;
        presolve_opts.enable_dual_fixing = presolve_dual_fixing;
        presolve_opts.enable_coefficient_tightening = presolve_coefficient_tightening;
        presolve_opts.enable_probing = presolve_probing;
        if (presolve_probing_rounds) {
            presolve_opts.probing_max_rounds = *presolve_probing_rounds;
        }
        if (presolve_probing_probes) {
            presolve_opts.probing_max_probes_per_round = *presolve_probing_probes;
        }
        if (presolve_probing_time) {
            presolve_opts.probing_time_limit = *presolve_probing_time;
        }

        // SOS and indicator constraints on continuous columns still make a
        // MIP: MipSolver linearizes them with binaries, where the LP path
        // would drop them (#254).
        if (lp.hasIntegers() || mipx::hasAdvancedModelFeatures(lp)) {
            // MIP solve — MipSolver prints its own banner.
            mipx::MipSolver solver;
            const bool collect_conflict_bench_stats =
                print_conflict_bench_stats && num_threads <= 1;
            solver.setNumThreads(num_threads);
            if (time_limit >= 0.0) {
                solver.setTimeLimit(time_limit);
            }
            solver.setNodeLimit(node_limit);
            solver.setGapTolerance(gap_tol);
            solver.setVerbose(verbose);
            solver.setPresolve(presolve);
            solver.setRootPresolveOptions(presolve_opts);
            solver.setCutsEnabled(cuts_enabled);
            solver.setDualRowPriceActualWork(dual_row_price_actual_work);
            solver.setDualRetainDseWeights(dual_retain_dse_weights);
            if (lp_mode == LpMode::Barrier) {
                solver.setRootLpPolicy(mipx::RootLpPolicy::BarrierRoot);
            } else if (lp_mode == LpMode::Pdlp) {
                solver.setRootLpPolicy(mipx::RootLpPolicy::PdlpRoot);
            } else if (lp_mode == LpMode::Concurrent) {
                solver.setRootLpPolicy(mipx::RootLpPolicy::ConcurrentRootExperimental);
            } else {
                solver.setRootLpPolicy(mipx::RootLpPolicy::DualDefault);
            }
            solver.setBarrierUseGpu(barrier_gpu);
            solver.setBarrierOrdering(barrier_ordering);
            solver.setBarrierDualize(barrier_dualize);
            solver.setBarrierGpuThresholds(barrier_gpu_min_rows, barrier_gpu_min_nnz);
            solver.setParallelMode(parallel_mode);
            solver.setPersistentTreeWorkers(persistent_tree_workers);
            solver.setDeterministicBatchBranching(deterministic_batch_branching);
            solver.setHeuristicSeed(heuristic_seed);
            solver.setRootHeuristicThresholds(root_heur_max_int_inf, root_heur_max_int_vars);
            if (root_feasjump_restarts) {
                solver.setRootFeasibilityJumpRestarts(*root_feasjump_restarts);
            }
            solver.setPreRootLpFreeEnabled(pre_root_lpfree);
            solver.setPreRootLpFreeWorkBudget(pre_root_work);
            solver.setPreRootLpFreeMaxRounds(pre_root_rounds);
            solver.setPreRootLpFreeEarlyStop(pre_root_early_stop);
            solver.setPreRootLpLightEnabled(pre_root_lplight);
            solver.setPreRootPortfolioEnabled(pre_root_portfolio);
            solver.setConflictsEnabled(conflicts_enabled);
            solver.setConflictWatchesEnabled(conflict_watches);
            solver.setConflictActivityBranchingEnabled(conflict_activity_branching);
            solver.setConflictRestartThreshold(conflict_restart_threshold);
            solver.setGeometricRestartsEnabled(geometric_restarts);
            solver.setBoundConflictLearningEnabled(bound_conflict_learning);
            solver.setReconvergenceCutsEnabled(reconvergence_cuts);
            solver.setConflictRayShorteningEnabled(conflict_ray_shortening);
            solver.setConflictLpProofsEnabled(conflict_lp_proofs);
            solver.setConflictBenchStats(collect_conflict_bench_stats);
            solver.setPreLpPropagationEnabled(pre_lp_propagation);
            solver.setPreLpPropagationBoundsEnabled(pre_lp_propagation_bounds);
            solver.setTreePresolveEnabled(tree_presolve_enabled);
            solver.setDeterministicTreePresolve(deterministic_tree_presolve);
            solver.setTreePresolveAutoTuning(tree_presolve_auto);
            solver.setTreePresolveControls(tree_presolve_max_depth, tree_presolve_min_frac,
                                           tree_presolve_depth_frequency);
            solver.setTreeCutsEnabled(tree_cuts_enabled);
            solver.setObjectivePropagationEnabled(objective_propagation);
            solver.setObjectiveIntegralityEnabled(objective_integrality);
            solver.setTreePseudocostsEnabled(tree_pseudocosts);
            solver.setBalancedBranchScore(balanced_branch_score);
            solver.setDirectionalInferenceEnabled(directional_inference);
            solver.setInferenceBranchCoverageEnabled(inference_branch_coverage);
            solver.setInferenceBranchScoreEnabled(inference_branch_score);
            solver.setBranchCutoffStatsEnabled(branch_cutoff_stats);
            solver.setSerialNodePolicyOverride(serial_node_policy);
            solver.setCutStrengthening(cut_strengthening);
            solver.setRootCutRecheck(root_cut_recheck);
            solver.setRootCutPropagationEnabled(root_cut_propagation);
            solver.setRootCutInferenceEnabled(root_cut_inference);
            solver.setRootCutoffFeedbackEnabled(root_cutoff_feedback);
            solver.setRootCutPropagationRowBudget(root_cut_propagation_row_budget);
            solver.setRootFlowCoverEnabled(root_flow_cover);
            solver.setProbingEnabled(probing_enabled);
            if (probing_rounds) {
                solver.setProbingMaxRounds(*probing_rounds);
            }
            if (probing_probes) {
                solver.setProbingMaxProbesPerRound(*probing_probes);
            }
            if (probing_time) {
                solver.setProbingTimeLimit(*probing_time);
            }
            solver.setSystematicProbingEnabled(systematic_probing);
            if (systematic_probing_pairs) {
                solver.setSystematicProbingMaxPairs(*systematic_probing_pairs);
            }
            if (systematic_probing_rows) {
                solver.setSystematicProbingRowsPerBranch(*systematic_probing_rows);
            }
            solver.setRootRapidLearningEnabled(root_rapid_learning);
            if (rapid_learning_nodes) {
                solver.setRootRapidLearningMaxNodes(*rapid_learning_nodes);
            }
            if (rapid_learning_rows) {
                solver.setRootRapidLearningRowBudget(*rapid_learning_rows);
            }
            solver.setCliqueTableEnabled(clique_table_enabled);
            if (plunge_depth || plunge_bound_quotient) {
                // The setter takes both; whichever was not given keeps the
                // solver's default.
                solver.setPlungeControls(
                    plunge_depth.value_or(solver.getPlungeMaxDepth()),
                    plunge_bound_quotient.value_or(solver.getPlungeBoundQuotient()));
            }
            solver.setSymmetryEnabled(symmetry_enabled);
            solver.setExactRefinementMode(exact_refine_mode);
            solver.setExactRefinementRationalCheck(exact_rational_check);
            solver.setExactRefinementWarningTolerance(exact_warning_tol);
            solver.setExactRefinementCertificateTolerance(exact_cert_tol);
            solver.setExactRefinementMaxRounds(exact_max_rounds);
            solver.setExactRefinementRepairPasses(exact_repair_passes);
            solver.setExactRefinementRationalScale(exact_rational_scale);
            solver.setSearchProfile(search_profile);
            solver.setSos2AdjacentIntervals(sos2_adjacent_intervals);
            solver.setSos2NativeBranching(sos2_native_branching);
            solver.setSos2Propagation(sos2_propagation);
            solver.load(lp);
            auto result = solver.solve();
            const char* parallel_mode_name = (parallel_mode == mipx::ParallelMode::Deterministic)
                                                 ? "deterministic"
                                                 : "opportunistic";
            const char* search_profile_name = "default";
            if (search_profile == mipx::SearchProfile::Stable) {
                search_profile_name = "stable";
            } else if (search_profile == mipx::SearchProfile::Aggressive) {
                search_profile_name = "aggressive";
            }
            log.log(
                "Run profile: mode=%s seed=%llu threads=%d search=%s "
                "conflicts=%s conflict_ray_shortening=%s bound_conflict_learning_requested=%s "
                "conflict_lp_proofs=%s "
                "conflict_activity_branching_requested=%s conflict_restart_threshold=%d "
                "pre_lp_propagation=%s "
                "pre_lp_propagation_bounds=%s "
                "tree_presolve=%s tree_presolve_min_frac=%d "
                "tree_presolve_freq=%d tree_presolve_auto=%s tree_cuts=%s "
                "cut_strengthening=%s objective_propagation=%s objective_integrality=%s "
                "tree_pseudocosts=%s balanced_branch_score=%s geometric_restarts_requested=%s\n",
                parallel_mode_name, static_cast<unsigned long long>(heuristic_seed), num_threads,
                search_profile_name, conflicts_enabled ? "on" : "off",
                conflict_ray_shortening && conflicts_enabled && num_threads <= 1 ? "on" : "off",
                bound_conflict_learning ? "on" : "off",
                conflict_lp_proofs && conflicts_enabled && num_threads <= 1 ? "on" : "off",
                conflict_activity_branching ? "on" : "off", conflict_restart_threshold,
                pre_lp_propagation ? "on" : "off", pre_lp_propagation_bounds ? "on" : "off",
                tree_presolve_enabled ? "on" : "off", tree_presolve_min_frac,
                tree_presolve_depth_frequency, tree_presolve_auto ? "on" : "off",
                tree_cuts_enabled ? "on" : "off", cut_strengthening ? "on" : "off",
                objective_propagation ? "on" : "off", objective_integrality ? "on" : "off",
                tree_pseudocosts ? "on" : "off", balanced_branch_score ? "on" : "off",
                geometric_restarts ? "on" : "off");

            if (result.status == mipx::Status::Optimal && result.gap_limit_reached) {
                log.log("Status: Gap limit\n");
            } else {
                switch (result.status) {
                    case mipx::Status::Optimal:
                        log.log("Status: Optimal\n");
                        break;
                    case mipx::Status::Infeasible:
                        log.log("Status: Infeasible\n");
                        break;
                    case mipx::Status::Unbounded:
                        log.log("Status: Unbounded\n");
                        break;
                    case mipx::Status::NodeLimit:
                        log.log("Status: Node limit\n");
                        break;
                    case mipx::Status::TimeLimit:
                        log.log("Status: Time limit\n");
                        break;
                    default:
                        log.log("Status: Error\n");
                        break;
                }
            }
            // MipResult::objective defaults to 0, so a run that ends without an
            // incumbent (a limit hit first, or infeasibility) would print a
            // plausible-looking 0 (#224). Report it only with a solution.
            // An empty model solves to Optimal with no columns to report.
            if (!result.solution.empty() || result.status == mipx::Status::Optimal) {
                log.log("Objective: %.10e\n", result.objective);
            } else {
                log.log("Objective: none\n");
            }
            log.log("Nodes: %d\n", result.nodes);
            log.log("LP iterations: %d\n", result.lp_iterations);
            log.log("Work units: %.2f\n", result.work_units);
            if (collect_conflict_bench_stats) {
                const auto& stats = solver.getConflictStats();
                log.log("Conflict bench bounds: sense=%s input_continuous_cols=%td final=%.17g\n",
                        lp.sense == mipx::Sense::Minimize ? "min" : "max",
                        std::ranges::count(lp.col_type, mipx::VarType::Continuous),
                        result.best_bound);
                log.log("Conflict bench tree: bound_learning=%d reconvergence=%d\n",
                        stats.bound_conflict_learning_in_tree, stats.reconvergence_cuts_in_tree);
                if (stats.root_bound_before_tree) {
                    log.log("Conflict bench root bound: %.17g\n", *stats.root_bound_before_tree);
                }
                log.log(
                    "Conflict bench counters: linear_clause_checks=%llu linear_literal_checks=%llu "
                    "watch_candidates=%llu "
                    "watch_body_scans=%llu watch_literal_checks=%llu lp_infeasible=%d "
                    "cutoff=%d combined=%d learned_1uip=%d reconvergence=%d "
                    "legacy_branch_overrides=%d conflict_restarts=%d\n",
                    static_cast<unsigned long long>(stats.linear_clause_checks),
                    static_cast<unsigned long long>(stats.linear_literal_checks),
                    static_cast<unsigned long long>(stats.watch_candidates),
                    static_cast<unsigned long long>(stats.watch_body_scans),
                    static_cast<unsigned long long>(stats.watch_literal_checks),
                    stats.lp_infeasible_conflicts, stats.lp_cutoff_proofs_learned,
                    stats.lp_combined_proofs_learned, stats.bound_conflicts_learned,
                    stats.reconvergence_cuts_added, stats.branch_score_overrides,
                    solver.getSearchStats().conflict_restarts);
            }
            if (systematic_probing) {
                const auto& stats = solver.getProbingStats();
                log.log(
                    "Systematic probing: down=%d up=%d rows=%llu fixings=%d implications=%d "
                    "conflicts=%d tree_fixings=%llu tree_cutoffs=%llu\n",
                    stats.branches_probed[0], stats.branches_probed[1],
                    static_cast<unsigned long long>(stats.row_visits), stats.fixings_found,
                    stats.implications_found, stats.checked_conflicts_published,
                    static_cast<unsigned long long>(stats.tree_implication_fixings),
                    static_cast<unsigned long long>(stats.tree_implication_cutoffs));
            }
            if (root_rapid_learning) {
                const auto& stats = solver.getRapidLearningStats();
                constexpr auto stops =
                    std::array{"not-run",  "exhausted", "node-limit", "row-limit",
                               "no-yield", "deadline",  "infeasible"};
                log.log(
                    "Root rapid learning: stop=%s nodes=%d rows=%d empty_blocks=%d "
                    "bounds=%llu conflicts=%d incumbents=%d rejected_proofs=%llu "
                    "duplicate_conflicts=%d rejected_incumbents=%d "
                    "tree_tightenings=%llu tree_cutoffs=%llu\n",
                    stops[static_cast<std::size_t>(stats.stop)], stats.nodes, stats.row_visits,
                    stats.empty_blocks, static_cast<unsigned long long>(stats.bounds_imported),
                    stats.conflicts_imported, stats.incumbents_imported,
                    static_cast<unsigned long long>(stats.proofs_rejected),
                    stats.duplicate_conflicts, stats.incumbents_rejected,
                    static_cast<unsigned long long>(stats.tree_conflict_tightenings),
                    static_cast<unsigned long long>(stats.tree_conflict_cutoffs));
            }
            if (print_det_heur_gate) {
                log.log("Deterministic tree heuristic gate openings: %llu\n",
                        static_cast<unsigned long long>(
                            solver.getSearchStats().det_heur_gate_openings));
            }
            if (directional_inference) {
                const auto& columns = solver.getBranchInferenceStats();
                const auto report = [&](const char* direction, auto member) {
                    mipx::MipDirectionalInferenceStats total;
                    for (const auto& column : columns) {
                        const auto& stats = column.*member;
                        total.samples += stats.samples;
                        total.tightened_columns += stats.tightened_columns;
                        total.cutoffs += stats.cutoffs;
                        total.skipped += stats.skipped;
                    }
                    const double mean = total.samples == 0
                                            ? 0.0
                                            : static_cast<double>(total.tightened_columns) /
                                                  static_cast<double>(total.samples);
                    log.log(
                        "Directional inference %s: samples=%llu mean=%.6f "
                        "cutoffs=%llu skipped=%llu\n",
                        direction, static_cast<unsigned long long>(total.samples), mean,
                        static_cast<unsigned long long>(total.cutoffs),
                        static_cast<unsigned long long>(total.skipped));
                };
                report("down", &mipx::MipBranchInferenceStats::down);
                report("up", &mipx::MipBranchInferenceStats::up);
            }
            bool print_inference_branch_coverage = inference_branch_coverage;
#ifdef MIPX_HAS_TBB
            if (print_inference_branch_coverage && num_threads > 1) {
                log.log("Inference branch coverage unavailable for parallel search\n");
                print_inference_branch_coverage = false;
            }
#endif
            if (print_inference_branch_coverage) {
                const auto& coverage = solver.getBranchInferenceCoverageStats();
                const auto report = [&](const char* name, const mipx::InferenceCoveragePath& path) {
                    for (std::size_t i = 0; i < path.bands.size(); ++i) {
                        const auto& band = path.bands[i];
                        log.log(
                            "Inference branch coverage %s gap=%d%%: decisions=%llu "
                            "close=%llu ready=%llu favored=%llu\n",
                            name, path.kGapPercents[i],
                            static_cast<unsigned long long>(path.decisions),
                            static_cast<unsigned long long>(band.close),
                            static_cast<unsigned long long>(band.ready),
                            static_cast<unsigned long long>(band.favored));
                    }
                };
                report("reliable", coverage.reliable);
                report("fallback", coverage.fallback);
            }
            if (inference_branch_score) {
#ifdef MIPX_HAS_TBB
                if (num_threads > 1) {
                    log.log("Inference branch score unavailable for parallel search\n");
                } else
#endif
                {
                    const auto& coverage = solver.getBranchInferenceCoverageStats();
                    const auto report = [&](const char* name,
                                            const mipx::InferenceCoveragePath& path) {
                        log.log("Inference branch score %s: changes=%llu survived=%llu\n", name,
                                static_cast<unsigned long long>(path.selection_changes),
                                static_cast<unsigned long long>(path.surviving_changes));
                    };
                    report("reliable", coverage.reliable);
                    report("fallback", coverage.fallback);
                }
            }
            if (branch_cutoff_stats) {
#ifdef MIPX_HAS_TBB
                if (num_threads > 1) {
                    log.log("Child LP cutoff statistics unavailable for parallel search\n");
                } else
#endif
                {
                    const auto& columns = solver.getBranchCutoffStats();
                    const auto report = [&](const char* name, auto member) {
                        mipx::MipDirectionalCutoffStats total;
                        for (const auto& column : columns) {
                            total.qualified_attempts += (column.*member).qualified_attempts;
                            total.cutoffs += (column.*member).cutoffs;
                        }
                        log.log("Child LP cutoffs %s: qualified_attempts=%llu cutoffs=%llu\n", name,
                                static_cast<unsigned long long>(total.qualified_attempts),
                                static_cast<unsigned long long>(total.cutoffs));
                    };
                    report("down", &mipx::MipBranchCutoffStats::down);
                    report("up", &mipx::MipBranchCutoffStats::up);
                }
            }
            if (serial_node_policy) {
                const auto& stats = solver.getSerialNodePolicyStats();
                if (stats.policy) {
                    log.log(
                        "Serial node policy %s: selected=%d closed=%d queue_pruned=%d "
                        "incumbents=%d restarts=%d proof_bound_gain=%.10g proof_completed=%s\n",
                        serial_node_policy_name, stats.nodes_selected, stats.nodes_closed,
                        stats.nodes_pruned_from_queue, stats.incumbents_accepted, stats.restarts,
                        stats.proof_bound_improvement, stats.proof_completed ? "yes" : "no");
                } else {
                    log.log("Serial node policy %s: no serial queue processed\n",
                            serial_node_policy_name);
                }
            }
            log.log("Time: %.2fs\n", result.time_seconds);
        } else {
            // LP solve — print banner from CLI.
            log.log("mipx v0.3\n");

            unsigned logical = std::thread::hardware_concurrency();
            const char* tbb_str = "";
            const char* simd_str = "";
#ifdef MIPX_HAS_TBB
            tbb_str = ", TBB";
#endif
#ifdef __AVX512F__
            simd_str = ", AVX-512";
#elif defined(__AVX2__)
            simd_str = ", AVX2";
#elif defined(__AVX__)
            simd_str = ", AVX";
#elif defined(__SSE4_2__)
            simd_str = ", SSE4.2";
#endif
            log.log("Thread count: %u logical processors, using up to 1 thread%s%s\n", logical,
                    tbb_str, simd_str);

            // Presolve.
            // Activity tightening can box one-sided columns, adding rows to
            // the barrier's standard form or degrading its crossover basis (#444).
            presolve_opts.enable_activity_bound_tightening = lp_mode != LpMode::Barrier;
            mipx::Presolver presolver;
            presolver.setOptions(presolve_opts);
            auto working = lp;
            bool did_presolve = false;
            if (presolve) {
                working = presolver.presolve(lp);
                const auto& stats = presolver.stats();
                // A presolve that stopped on infeasibility leaves a half-reduced
                // model with no postsolve to match. Solve the original instead,
                // so the verdict comes from the LP solver on the real model.
                if (presolver.isInfeasible()) {
                    working = lp;
                } else if (stats.vars_removed > 0 || stats.rows_removed > 0) {
                    did_presolve = true;
                } else {
                    working = lp;
                }
            }

            log.log("Solving LP with:\n");
            log.log("  %d rows, %d cols, %d nonzeros\n", working.num_rows, working.num_cols,
                    working.matrix.numNonzeros());
            log.log("\n");

            if (did_presolve) {
                const auto& stats = presolver.stats();
                log.log(
                    "Presolve: %d vars removed, %d rows removed, "
                    "%d bounds tightened, %d rounds (%d changed), %.3fs "
                    "[rules: %s] [examined: %d rows, %d cols]\n\n",
                    stats.vars_removed, stats.rows_removed, stats.bounds_tightened, stats.rounds,
                    stats.rounds_with_changes, stats.time_seconds,
                    mipx::formatPresolveRuleChanges(stats).c_str(), stats.rows_examined,
                    stats.cols_examined);
            }

            struct LpRun {
                mipx::LpResult result;
                std::vector<mipx::Real> primals;
                bool used_gpu = false;
                std::string error;
                // Set by the dual simplex only: which pivot-row kernel ran.
                std::optional<mipx::DualSimplexSolver::RowPriceStats> row_price;
                std::optional<mipx::DualSimplexSolver::DseFallbackStats> dse_fallbacks;
            };
            // One solve of `model` on the chosen backend; a negative limit is none.
            auto solveLp = [&](const mipx::LpProblem& model, double seconds) {
                LpRun run;
                if (lp_mode == LpMode::Barrier) {
                    mipx::BarrierSolver solver;
                    mipx::BarrierOptions bopts;
                    bopts.verbose = verbose;
                    bopts.algorithm = barrier_augmented ? mipx::BarrierAlgorithm::CpuAugmented
                                      : barrier_sparse_ne
                                          ? mipx::BarrierAlgorithm::CpuSparseCholesky
                                      : barrier_gpu ? mipx::BarrierAlgorithm::Auto
                                                    : mipx::BarrierAlgorithm::CpuCholesky;
                    bopts.ordering = barrier_ordering;
                    bopts.dualize = barrier_dualize;
                    bopts.ne_max_dense_cols = barrier_dense_cols;
                    bopts.ne_lower_updates = barrier_ne_lower_updates;
                    bopts.gpu_ne_max_dense_cols = barrier_gpu_dense_cols;
                    bopts.implicit_upper_bounds = barrier_implicit_bounds;
                    bopts.crossover_pushes = barrier_crossover_pushes;
                    bopts.gondzio_correctors = barrier_gondzio;
                    bopts.homogeneous = barrier_hsd;
                    bopts.initial_point = barrier_initial_point;
                    bopts.threads = std::max(1, num_threads);
                    bopts.max_solve_seconds = seconds;  // #275
                    solver.setOptions(bopts);
                    solver.load(model);
                    run.result = solver.solve();
                    run.primals = solver.getPrimalValues();
                    run.used_gpu = solver.usedGpu();
                    run.error = solver.lastError();
                    return run;
                }
                if (lp_mode == LpMode::Pdlp) {
                    mipx::PdlpSolver solver;
                    auto popts = pdlp_options;
                    popts.verbose = verbose;
                    popts.use_gpu = barrier_gpu;
                    popts.gpu_min_rows = barrier_gpu_min_rows;
                    popts.gpu_min_nnz = barrier_gpu_min_nnz;
                    popts.max_solve_seconds = seconds;
                    solver.setOptions(popts);
                    solver.load(model);
                    run.result = solver.solve();
                    run.primals = solver.getPrimalValues();
                    run.used_gpu = solver.usedGpu();
                    return run;
                }
                mipx::DualSimplexSolver solver;
                solver.setVerbose(verbose);
                mipx::DualSimplexOptions dopts = solver.getOptions();
                dopts.enable_idiot_crash = dual_idiot_crash;
                if (dual_primal_devex_override >= 0) {
                    dopts.enable_primal_devex = dual_primal_devex_override != 0;
                }
                if (dual_bfrt_override >= 0) {
                    if (dual_bfrt_override != 0) {
                        dopts.enable_bfrt = true;
                    } else {
                        dopts.enable_bfrt = false;
                        dopts.enable_auto_bfrt_wide = false;
                    }
                }
                if (dual_lu_update_limit != std::numeric_limits<mipx::Int>::min()) {
                    dopts.lu_update_limit = dual_lu_update_limit;
                }
                if (!std::isnan(dual_lu_ft_drop_tolerance)) {
                    dopts.lu_ft_drop_tolerance = dual_lu_ft_drop_tolerance;
                }
                dopts.force_dense_pivot_row = dual_dense_row_price;
                dopts.row_price_actual_work = dual_row_price_actual_work;
                dopts.retain_dse_weights = dual_retain_dse_weights;
                if (seconds >= 0.0) {
                    dopts.max_solve_seconds = seconds;
                }
                solver.setOptions(dopts);
                solver.load(model);
                run.result = solver.solve();
                run.primals = solver.getPrimalValues();
                run.row_price = solver.getRowPriceStats();
                run.dse_fallbacks = solver.getDseFallbackStats();
                return run;
            };

            auto t0 = std::chrono::steady_clock::now();
            auto secondsSince = [](std::chrono::steady_clock::time_point start) {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                    .count();
            };
            LpRun run = solveLp(working, time_limit);

            // Presolve holds its reductions to its own tolerances, so the
            // postsolved point can miss the model as loaded (#244). Check it
            // there, as the MIP path does (#257). A backend's Optimal holds
            // rows only to its own tolerance, which can be looser than the
            // check's 1e-6 (PDLP's is relative), and solving again without
            // presolve would give a point no better. So only a miss presolve
            // added, past the reduced point's own by more than 1e-6, sends the
            // model to be solved again without presolve in the time left; that
            // answer stands as it would with --no-presolve.
            double obj = run.result.objective;
            if (did_presolve && run.result.status == mipx::Status::Optimal) {
                auto postsolved = presolver.postsolve(run.primals);
                const mipx::Real loaded_miss = mipx::pointHoldsModel(lp, postsolved)
                                                   ? 0.0
                                                   : mipx::pointViolation(lp, postsolved);
                const mipx::Real reduced_miss =
                    loaded_miss > 0.0 ? mipx::pointViolation(working, run.primals) : 0.0;
                if (loaded_miss <= reduced_miss + 1e-6) {
                    obj = lp.obj_offset;
                    for (mipx::Index j = 0; j < lp.num_cols; ++j) {
                        obj += lp.obj[j] * postsolved[j];
                    }
                } else {
                    std::fflush(stdout);
                    log.log(
                        "Postsolved point misses the model as loaded by %.3g (the presolved "
                        "point its model by %.3g); solving again without presolve\n",
                        loaded_miss, reduced_miss);
                    const mipx::Int iterations = run.result.iterations;
                    const double work_units = run.result.work_units;
                    const auto row_price = run.row_price;
                    const auto dse_fallbacks = run.dse_fallbacks;
                    const double time_left =
                        time_limit >= 0.0 ? time_limit - secondsSince(t0) : -1.0;
                    if (time_limit >= 0.0 && time_left <= 0.0) {
                        run.result.status = mipx::Status::TimeLimit;
                    } else {
                        run = solveLp(lp, time_left);
                        run.result.iterations += iterations;
                        run.result.work_units += work_units;
                        if (run.row_price && row_price) {
                            run.row_price->sparse_assemblies += row_price->sparse_assemblies;
                            run.row_price->dense_assemblies += row_price->dense_assemblies;
                            run.row_price->fill_in_fallbacks += row_price->fill_in_fallbacks;
                        }
                        if (run.dse_fallbacks && dse_fallbacks) {
                            run.dse_fallbacks->cost += dse_fallbacks->cost;
                            run.dse_fallbacks->degenerate += dse_fallbacks->degenerate;
                        }
                        obj = run.result.objective;
                    }
                }
            }
            const mipx::LpResult& result = run.result;
            const double solve_seconds = secondsSince(t0);

            if (lp_mode == LpMode::Barrier && (verbose || print_backend)) {
                log.log("Barrier backend: %s\n", run.used_gpu ? "GPU" : "CPU");
            }
            if (lp_mode == LpMode::Pdlp && (verbose || print_backend)) {
                log.log("PDLP backend: %s\n", run.used_gpu ? "GPU" : "CPU");
            }
            if (run.row_price && verbose) {
                // Which pivot-row kernel ran, over every solve above, so a
                // --dual-dense-row-price benchmark can confirm it got the
                // dense one (#178). Fill-in fallbacks are among the dense.
                const auto& rps = *run.row_price;
                std::fflush(stdout);  // the solver's printf log comes first
                log.log("Row price: %d sparse, %d dense (%d after a fill-in fallback)\n",
                        static_cast<int>(rps.sparse_assemblies),
                        static_cast<int>(rps.dense_assemblies),
                        static_cast<int>(rps.fill_in_fallbacks));
            }
            if (run.dse_fallbacks && verbose) {
                // The cost check reads work units, so a benchmark of
                // --dual-row-price-actual-work compares these (#379).
                log.log("DSE fallbacks: %d on cost, %d on degeneracy\n",
                        static_cast<int>(run.dse_fallbacks->cost),
                        static_cast<int>(run.dse_fallbacks->degenerate));
            }

            std::fflush(stdout);  // flush LP solver's printf output before Logger write()
            log.log("\n");
            const char* status_text = "Error";
            switch (result.status) {
                case mipx::Status::Optimal:
                    status_text = "Optimal";
                    log.log("Optimal: %.10e (%d iterations)\n", obj, result.iterations);
                    break;
                case mipx::Status::Infeasible:
                    status_text = "Infeasible";
                    log.log("Status: Infeasible\n");
                    break;
                case mipx::Status::Unbounded:
                    status_text = "Unbounded";
                    log.log("Status: Unbounded\n");
                    break;
                case mipx::Status::IterLimit:
                    status_text = "Iteration limit";
                    log.log("Status: Iteration limit\n");
                    break;
                case mipx::Status::TimeLimit:
                    status_text = "Time limit";
                    log.log("Status: Time limit\n");
                    break;
                default:
                    status_text = "Error";
                    log.log("Status: Error\n");
                    if (!run.error.empty()) {
                        log.log("Error detail: %s\n", run.error.c_str());
                    }
                    break;
            }
            // Stable summary lines consumed by benchmark scripts.
            log.log("Status: %s\n", status_text);
            // Only an optimal result carries a primal point to report. On any
            // other exit result.objective is the solver's working value (for
            // the dual simplex, a dual objective in presolved space), which
            // reads like a result and is not one (#206).
            if (result.status == mipx::Status::Optimal) {
                log.log("Objective: %.10e\n", obj);
            } else {
                log.log("Objective: none\n");
            }
            log.log("Iterations: %d\n", result.iterations);
            log.log("Work units: %.2f\n", result.work_units);
            log.log("Time: %.2fs\n", solve_seconds);
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
