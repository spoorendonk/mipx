// mipx-presolve-bench-highs: HiGHS's presolve alone, without the solve, on
// one LP file, beside mipx-presolve-bench (#287). An opt-in bench target
// (MIPX_BENCH_HIGHS, OFF by default); the mipx library never links HiGHS.
//
// The model is read with mipx's reader and passed to HiGHS through its API,
// so both presolves see the same LP; HiGHS's read limits are widened as in
// mipx-lpseq-replay-highs, and an entry HiGHS would still drop is refused.
// Highs::presolve() runs --repeats times, each on a fresh Highs, one thread,
// rule logging off; `seconds` is the fastest call, `seconds_median` the
// median. One more, untimed run with presolve_rule_logging on gives the rows
// and columns each of HiGHS's rules removed (<rule>_rows, <rule>_cols; HiGHS
// logs them for LPs only).
//
// Status: HiGHS's presolve status (reduced, reduced_to_empty, not_reduced,
// infeasible, unbounded_or_infeasible, timeout, error). Exit 0, 1 when
// HiGHS's presolve errs, 2 on a usage or read error.

#include "presolve_bench_common.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <Highs.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace mipx;
namespace pb = mipx::presolve_bench;

namespace {

static_assert(sizeof(HighsInt) == sizeof(Index), "HiGHS built with 64-bit HighsInt");

const std::string kUsage =
    "Usage: mipx-presolve-bench-highs <model> [--name NAME] [--format auto|mps|lp]\n"
    "       [--repeats N] [--csv FILE]\n";

// As in mipx-lpseq-replay-highs: HiGHS's limits widened to their ends, so it
// takes the LP mipx read; what it would still drop is refused.
constexpr double kHighsSmallestEntry = 1e-12;

bool widenLimits(Highs& h) {
    return h.setOptionValue("infinite_bound", kHighsInf) == HighsStatus::kOk &&
           h.setOptionValue("infinite_cost", kHighsInf) == HighsStatus::kOk &&
           h.setOptionValue("large_matrix_value", kHighsInf) == HighsStatus::kOk &&
           h.setOptionValue("small_matrix_value", kHighsSmallestEntry) == HighsStatus::kOk;
}

HighsLp toHighs(const LpProblem& lp) {
    HighsLp h;
    h.num_col_ = lp.num_cols;
    h.num_row_ = lp.num_rows;
    h.sense_ = lp.sense == Sense::Maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
    h.offset_ = lp.obj_offset;
    h.col_cost_ = lp.obj;
    h.col_lower_ = lp.col_lower;
    h.col_upper_ = lp.col_upper;
    h.row_lower_ = lp.row_lower;
    h.row_upper_ = lp.row_upper;
    if (lp.hasIntegers()) {
        h.integrality_.resize(lp.num_cols);
        for (Index j = 0; j < lp.num_cols; ++j) {
            h.integrality_[j] = lp.col_type[j] == VarType::Continuous ? HighsVarType::kContinuous
                                                                      : HighsVarType::kInteger;
        }
    }
    h.a_matrix_.format_ = MatrixFormat::kRowwise;
    h.a_matrix_.num_col_ = lp.num_cols;
    h.a_matrix_.num_row_ = lp.num_rows;
    const auto starts = lp.matrix.csr_row_starts();
    const auto cols = lp.matrix.csr_col_indices();
    const auto values = lp.matrix.csr_values();
    h.a_matrix_.start_.assign(starts.begin(), starts.end());
    h.a_matrix_.index_.assign(cols.begin(), cols.end());
    h.a_matrix_.value_.assign(values.begin(), values.end());
    if (h.a_matrix_.start_.empty()) {
        h.a_matrix_.start_.assign(1, 0);
    }
    for (const double v : h.a_matrix_.value_) {
        if (v != 0.0 && std::abs(v) <= kHighsSmallestEntry) {
            std::fprintf(stderr, "Error: a coefficient below 1e-12, which HiGHS drops\n");
            std::exit(pb::kExitUsage);
        }
    }
    return h;
}

/// A Highs holding `lp`, set for a presolve-only run.
std::unique_ptr<Highs> loaded(const HighsLp& lp, bool rule_logging) {
    auto h = std::make_unique<Highs>();
    const bool ok = widenLimits(*h) &&
                    h->setOptionValue("output_flag", false) == HighsStatus::kOk &&
                    h->setOptionValue("threads", 1) == HighsStatus::kOk &&
                    h->setOptionValue("presolve", "on") == HighsStatus::kOk &&
                    h->setOptionValue("presolve_rule_logging", rule_logging) == HighsStatus::kOk;
    if (!ok) {
        std::fprintf(stderr, "Error: HiGHS refused an option\n");
        std::exit(pb::kExitUsage);
    }
    if (h->passModel(lp) == HighsStatus::kError) {
        std::fprintf(stderr, "Error: HiGHS refused the model\n");
        std::exit(pb::kExitUsage);
    }
    return h;
}

const char* statusName(HighsPresolveStatus s) {
    switch (s) {
        case HighsPresolveStatus::kNotReduced:
            return "not_reduced";
        case HighsPresolveStatus::kInfeasible:
            return "infeasible";
        case HighsPresolveStatus::kUnboundedOrInfeasible:
            return "unbounded_or_infeasible";
        case HighsPresolveStatus::kReduced:
            return "reduced";
        case HighsPresolveStatus::kReducedToEmpty:
            return "reduced_to_empty";
        case HighsPresolveStatus::kTimeout:
            return "timeout";
        default:
            return "error";
    }
}

// HiGHS's presolve rules (v1.16.0), by their enum, so a HiGHS that renumbers
// or adds one fails to build here instead of mislabelling a column: an entry
// left out is value-initialized, {0, nullptr}, and breaks the order check.
constexpr std::array<std::pair<HighsInt, const char*>, kPresolveRuleCount> kRules = {{
    {kPresolveRuleEmptyRow, "empty_row"},
    {kPresolveRuleSingletonRow, "singleton_row"},
    {kPresolveRuleRedundantRow, "redundant_row"},
    {kPresolveRuleEmptyCol, "empty_col"},
    {kPresolveRuleFixedCol, "fixed_col"},
    {kPresolveRuleDominatedCol, "dominated_col"},
    {kPresolveRuleForcingRow, "forcing_row"},
    {kPresolveRuleForcingCol, "forcing_col"},
    {kPresolveRuleFreeColSubstitution, "free_col_substitution"},
    {kPresolveRuleDoubletonEquation, "doubleton_equation"},
    {kPresolveRuleDependentEquations, "dependent_equations"},
    {kPresolveRuleDependentFreeCols, "dependent_free_cols"},
    {kPresolveRuleAggregator, "aggregator"},
    {kPresolveRuleParallelRowsAndCols, "parallel_rows_and_cols"},
    {kPresolveRuleSparsify, "sparsify"},
    {kPresolveRuleProbing, "probing"},
    {kPresolveRuleEnumeration, "enumeration"},
    {kPresolveRuleDualFixing, "dual_fixing"},
    {kPresolveRuleZeroCostSingleton, "zero_cost_singleton"},
    {kPresolveRuleColStuffing, "col_stuffing"},
    {kPresolveRuleDualFixProbing, "dual_fix_probing"},
    {kPresolveRuleInitialSweep, "initial_sweep"},
    {kPresolveRuleFourierMotzkin, "fourier_motzkin"},
    {kPresolveRuleWeaklyDominatedCol, "weakly_dominated_col"},
}};

constexpr bool rulesInOrder() {
    for (std::size_t k = 0; k < kRules.size(); ++k) {
        if (kRules[k].first != static_cast<HighsInt>(k)) {
            return false;
        }
    }
    return true;
}
static_assert(rulesInOrder(), "kRules must list HiGHS's presolve rules in enum order");

}  // namespace

int main(int argc, char** argv) {
    pb::CommonOptions opt;
    for (int i = 1; i < argc; ++i) {
        if (!pb::parseCommon(argc, argv, i, opt, kUsage)) {
            pb::usageError(kUsage, std::string("unknown argument ") + argv[i]);
        }
    }
    if (opt.path.empty()) {
        pb::usageError(kUsage, "no model file");
    }
    const std::string instance = opt.name.empty() ? pb::instanceName(opt.path) : opt.name;
    const LpProblem lp = pb::readOrExit(opt);
    const HighsLp hlp = toHighs(lp);

    std::vector<double> times;
    std::unique_ptr<Highs> last;
    for (int rep = 0; rep < opt.repeats; ++rep) {
        last = loaded(hlp, false);
        HighsStatus rc = HighsStatus::kOk;
        times.push_back(pb::secondsOf([&] { rc = last->presolve(); }));
        if (rc == HighsStatus::kError) {
            break;
        }
    }
    const HighsPresolveStatus presolve_status = last->getModelPresolveStatus();
    const std::string status = statusName(presolve_status);

    pb::Row row;
    pb::addCommon(row, instance, "highs", status, times, lp);
    const HighsLp& reduced = last->getPresolvedLp();
    if (presolve_status == HighsPresolveStatus::kReduced ||
        presolve_status == HighsPresolveStatus::kReducedToEmpty) {
        row.addSizes("after", reduced.num_row_, reduced.num_col_, reduced.a_matrix_.numNz());
    } else if (presolve_status == HighsPresolveStatus::kNotReduced) {
        row.addSizes("after", lp);
    } else {
        row.addSizes("after", -1, -1, -1);
    }

    const auto logged = loaded(hlp, true);
    logged->presolve();
    const HighsPresolveLog& log = logged->getPresolveLog();
    for (const auto& [rule, name] : kRules) {
        const bool have = static_cast<std::size_t>(rule) < log.rule.size();
        row.add(std::string(name) + "_rows",
                have ? static_cast<long long>(log.rule[rule].row_removed) : 0LL);
        row.add(std::string(name) + "_cols",
                have ? static_cast<long long>(log.rule[rule].col_removed) : 0LL);
    }
    row.emit(opt.csv);
    return status == "error" ? pb::kExitCheckFailed : pb::kExitOk;
}
