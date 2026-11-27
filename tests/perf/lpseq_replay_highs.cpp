// mipx-lpseq-replay-highs: replay an LP sequence in the "lpseq 1"
// format on HiGHS, through its API, so it can run warm (#268). An opt-in
// bench target (MIPX_BENCH_HIGHS, OFF by default); the mipx library never
// links HiGHS. It reads the ops with the same parser as mipx-lpseq-replay
// (lpseq_format.h) and writes the same CSV:
//   seq,arm,status,objective,recorded,rel_err,iterations,seconds,reloaded
//
// Arms: simplex-warm (dual simplex) and primal-warm (primal simplex) keep
// one Highs object across the changes.
// Arms simplex-cold, ipx-cold and hipo-cold solve each LP on a fresh Highs
// (interior points without crossover); pami-warm is simplex-warm on HiGHS's
// parallel dual simplex (PAMI, simplex_strategy 3), for use with --threads
// (parallel=on alone does not select it in HiGHS 1.16; its width stays at
// simplex_max_concurrency, 8, whatever --threads says). One thread unless
// --threads N, presolve off unless --presolve. Exit 0 when every solve
// matches the recorded status and objective (simplex --rel-tol 1e-6,
// interior point --ipm-rel-tol 1e-5), 1 when one does not, 2 on a usage or
// format error.

#include "lpseq_format.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <Highs.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace lpseq = mipx::lpseq;

namespace {

// The parser's index vectors go to HiGHS as they are.
static_assert(sizeof(HighsInt) == sizeof(int), "HiGHS built with 64-bit HighsInt");

struct Arm {
    std::string name;
    bool warm = false;
    std::string solver;  // HiGHS "solver" option: simplex, ipx, hipo
    int strategy = 1;    // simplex_strategy: 1 dual, 3 dual PAMI, 4 primal
    std::unique_ptr<Highs> highs;
};

struct Options {
    std::string dir;
    std::vector<Arm> arms;
    bool presolve = false;
    int threads = 1;
    double rel_tol = 1e-6;
    double ipm_rel_tol = 1e-5;
    std::string csv;
};

// HiGHS reads an LP through limits mipx has none of: bounds and costs past
// infinite_bound / infinite_cost (1e20) are infinite, entries below
// small_matrix_value (1e-9) dropped, entries past large_matrix_value (1e15)
// refused. Widened to their ends, so HiGHS reads the LP mipx reads; what it
// would still drop, entries in (0, kHighsSmallestEntry], is refused instead.
constexpr double kHighsSmallestEntry = 1e-12;

bool widenLimits(Highs& h) {
    return h.setOptionValue("infinite_bound", kHighsInf) == HighsStatus::kOk &&
           h.setOptionValue("infinite_cost", kHighsInf) == HighsStatus::kOk &&
           h.setOptionValue("large_matrix_value", kHighsInf) == HighsStatus::kOk &&
           h.setOptionValue("small_matrix_value", kHighsSmallestEntry) == HighsStatus::kOk;
}

void checkEntries(const std::vector<double>& values, const std::string& what) {
    for (const double v : values) {
        if (v != 0.0 && std::abs(v) <= kHighsSmallestEntry) {
            lpseq::fail(what + ": a coefficient below 1e-12, which HiGHS drops");
        }
    }
}

void configure(Highs& h, const Arm& arm, const Options& opt) {
    const bool ok = widenLimits(h) && h.setOptionValue("output_flag", false) == HighsStatus::kOk &&
                    h.setOptionValue("threads", opt.threads) == HighsStatus::kOk &&
                    h.setOptionValue("presolve", opt.presolve ? "on" : "off") == HighsStatus::kOk &&
                    h.setOptionValue("solver", arm.solver) == HighsStatus::kOk &&
                    h.setOptionValue("simplex_strategy", arm.strategy) == HighsStatus::kOk &&
                    h.setOptionValue("run_crossover", "off") == HighsStatus::kOk;
    if (!ok) {
        lpseq::fail("HiGHS refused an option of arm " + arm.name);
    }
}

const char* statusName(HighsModelStatus s) {
    switch (s) {
        case HighsModelStatus::kOptimal:
            return "optimal";
        case HighsModelStatus::kInfeasible:
            return "infeasible";
        case HighsModelStatus::kUnbounded:
            return "unbounded";
        default:
            return "error";
    }
}

void check(HighsStatus s, const char* what) {
    if (s == HighsStatus::kError) {
        lpseq::fail(std::string("HiGHS refused ") + what);
    }
}

// Apply a change to `h` (the mirror or a warm arm).
void apply(const lpseq::Op& op, Highs& h) {
    using lpseq::OpKind;
    const auto k = static_cast<HighsInt>(op.kind == OpKind::AddCols || op.kind == OpKind::AddRows
                                             ? op.lower.size()
                                             : op.indices.size());
    const auto nnz = static_cast<HighsInt>(op.indices.size());
    switch (op.kind) {
        case OpKind::AddCols:
            check(h.addCols(k, op.cost.data(), op.lower.data(), op.upper.data(), nnz,
                            op.starts.data(), op.indices.data(), op.values.data()),
                  "add_cols");
            break;
        case OpKind::AddRows:
            check(h.addRows(k, op.lower.data(), op.upper.data(), nnz, op.starts.data(),
                            op.indices.data(), op.values.data()),
                  "add_rows");
            break;
        case OpKind::DelCols:
            check(h.deleteCols(k, op.indices.data()), "del_cols");
            break;
        case OpKind::DelRows:
            check(h.deleteRows(k, op.indices.data()), "del_rows");
            break;
        case OpKind::ColBounds:
        case OpKind::RowBounds:
            for (std::size_t t = 0; t < op.at.size(); ++t) {
                const double lo = op.lower[t];
                const double up = op.upper[t];
                check(op.kind == OpKind::ColBounds ? h.changeColBounds(op.at[t], lo, up)
                                                   : h.changeRowBounds(op.at[t], lo, up),
                      "bounds");
            }
            break;
        case OpKind::ColCosts:
            for (std::size_t t = 0; t < op.at.size(); ++t) {
                check(h.changeColCost(op.at[t], op.cost[t]), "col_costs");
            }
            break;
        case OpKind::Coeffs:
            for (std::size_t t = 0; t < op.at.size(); ++t) {
                check(h.changeCoeff(op.at[t], op.col[t], op.values[t]), "coeffs");
            }
            break;
        case OpKind::Solve:
            break;
    }
}

int run(Options& opt) {
    // Bounds pass to HiGHS as the parser gives them: its kHighsInf (not
    // constexpr) is the same IEEE infinity.
    if (kHighsInf != lpseq::kInf) {
        lpseq::fail("this HiGHS's infinity is not IEEE infinity");
    }
    lpseq::checkManifest(opt.dir);
    // The mirror: a Highs that only holds the LP, given to the cold arms.
    Highs model;
    if (!widenLimits(model) || model.setOptionValue("output_flag", false) != HighsStatus::kOk) {
        lpseq::fail("HiGHS refused an option of the reader");
    }
    const std::string master = opt.dir + "/master_0000.mps";
    if (model.readModel(master) == HighsStatus::kError) {
        lpseq::fail(master + ": HiGHS cannot read it");
    }
    const HighsLp& lp = model.getLp();
    checkEntries(lp.a_matrix_.value_, master);
    lpseq::checkNames(lp.row_names_, static_cast<int>(lp.num_row_), 'r', master);
    lpseq::checkNames(lp.col_names_, static_cast<int>(lp.num_col_), 'c', master);
    if (lp.sense_ != ObjSense::kMinimize) {
        lpseq::fail(master + ": the format minimises");
    }
    for (Arm& arm : opt.arms) {
        if (arm.warm) {
            arm.highs = std::make_unique<Highs>();
            configure(*arm.highs, arm, opt);
            check(arm.highs->passModel(lp), "the master");
        }
    }

    std::ofstream csv_file;
    std::ostream* out = &std::cout;
    if (!opt.csv.empty()) {
        csv_file.open(opt.csv);
        if (!csv_file) {
            lpseq::fail(opt.csv + ": cannot write");
        }
        out = &csv_file;
    }
    *out << lpseq::kCsvHeader;

    std::ifstream ops(opt.dir + "/ops.txt");
    if (!ops) {
        lpseq::fail(opt.dir + "/ops.txt: cannot open");
    }
    bool all_match = true;
    lpseq::Op op;
    while (lpseq::readOp(ops, static_cast<int>(model.getNumRow()),
                         static_cast<int>(model.getNumCol()), op)) {
        if (op.kind != lpseq::OpKind::Solve) {
            checkEntries(op.values, "ops.txt");
            apply(op, model);
            for (Arm& arm : opt.arms) {
                if (arm.warm) {
                    apply(op, *arm.highs);
                }
            }
            continue;
        }
        for (Arm& arm : opt.arms) {
            std::unique_ptr<Highs> cold;
            Highs* h = arm.highs.get();
            if (!arm.warm) {
                cold = std::make_unique<Highs>();
                configure(*cold, arm, opt);
                check(cold->passModel(model.getLp()), "the LP");
                h = cold.get();
            }
            const auto t0 = std::chrono::steady_clock::now();
            h->run();
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const HighsInfo& info = h->getInfo();
            const double tol = arm.solver == "simplex" ? opt.rel_tol : opt.ipm_rel_tol;
            double err = 0.0;
            std::string status = statusName(h->getModelStatus());
            // With crossover off, simplex iterations on an interior point
            // arm mean HiGHS fell back to simplex (e.g. HiPO without its
            // extras, or an imprecise IPM finished by simplex): reported as
            // an error, not booked as the method's. A solve presolve or the
            // trivial cases settle (no iterations of either) stands.
            if (arm.solver != "simplex" && info.simplex_iteration_count > 0) {
                status = "error";
            }
            all_match =
                lpseq::matches(op, status, info.objective_function_value, tol, err) && all_match;
            const long iterations =
                static_cast<long>(std::max<HighsInt>(0, info.simplex_iteration_count)) +
                std::max<HighsInt>(0, info.ipm_iteration_count);
            lpseq::writeRow(*out, op, arm.name, status, info.objective_function_value, err,
                            iterations, secs, false);
        }
    }
    return all_match ? 0 : 1;
}

void usage() {
    std::cerr << "usage: mipx-lpseq-replay-highs DIR "
                 "[--arm simplex-warm|pami-warm|primal-warm|simplex-cold|ipx-cold|hipo-cold]... "
                 "[--presolve] [--threads N] [--rel-tol T] [--ipm-rel-tol T] [--csv FILE]\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--arm" && i + 1 < argc) {
            const std::string v = argv[++i];
            Arm arm;
            if (v == "simplex-warm") {
                arm = {v, true, "simplex", 1, nullptr};
            } else if (v == "pami-warm") {
                arm = {v, true, "simplex", 3, nullptr};
            } else if (v == "primal-warm") {
                arm = {v, true, "simplex", 4, nullptr};
            } else if (v == "simplex-cold") {
                arm = {v, false, "simplex", 1, nullptr};
            } else if (v == "ipx-cold") {
                arm = {v, false, "ipx", 1, nullptr};
            } else if (v == "hipo-cold") {
                arm = {v, false, "hipo", 1, nullptr};
            } else {
                usage();
                return 2;
            }
            opt.arms.push_back(std::move(arm));
        } else if (a == "--presolve") {
            opt.presolve = true;
        } else if (a == "--threads" && i + 1 < argc) {
            opt.threads = std::atoi(argv[++i]);
            if (opt.threads < 1) {
                usage();
                return 2;
            }
        } else if (a == "--rel-tol" && i + 1 < argc) {
            opt.rel_tol = std::atof(argv[++i]);
        } else if (a == "--ipm-rel-tol" && i + 1 < argc) {
            opt.ipm_rel_tol = std::atof(argv[++i]);
        } else if (a == "--csv" && i + 1 < argc) {
            opt.csv = argv[++i];
        } else if (!a.empty() && a[0] != '-' && opt.dir.empty()) {
            opt.dir = a;
        } else {
            usage();
            return 2;
        }
    }
    if (opt.dir.empty()) {
        usage();
        return 2;
    }
    if (opt.arms.empty()) {
        opt.arms.push_back({"simplex-warm", true, "simplex", 1, nullptr});
    }
    try {
        return run(opt);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
}
