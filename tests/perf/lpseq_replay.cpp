// mipx-lpseq-replay: replay an LP sequence on mipx's LP solvers and check
// each solve against its recorded objective (#267). See lpseq_format.h and
// tests/perf/README.md for the format.
//
// Arms: dual-warm keeps one DualSimplexSolver and applies the ops to it (its
// basis carries over); dual-cold, barrier, barrier-hsd and pdlp load the LP
// afresh for each solve. A coeffs op, which LpSolver has no call for, makes dual-warm
// reload the LP at that point, keeping its basis (counted as a reload).
//
// Output: one CSV line per (solve, arm) on stdout, or to --csv FILE:
//   seq,arm,status,objective,recorded,rel_err,iterations,seconds,reloaded
// --dual-stats-csv FILE writes pivot counts for the dual arms separately.
// --edit-csv FILE times dual-warm's edit calls per op (#442):
//   after,arm,op,items,basic,seconds
// Exit 0 when every solve of every arm matches the recorded status and
// objective (relative to max(1, |recorded|), within --rel-tol, 1e-6, or for
// pdlp --pdlp-rel-tol, 1e-4), 1 when one does not, 2 on a usage or format
// error.

#include "lpseq_format.h"
#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/pdlp.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using namespace mipx;

namespace {

LpProblem readMaster(const std::string& path) {
    LpProblem lp = readMps(path);
    lpseq::checkNames(lp.row_names, lp.num_rows, 'r', path);
    lpseq::checkNames(lp.col_names, lp.num_cols, 'c', path);
    if (lp.sense != Sense::Minimize) {
        lpseq::fail(path + ": the format minimises");
    }
    return lp;
}

void rebuild(LpProblem& lp, std::vector<Triplet> trips) {
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(trips));
}

std::vector<Triplet> triplets(const LpProblem& lp) {
    std::vector<Triplet> t;
    t.reserve(static_cast<std::size_t>(lp.matrix.numNonzeros()));
    for (Index i = 0; i < lp.num_rows; ++i) {
        auto row = lp.matrix.row(i);
        for (Index p = 0; p < row.size(); ++p) {
            t.push_back({i, row.indices[p], row.values[p]});
        }
    }
    return t;
}

// The mirror: the LP as the ops leave it, for the cold arms and reloads.
void mirrorAddRows(LpProblem& lp, const lpseq::Op& op) {
    auto t = triplets(lp);
    const auto k = static_cast<Index>(op.lower.size());
    for (Index r = 0; r < k; ++r) {
        const Index end = (r + 1 < k) ? op.starts[r + 1] : static_cast<Index>(op.indices.size());
        for (Index p = op.starts[r]; p < end; ++p) {
            t.push_back({lp.num_rows + r, op.indices[p], op.values[p]});
        }
        lp.row_lower.push_back(op.lower[r]);
        lp.row_upper.push_back(op.upper[r]);
        lp.row_names.push_back("r" + std::to_string(lp.num_rows + r));
    }
    lp.num_rows += k;
    rebuild(lp, std::move(t));
}

void mirrorDelRows(LpProblem& lp, const std::vector<Index>& sorted_rows) {
    std::vector<Index> map(static_cast<std::size_t>(lp.num_rows), 0);
    Index next = 0;
    std::size_t r = 0;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (r < sorted_rows.size() && sorted_rows[r] == i) {
            map[i] = -1;
            ++r;
        } else {
            map[i] = next++;
        }
    }
    std::vector<Triplet> t;
    for (const auto& e : triplets(lp)) {
        if (map[e.row] >= 0) {
            t.push_back({map[e.row], e.col, e.val});
        }
    }
    std::vector<Real> lo, up;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (map[i] >= 0) {
            lo.push_back(lp.row_lower[i]);
            up.push_back(lp.row_upper[i]);
        }
    }
    lp.row_lower = std::move(lo);
    lp.row_upper = std::move(up);
    lp.num_rows = next;
    lp.row_names.resize(static_cast<std::size_t>(next));
    rebuild(lp, std::move(t));
}

// One pass: drop the entries at the changed positions (unique per op, which
// readOp checks), then add the changes that are not 0.
void mirrorSetCoeffs(LpProblem& lp, const lpseq::Op& op) {
    const auto key = [&](Index i, Index j) {
        return static_cast<std::int64_t>(i) * static_cast<std::int64_t>(lp.num_cols) + j;
    };
    std::unordered_set<std::int64_t> changed;
    for (std::size_t t = 0; t < op.at.size(); ++t) {
        changed.insert(key(op.at[t], op.col[t]));
    }
    auto t = triplets(lp);
    std::erase_if(t, [&](const Triplet& e) { return changed.contains(key(e.row, e.col)); });
    for (std::size_t c = 0; c < op.at.size(); ++c) {
        if (op.values[c] != 0.0) {
            t.push_back({op.at[c], op.col[c], op.values[c]});
        }
    }
    rebuild(lp, std::move(t));
}

enum class ArmKind { DualWarm, DualCold, Barrier, BarrierHsd, Pdlp };

struct Arm {
    ArmKind kind;
    std::string name;
    std::unique_ptr<DualSimplexSolver> warm;  // DualWarm only
    bool stale = false;                       // a coeffs op since its last solve
};

struct Options {
    std::string dir;
    std::vector<ArmKind> arms;
    bool primal_devex = false;
    bool primal_bound_perturbation = false;
    // CPU factor for both barrier arms (--barrier-ne); Auto, the solver's
    // default, unless set.
    BarrierAlgorithm barrier_algorithm = BarrierAlgorithm::Auto;
    Real rel_tol = 1e-6;
    Real pdlp_rel_tol = 1e-4;
    std::string csv;
    std::string dual_stats_csv;
    std::string refactor_csv;
    std::string edit_csv;
};

// --refactor-csv: a dual arm's factorizations of one solve by trigger (#517),
// DualSimplexSolver::RefactorStats in its field order. cold_start is 1 when a
// pass started from a crash basis (a dual-warm solve's cold retry included).
constexpr auto kRefactorCsvHeader =
    "seq,arm,iterations,cold_start,start,update_limit,update_cost,growth,pivot_rejected,"
    "pivot_mismatch,empty_ratio_test,no_bfrt_group,ray_guard,singular_recovery,"
    "degenerate_stall,bfrt_high_flip,stall_restart,cold_restore,unscaled_cleanup,edits\n";

void writeRefactorRow(std::ostream& out, const std::string& seq, const std::string& arm,
                      Int iterations, const DualSimplexSolver& dual) {
    const auto& r = dual.getRefactorStats();
    out << seq << ',' << arm << ',' << iterations << ','
        << (dual.basisStartStats().cold_start ? 1 : 0) << ',' << r.start << ',' << r.update_limit
        << ',' << r.update_cost << ',' << r.growth << ',' << r.pivot_rejected << ','
        << r.pivot_mismatch << ',' << r.empty_ratio_test << ',' << r.no_bfrt_group << ','
        << r.ray_guard << ',' << r.singular_recovery << ',' << r.degenerate_stall << ','
        << r.bfrt_high_flip << ',' << r.stall_restart << ',' << r.cold_restore << ','
        << r.unscaled_cleanup << ',' << r.edits << '\n';
}

DualSimplexOptions dualOptions(const Options& o) {
    DualSimplexOptions d;
    d.enable_primal_devex = o.primal_devex;
    d.enable_primal_bound_perturbation = o.primal_bound_perturbation;
    return d;
}

const char* statusName(Status s) {
    switch (s) {
        case Status::Optimal:
            return "optimal";
        case Status::Infeasible:
            return "infeasible";
        case Status::Unbounded:
            return "unbounded";
        default:
            return "error";
    }
}

std::unique_ptr<LpSolver> coldSolver(ArmKind kind, const Options& opt) {
    if (kind == ArmKind::DualCold) {
        auto d = std::make_unique<DualSimplexSolver>();
        d->setVerbose(false);
        d->setOptions(dualOptions(opt));
        return d;
    }
    if (kind == ArmKind::Barrier || kind == ArmKind::BarrierHsd) {
        auto b = std::make_unique<BarrierSolver>();
        BarrierOptions bo;
        bo.verbose = false;
        bo.algorithm = opt.barrier_algorithm;
        // The homogeneous self-dual model (#370) runs in the CPU IPM loop.
        bo.homogeneous = kind == ArmKind::BarrierHsd;
        b->setOptions(bo);
        return b;
    }
    auto p = std::make_unique<PdlpSolver>();
    PdlpOptions po;
    po.verbose = false;
    p->setOptions(po);
    return p;
}

const char* opName(lpseq::OpKind k) {
    using lpseq::OpKind;
    switch (k) {
        case OpKind::AddCols:
            return "add_cols";
        case OpKind::DelCols:
            return "del_cols";
        case OpKind::AddRows:
            return "add_rows";
        case OpKind::DelRows:
            return "del_rows";
        case OpKind::ColBounds:
            return "col_bounds";
        case OpKind::RowBounds:
            return "row_bounds";
        case OpKind::ColCosts:
            return "col_costs";
        case OpKind::Coeffs:
            return "coeffs";
        case OpKind::Solve:
            break;
    }
    return "solve";
}

// --edit-csv: the time of each warm solver's edit calls for one op (#442),
// outside the solve timer. after = the seq of the last solve ("-" before the
// first), items = the columns, rows or entries the op changes, basic = for
// del_cols the deleted columns that were basic (read before the call, untimed).
struct EditLog {
    std::ostream* out = nullptr;
    std::string after = "-";
};

std::size_t opItems(const lpseq::Op& op) {
    using lpseq::OpKind;
    switch (op.kind) {
        case OpKind::AddCols:
        case OpKind::AddRows:
            return op.starts.size();
        case OpKind::DelCols:
        case OpKind::DelRows:
            return op.indices.size();
        default:
            return op.at.size();
    }
}

// Apply a change to the mirror and to every warm solver.
void apply(const lpseq::Op& op, LpProblem& lp, std::vector<Arm>& arms, EditLog& log) {
    using lpseq::OpKind;
    switch (op.kind) {
        case OpKind::AddCols:
            appendColumns(lp, op.starts, op.indices, op.values, op.cost, op.lower, op.upper);
            break;
        case OpKind::AddRows:
            mirrorAddRows(lp, op);
            break;
        case OpKind::DelCols:
            removeColumns(lp, op.indices);
            break;
        case OpKind::DelRows:
            mirrorDelRows(lp, op.indices);
            break;
        case OpKind::ColBounds:
        case OpKind::RowBounds:
            for (std::size_t t = 0; t < op.at.size(); ++t) {
                auto& lo = op.kind == OpKind::ColBounds ? lp.col_lower : lp.row_lower;
                auto& up = op.kind == OpKind::ColBounds ? lp.col_upper : lp.row_upper;
                lo[op.at[t]] = op.lower[t];
                up[op.at[t]] = op.upper[t];
            }
            break;
        case OpKind::ColCosts:
            for (std::size_t t = 0; t < op.at.size(); ++t) {
                lp.obj[op.at[t]] = op.cost[t];
            }
            break;
        case OpKind::Coeffs:
            mirrorSetCoeffs(lp, op);
            break;
        case OpKind::Solve:
            return;
    }
    for (auto& arm : arms) {
        if (!arm.warm) {
            continue;
        }
        DualSimplexSolver& s = *arm.warm;
        std::size_t basic = 0;
        if (log.out != nullptr && op.kind == OpKind::DelCols) {
            const auto basis = s.getBasis();
            for (const int j : op.indices) {
                if (static_cast<std::size_t>(j) < basis.size() && basis[j] == BasisStatus::Basic) {
                    ++basic;
                }
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        switch (op.kind) {
            case OpKind::AddCols:
                s.addCols(op.starts, op.indices, op.values, op.cost, op.lower, op.upper);
                break;
            case OpKind::AddRows:
                s.addRows(op.starts, op.indices, op.values, op.lower, op.upper);
                break;
            case OpKind::DelCols:
                s.removeCols(op.indices);
                break;
            case OpKind::DelRows:
                s.removeRows(op.indices);
                break;
            case OpKind::ColBounds:
                for (std::size_t t = 0; t < op.at.size(); ++t) {
                    s.setColBounds(op.at[t], op.lower[t], op.upper[t]);
                }
                break;
            case OpKind::RowBounds:
                for (std::size_t t = 0; t < op.at.size(); ++t) {
                    s.setRowBounds(op.at[t], op.lower[t], op.upper[t]);
                }
                break;
            case OpKind::ColCosts:
                for (std::size_t t = 0; t < op.at.size(); ++t) {
                    s.setColObjective(op.at[t], op.cost[t]);
                }
                break;
            case OpKind::Coeffs:
                arm.stale = true;
                break;
            case OpKind::Solve:
                break;
        }
        // A coeffs op only marks the arm for a reload at the next solve,
        // which the solve row flags; it has no edit call to time.
        if (log.out != nullptr && op.kind != OpKind::Coeffs) {
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            *log.out << std::format("{},{},{},{},{},{:.9f}\n", log.after, arm.name, opName(op.kind),
                                    opItems(op), basic, secs);
        }
    }
}

int run(const Options& opt) {
    lpseq::checkManifest(opt.dir);
    LpProblem lp = readMaster(opt.dir + "/master_0000.mps");

    std::vector<Arm> arms;
    for (const ArmKind k : opt.arms) {
        Arm a{k, "", nullptr};
        switch (k) {
            case ArmKind::DualWarm:
                a.name = "dual-warm";
                a.warm = std::make_unique<DualSimplexSolver>();
                a.warm->setVerbose(false);
                a.warm->setOptions(dualOptions(opt));
                a.warm->load(lp);
                break;
            case ArmKind::DualCold:
                a.name = "dual-cold";
                break;
            case ArmKind::Barrier:
                a.name = "barrier";
                break;
            case ArmKind::BarrierHsd:
                a.name = "barrier-hsd";
                break;
            case ArmKind::Pdlp:
                a.name = "pdlp";
                break;
        }
        arms.push_back(std::move(a));
    }

    // Every output file must be a file of its own, checked before any is
    // opened and again once all are, as a dangling symlink can start
    // resolving to a file just created.
    namespace fs = std::filesystem;
    const std::array<const std::string*, 4> outputs = {&opt.csv, &opt.dual_stats_csv,
                                                       &opt.refactor_csv, &opt.edit_csv};
    const auto same_existing_file = [](const std::string& a, const std::string& b) {
        return !a.empty() && !b.empty() && fs::exists(a) && fs::exists(b) && fs::equivalent(a, b);
    };
    const auto same_file = [&](const std::string& a, const std::string& b) {
        return !a.empty() && !b.empty() &&
               (fs::weakly_canonical(fs::absolute(a)) == fs::weakly_canonical(fs::absolute(b)) ||
                same_existing_file(a, b));
    };
    const auto check_distinct = [&](const auto& same) {
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            for (std::size_t j = i + 1; j < outputs.size(); ++j) {
                if (same(*outputs[i], *outputs[j])) {
                    lpseq::fail(
                        "--csv, --dual-stats-csv, --refactor-csv and --edit-csv must name "
                        "different files");
                }
            }
        }
    };
    check_distinct(same_file);
    // Checked again after each open, so a clash stops the run before it
    // truncates a file outside the clash.
    const auto open = [&](std::ofstream& f, const std::string& path) {
        if (path.empty()) {
            return;
        }
        f.open(path);
        if (!f) {
            lpseq::fail(path + ": cannot write");
        }
        check_distinct(same_existing_file);
    };
    std::ofstream dual_stats;
    std::ofstream refactor_csv;
    std::ofstream edit_csv;
    std::ofstream csv_file;
    open(dual_stats, opt.dual_stats_csv);
    open(refactor_csv, opt.refactor_csv);
    open(edit_csv, opt.edit_csv);
    open(csv_file, opt.csv);

    std::ostream* out = csv_file.is_open() ? static_cast<std::ostream*>(&csv_file) : &std::cout;
    *out << lpseq::kCsvHeader;
    if (dual_stats.is_open()) {
        dual_stats << "seq,arm,added_columns,dual_pivots,primal_pivots,primal_degenerate_pivots,"
                      "primal_bound_perturbation_activations\n";
    }
    if (refactor_csv.is_open()) {
        refactor_csv << kRefactorCsvHeader;
    }

    EditLog log;
    if (edit_csv.is_open()) {
        edit_csv << "after,arm,op,items,basic,seconds\n";
        log.out = &edit_csv;
    }

    std::ifstream ops(opt.dir + "/ops.txt");
    if (!ops) {
        lpseq::fail(opt.dir + "/ops.txt: cannot open");
    }
    bool all_match = true;
    std::size_t added_columns = 0;
    lpseq::Op op;
    while (lpseq::readOp(ops, lp.num_rows, lp.num_cols, op)) {
        if (op.kind != lpseq::OpKind::Solve) {
            apply(op, lp, arms, log);
            if (op.kind == lpseq::OpKind::AddCols) {
                added_columns += op.starts.size();
            }
            continue;
        }
        for (Arm& arm : arms) {
            std::unique_ptr<LpSolver> cold;
            LpSolver* s = arm.warm.get();
            bool reloaded = false;
            if (arm.warm && arm.stale) {
                // The dimensions are the same, so the basis carries over
                // (none before the first solve).
                const auto basis = arm.warm->getBasis();
                arm.warm->load(lp);
                arm.warm->setBasis(basis);
                arm.stale = false;
                reloaded = true;
            }
            if (s == nullptr) {
                cold = coldSolver(arm.kind, opt);
                cold->load(lp);
                s = cold.get();
            }
            const auto t0 = std::chrono::steady_clock::now();
            const LpResult r = s->solve();
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // PDLP is a first-order method: it stops at its own, looser
            // relative gap, so it is held to --pdlp-rel-tol.
            const Real tol = arm.kind == ArmKind::Pdlp ? opt.pdlp_rel_tol : opt.rel_tol;
            Real err = 0.0;
            all_match =
                lpseq::matches(op, statusName(r.status), r.objective, tol, err) && all_match;
            lpseq::writeRow(*out, op, arm.name, statusName(r.status), r.objective, err,
                            r.iterations, secs, reloaded);
            if (dual_stats.is_open()) {
                if (const auto* dual = dynamic_cast<const DualSimplexSolver*>(s)) {
                    const auto& stats = dual->getDegeneracyStats();
                    dual_stats << op.seq << ',' << arm.name << ',' << added_columns << ','
                               << stats.dual_pivots << ',' << stats.primal_pivots << ','
                               << stats.primal_degenerate_pivots << ','
                               << stats.primal_bound_perturbation_activations << '\n';
                }
            }
            if (refactor_csv.is_open()) {
                if (const auto* dual = dynamic_cast<const DualSimplexSolver*>(s)) {
                    writeRefactorRow(refactor_csv, op.seq, arm.name, r.iterations, *dual);
                }
            }
        }
        added_columns = 0;
        log.after = op.seq;
    }
    if (edit_csv.is_open()) {
        edit_csv.close();
        if (!edit_csv) {
            lpseq::fail(opt.edit_csv + ": write failed");
        }
    }
    if (dual_stats.is_open()) {
        dual_stats.close();
        if (!dual_stats) {
            lpseq::fail(opt.dual_stats_csv + ": write failed");
        }
    }
    if (refactor_csv.is_open()) {
        refactor_csv.close();
        if (!refactor_csv) {
            lpseq::fail(opt.refactor_csv + ": write failed");
        }
    }
    return all_match ? 0 : 1;
}

void usage() {
    std::cerr
        << "usage: mipx-lpseq-replay DIR [--arm dual-warm|dual-cold|barrier|barrier-hsd|pdlp]... "
           "[--barrier-ne dense|sparse] [--primal-devex] [--primal-bound-perturbation] "
           "[--rel-tol T] [--pdlp-rel-tol T] [--csv FILE] [--dual-stats-csv FILE] "
           "[--refactor-csv FILE] [--edit-csv FILE]\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--arm" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "dual-warm") {
                opt.arms.push_back(ArmKind::DualWarm);
            } else if (v == "dual-cold") {
                opt.arms.push_back(ArmKind::DualCold);
            } else if (v == "barrier") {
                opt.arms.push_back(ArmKind::Barrier);
            } else if (v == "barrier-hsd") {
                opt.arms.push_back(ArmKind::BarrierHsd);
            } else if (v == "pdlp") {
                opt.arms.push_back(ArmKind::Pdlp);
            } else {
                usage();
                return 2;
            }
        } else if (a == "--barrier-ne" && i + 1 < argc) {
            // CPU normal-equations factor, as mipx-solve --barrier-ne X --no-gpu.
            const std::string v = argv[++i];
            if (v == "dense") {
                opt.barrier_algorithm = BarrierAlgorithm::CpuCholesky;
            } else if (v == "sparse") {
                opt.barrier_algorithm = BarrierAlgorithm::CpuSparseCholesky;
            } else {
                usage();
                return 2;
            }
        } else if (a == "--primal-devex") {
            opt.primal_devex = true;
        } else if (a == "--primal-bound-perturbation") {
            opt.primal_bound_perturbation = true;
        } else if (a == "--rel-tol" && i + 1 < argc) {
            opt.rel_tol = std::atof(argv[++i]);
        } else if (a == "--pdlp-rel-tol" && i + 1 < argc) {
            opt.pdlp_rel_tol = std::atof(argv[++i]);
        } else if (a == "--csv" && i + 1 < argc) {
            opt.csv = argv[++i];
        } else if (a == "--dual-stats-csv" && i + 1 < argc) {
            opt.dual_stats_csv = argv[++i];
        } else if (a == "--refactor-csv" && i + 1 < argc) {
            opt.refactor_csv = argv[++i];
        } else if (a == "--edit-csv" && i + 1 < argc) {
            opt.edit_csv = argv[++i];
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
        opt.arms.push_back(ArmKind::DualWarm);
    }
    try {
        return run(opt);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
}
