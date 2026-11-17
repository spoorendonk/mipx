// mipx-lpseq-replay: replay an LP sequence in mcf's "mcf-lpseq 1" format
// (tinnet-ai/mcf#205, mcf's recording LP backend; see lpseq_format.h and
// tests/perf/README.md) on mipx's LP solvers, and check every solve's
// objective against the recorded one (#267).
//
// Arms: dual-warm keeps one DualSimplexSolver and applies the ops to it (its
// basis carries over); dual-cold, barrier and pdlp load the LP afresh for
// each solve. A coeffs op, which LpSolver has no call for, makes dual-warm
// reload the LP at that point, keeping its basis (counted as a reload).
//
// Output: one CSV line per (solve, arm) on stdout, or to --csv FILE:
//   seq,arm,status,objective,recorded,rel_err,iterations,seconds,reloaded
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

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

enum class ArmKind { DualWarm, DualCold, Barrier, Pdlp };

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
    Real rel_tol = 1e-6;
    Real pdlp_rel_tol = 1e-4;
    std::string csv;
};

DualSimplexOptions dualOptions(const Options& o) {
    DualSimplexOptions d;
    d.enable_primal_devex = o.primal_devex;
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
    if (kind == ArmKind::Barrier) {
        auto b = std::make_unique<BarrierSolver>();
        BarrierOptions bo;
        bo.verbose = false;
        b->setOptions(bo);
        return b;
    }
    auto p = std::make_unique<PdlpSolver>();
    PdlpOptions po;
    po.verbose = false;
    p->setOptions(po);
    return p;
}

// Apply a change to the mirror and to every warm solver.
void apply(const lpseq::Op& op, LpProblem& lp, std::vector<Arm>& arms) {
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
            case ArmKind::Pdlp:
                a.name = "pdlp";
                break;
        }
        arms.push_back(std::move(a));
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
    *out << "seq,arm,status,objective,recorded,rel_err,iterations,seconds,reloaded\n";

    std::ifstream ops(opt.dir + "/ops.txt");
    if (!ops) {
        lpseq::fail(opt.dir + "/ops.txt: cannot open");
    }
    bool all_match = true;
    lpseq::Op op;
    while (lpseq::readOp(ops, lp.num_rows, lp.num_cols, op)) {
        if (op.kind != lpseq::OpKind::Solve) {
            apply(op, lp, arms);
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
            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s,%s,%s,%.17g,%.17g,%.3e,%d,%.6f,%d\n",
                          op.seq.c_str(), arm.name.c_str(), statusName(r.status), r.objective,
                          op.recorded, err, r.iterations, secs, reloaded ? 1 : 0);
            *out << buf;
        }
    }
    return all_match ? 0 : 1;
}

void usage() {
    std::cerr << "usage: mipx-lpseq-replay DIR [--arm dual-warm|dual-cold|barrier|pdlp]... "
                 "[--primal-devex] [--rel-tol T] [--pdlp-rel-tol T] [--csv FILE]\n";
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
            } else if (v == "pdlp") {
                opt.arms.push_back(ArmKind::Pdlp);
            } else {
                usage();
                return 2;
            }
        } else if (a == "--primal-devex") {
            opt.primal_devex = true;
        } else if (a == "--rel-tol" && i + 1 < argc) {
            opt.rel_tol = std::atof(argv[++i]);
        } else if (a == "--pdlp-rel-tol" && i + 1 < argc) {
            opt.pdlp_rel_tol = std::atof(argv[++i]);
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
        opt.arms.push_back(ArmKind::DualWarm);
    }
    try {
        return run(opt);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
}
