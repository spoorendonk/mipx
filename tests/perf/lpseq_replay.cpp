// mipx-lpseq-replay: replay an LP sequence written in the "mcf-lpseq 1"
// format (tinnet-ai/mcf#205, mcf's recording LP backend) on mipx's LP
// solvers, and check every solve's objective against the recorded one (#267).
//
// A sequence directory holds manifest.txt (key = value lines, first
// `format = mcf-lpseq 1`), master_0000.mps (the LP at its first solve, free
// MPS, rows r<i> and columns c<j> by index) and ops.txt: one op per line,
// indices as of the LP it applies to, applied in order:
//
//   add_cols <k>  then k lines  <cost> <lb> <ub> <nnz> <row> <val> ...
//   del_cols <k> <j1> ...       (indices before the delete)
//   add_rows <k>  then k lines  <lb> <ub> <nnz> <col> <val> ...
//   del_rows <k> <i1> ...
//   col_bounds <k> <j> <lb> <ub> ...
//   row_bounds <k> <i> <lb> <ub> ...
//   col_costs <k> <j> <cost> ...
//   coeffs <k> <i> <j> <val> ...  (0 removes the entry)
//   solve <seq> <status> <obj> <iters> <secs>
//
// Arms: dual-warm keeps one DualSimplexSolver and applies the ops to it (its
// basis carries over); dual-cold, barrier and pdlp load the LP afresh for
// each solve. A coeffs op, which LpSolver has no call for, makes dual-warm
// reload the LP at that point (counted as a reload).
//
// Output: one CSV line per (solve, arm) on stdout, or to --csv FILE:
//   seq,arm,status,objective,recorded,rel_err,iterations,seconds,reloaded
// Exit 0 when every solve of every arm matches the recorded status and
// objective (relative to max(1, |recorded|), within --rel-tol, 1e-6, or for
// pdlp --pdlp-rel-tol, 1e-4), 1 when one does not, 2 on a usage or format
// error.

#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/pdlp.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using namespace mipx;

namespace {

[[noreturn]] void formatError(const std::string& what) {
    throw std::runtime_error("lpseq: " + what);
}

// from_chars reads %.17g back exactly, subnormals and inf/-inf included
// (stod would throw out_of_range on a subnormal). A literal past double's
// range (1e400) leaves the value unset, so it is an error, not 0.
Real parseReal(const std::string& s) {
    const char* first = s.data();
    if (!s.empty() && s[0] == '+') {
        ++first;  // from_chars takes no leading '+'
    }
    Real v = 0.0;
    const auto [end, ec] = std::from_chars(first, s.data() + s.size(), v);
    if (end != s.data() + s.size() || ec != std::errc() || std::isnan(v)) {
        formatError("not a number: " + s);
    }
    return v;
}

std::string trim(std::string_view s) {
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string_view::npos) {
        return {};
    }
    const auto e = s.find_last_not_of(" \t\r");
    return std::string(s.substr(b, e - b + 1));
}

Index parseIndex(const std::string& s) {
    Index v = 0;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc() || end != s.data() + s.size() || v < 0) {
        formatError("not an index: " + s);
    }
    return v;
}

// An index the op addresses, checked against the LP it applies to.
Index parseIndexBelow(const std::string& s, Index bound, const char* what) {
    const Index v = parseIndex(s);
    if (v >= bound) {
        formatError(std::string(what) + " index " + s + " out of range (" + std::to_string(bound) +
                    ")");
    }
    return v;
}

// The master as mipx reads it, with rows and columns in the recorder's
// order. mipx's reader drops every N row after the objective; the recorder
// writes a row with both sides infinite as one, which would shift the rows
// after it, so such a master is refused rather than replayed wrong.
LpProblem readMaster(const std::string& path) {
    LpProblem lp = readMps(path);
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (i >= static_cast<Index>(lp.row_names.size()) ||
            lp.row_names[i] != "r" + std::to_string(i)) {
            formatError(path + ": row " + std::to_string(i) + " is not r" + std::to_string(i) +
                        " (a free row, written as N, is not supported)");
        }
    }
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (j >= static_cast<Index>(lp.col_names.size()) ||
            lp.col_names[j] != "c" + std::to_string(j)) {
            formatError(path + ": column " + std::to_string(j) + " is not c" + std::to_string(j));
        }
    }
    if (lp.sense != Sense::Minimize) {
        formatError(path + ": the format minimises");
    }
    return lp;
}

void checkManifest(const std::string& dir) {
    std::ifstream in(dir + "/manifest.txt");
    if (!in) {
        formatError(dir + "/manifest.txt: cannot open");
    }
    // key = value lines; the format must be exactly "mcf-lpseq 1", and the
    // replay compares objectives of a minimised LP without a constant.
    std::string line;
    bool format = false;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string_view whole(line);
        const std::string key = trim(whole.substr(0, eq));
        const std::string value = trim(whole.substr(eq + 1));
        if (key == "format") {
            if (value != "mcf-lpseq 1") {
                formatError("unsupported format: " + value);
            }
            format = true;
        } else if (key == "sense" && value != "min") {
            formatError("unsupported sense: " + value);
        } else if (key == "obj_offset" && parseReal(value) != 0.0) {
            formatError("unsupported obj_offset: " + value);
        }
    }
    if (!format) {
        formatError(dir + "/manifest.txt: no format line");
    }
}

// Row-wise and column-wise batches, as the LpSolver calls take them.
struct Batch {
    std::vector<Index> starts, indices;
    std::vector<Real> values, a, b, c;  // cols: cost, lb, ub; rows: lb, ub
};

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

void mirrorAddRows(LpProblem& lp, const Batch& rows) {
    auto t = triplets(lp);
    const auto k = static_cast<Index>(rows.a.size());
    for (Index r = 0; r < k; ++r) {
        const Index end =
            (r + 1 < k) ? rows.starts[r + 1] : static_cast<Index>(rows.indices.size());
        for (Index p = rows.starts[r]; p < end; ++p) {
            t.push_back({lp.num_rows + r, rows.indices[p], rows.values[p]});
        }
        lp.row_lower.push_back(rows.a[r]);
        lp.row_upper.push_back(rows.b[r]);
        lp.row_names.push_back("r" + std::to_string(lp.num_rows + r));
    }
    lp.num_rows += k;
    rebuild(lp, std::move(t));
}

void mirrorDelRows(LpProblem& lp, std::vector<Index> rows) {
    std::ranges::sort(rows);
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    std::vector<Index> map(static_cast<std::size_t>(lp.num_rows), 0);
    Index next = 0;
    std::size_t r = 0;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (r < rows.size() && rows[r] == i) {
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

// One pass: drop the entries at the changed positions, then add the changes
// that are not 0 (each position once per op, as the format says).
void mirrorSetCoeffs(LpProblem& lp, const std::vector<Triplet>& changes) {
    const auto key = [&](Index i, Index j) {
        return static_cast<std::int64_t>(i) * static_cast<std::int64_t>(lp.num_cols) + j;
    };
    std::unordered_set<std::int64_t> changed;
    for (const auto& c : changes) {
        if (!changed.insert(key(c.row, c.col)).second || !std::isfinite(c.val)) {
            formatError("coeffs: a position twice in one op, or a value not finite");
        }
    }
    auto t = triplets(lp);
    std::erase_if(t, [&](const Triplet& e) { return changed.contains(key(e.row, e.col)); });
    for (const auto& c : changes) {
        if (c.val != 0.0) {
            t.push_back(c);
        }
    }
    rebuild(lp, std::move(t));
}

enum class ArmKind { DualWarm, DualCold, Barrier, Pdlp };

struct Arm {
    ArmKind kind;
    std::string name;
    std::unique_ptr<DualSimplexSolver> warm;  // DualWarm only
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

int run(const Options& opt) {
    checkManifest(opt.dir);
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
            formatError(opt.csv + ": cannot write");
        }
        out = &csv_file;
    }
    *out << "seq,arm,status,objective,recorded,rel_err,iterations,seconds,reloaded\n";

    std::ifstream ops(opt.dir + "/ops.txt");
    if (!ops) {
        formatError(opt.dir + "/ops.txt: cannot open");
    }
    bool all_match = true;
    std::vector<char> warm_stale(arms.size(), 0);  // a coeffs op since the last solve
    std::string line;
    auto readBatch = [&](Index k, bool cols) {
        Batch b;
        for (Index t = 0; t < k; ++t) {
            if (!std::getline(ops, line)) {
                formatError("ops.txt: batch cut short");
            }
            std::istringstream ls(line);
            std::string x, y, z, n;
            ls >> x >> y;
            if (cols) {
                ls >> z;
            }
            ls >> n;
            if (cols) {
                b.a.push_back(parseReal(x));
                b.b.push_back(parseReal(y));
                b.c.push_back(parseReal(z));
            } else {
                b.a.push_back(parseReal(x));
                b.b.push_back(parseReal(y));
            }
            const Index nnz = parseIndex(n);
            b.starts.push_back(static_cast<Index>(b.indices.size()));
            for (Index p = 0; p < nnz; ++p) {
                std::string idx, val;
                if (!(ls >> idx >> val)) {
                    formatError("ops.txt: entry list cut short: " + line);
                }
                b.indices.push_back(parseIndex(idx));
                b.values.push_back(parseReal(val));
            }
        }
        return b;
    };

    while (std::getline(ops, line)) {
        std::istringstream ls(line);
        std::string op;
        if (!(ls >> op)) {
            continue;
        }
        if (op == "solve") {
            std::string seq, status, obj;
            ls >> seq >> status >> obj;
            const Real recorded = parseReal(obj);
            for (std::size_t a = 0; a < arms.size(); ++a) {
                Arm& arm = arms[a];
                std::unique_ptr<LpSolver> cold;
                LpSolver* s = arm.warm.get();
                bool reloaded = false;
                if (arm.kind == ArmKind::DualWarm && warm_stale[a] != 0) {
                    // The dimensions are the same, so the basis carries over.
                    const auto basis = arm.warm->getBasis();
                    arm.warm->load(lp);
                    arm.warm->setBasis(basis);
                    warm_stale[a] = 0;
                    reloaded = true;
                }
                if (s == nullptr) {
                    if (arm.kind == ArmKind::DualCold) {
                        auto d = std::make_unique<DualSimplexSolver>();
                        d->setVerbose(false);
                        d->setOptions(dualOptions(opt));
                        cold = std::move(d);
                    } else if (arm.kind == ArmKind::Barrier) {
                        auto b = std::make_unique<BarrierSolver>();
                        BarrierOptions bo;
                        bo.verbose = false;
                        b->setOptions(bo);
                        cold = std::move(b);
                    } else {
                        auto p = std::make_unique<PdlpSolver>();
                        PdlpOptions po;
                        po.verbose = false;
                        p->setOptions(po);
                        cold = std::move(p);
                    }
                    cold->load(lp);
                    s = cold.get();
                }
                const auto t0 = std::chrono::steady_clock::now();
                const LpResult r = s->solve();
                const double secs =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const Real err =
                    std::abs(r.objective - recorded) / std::max<Real>(1.0, std::abs(recorded));
                // PDLP is a first-order method: it stops at its own, looser
                // relative gap, so it is held to --pdlp-rel-tol.
                const Real tol = arm.kind == ArmKind::Pdlp ? opt.pdlp_rel_tol : opt.rel_tol;
                const bool match =
                    status == statusName(r.status) && (status != "optimal" || err <= tol);
                all_match = all_match && match;
                char buf[256];
                std::snprintf(buf, sizeof(buf), "%s,%s,%s,%.17g,%.17g,%.3e,%d,%.6f,%d\n",
                              seq.c_str(), arm.name.c_str(), statusName(r.status), r.objective,
                              recorded, err, r.iterations, secs, reloaded ? 1 : 0);
                *out << buf;
            }
            continue;
        }
        std::string count;
        ls >> count;
        const Index k = parseIndex(count);
        if (op == "add_cols") {
            const Batch b = readBatch(k, true);
            checkColumnBatch(lp.num_rows, b.starts, b.indices, b.values, b.a, b.b, b.c);
            appendColumns(lp, b.starts, b.indices, b.values, b.a, b.b, b.c);
            for (auto& arm : arms) {
                if (arm.warm) {
                    arm.warm->addCols(b.starts, b.indices, b.values, b.a, b.b, b.c);
                }
            }
        } else if (op == "add_rows") {
            const Batch b = readBatch(k, false);
            for (const Index j : b.indices) {
                if (j >= lp.num_cols) {
                    formatError("add_rows: column index " + std::to_string(j) + " out of range");
                }
            }
            for (const Real v : b.values) {
                if (!std::isfinite(v)) {
                    formatError("add_rows: coefficient not finite");
                }
            }
            for (std::size_t r = 0; r < b.a.size(); ++r) {
                if (b.a[r] == kInf || b.b[r] == -kInf) {
                    formatError("add_rows: bound on the wrong side");
                }
            }
            mirrorAddRows(lp, b);
            for (auto& arm : arms) {
                if (arm.warm) {
                    arm.warm->addRows(b.starts, b.indices, b.values, b.a, b.b);
                }
            }
        } else if (op == "del_cols" || op == "del_rows") {
            std::vector<Index> idx;
            for (Index t = 0; t < k; ++t) {
                std::string x;
                ls >> x;
                idx.push_back(op == "del_cols" ? parseIndexBelow(x, lp.num_cols, "del_cols")
                                               : parseIndexBelow(x, lp.num_rows, "del_rows"));
            }
            if (op == "del_cols") {
                removeColumns(lp, sortedColumnSet(lp.num_cols, idx));
            } else {
                mirrorDelRows(lp, idx);
            }
            for (auto& arm : arms) {
                if (arm.warm) {
                    if (op == "del_cols") {
                        arm.warm->removeCols(idx);
                    } else {
                        arm.warm->removeRows(idx);
                    }
                }
            }
        } else if (op == "col_bounds" || op == "row_bounds") {
            for (Index t = 0; t < k; ++t) {
                std::string x, lo, up;
                ls >> x >> lo >> up;
                const Index i = op == "col_bounds" ? parseIndexBelow(x, lp.num_cols, "col_bounds")
                                                   : parseIndexBelow(x, lp.num_rows, "row_bounds");
                const Real l = parseReal(lo);
                const Real u = parseReal(up);
                if (op == "col_bounds") {
                    lp.col_lower[i] = l;
                    lp.col_upper[i] = u;
                } else {
                    lp.row_lower[i] = l;
                    lp.row_upper[i] = u;
                }
                for (auto& arm : arms) {
                    if (arm.warm) {
                        if (op == "col_bounds") {
                            arm.warm->setColBounds(i, l, u);
                        } else {
                            arm.warm->setRowBounds(i, l, u);
                        }
                    }
                }
            }
        } else if (op == "col_costs") {
            for (Index t = 0; t < k; ++t) {
                std::string x, c;
                ls >> x >> c;
                const Index j = parseIndexBelow(x, lp.num_cols, "col_costs");
                lp.obj[j] = parseReal(c);
                for (auto& arm : arms) {
                    if (arm.warm) {
                        arm.warm->setColObjective(j, lp.obj[j]);
                    }
                }
            }
        } else if (op == "coeffs") {
            std::vector<Triplet> changes;
            for (Index t = 0; t < k; ++t) {
                std::string i, j, v;
                ls >> i >> j >> v;
                changes.push_back({parseIndexBelow(i, lp.num_rows, "coeffs row"),
                                   parseIndexBelow(j, lp.num_cols, "coeffs column"), parseReal(v)});
            }
            mirrorSetCoeffs(lp, changes);
            for (std::size_t a = 0; a < arms.size(); ++a) {
                warm_stale[a] = arms[a].warm ? 1 : 0;
            }
        } else {
            formatError("ops.txt: unknown op " + op);
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
