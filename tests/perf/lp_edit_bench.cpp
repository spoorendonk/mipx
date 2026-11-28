// mipx-lp-edit-bench: the cost of model edits on a kept BarrierSolver
// (presolve on, off) and PdlpSolver, timed against the solve (#442).
// Usage: mipx-lp-edit-bench LP.mps ROUNDS EDITS [barrier-p|barrier-np|pdlp]...
// Per round: EDITS setColObjective calls (cost scaled by 1 +- 1e-3 on a spread of
// columns) and EDITS setRowBounds calls (each row set to its own bounds), then
// solve(). Prints per arm: edit seconds, solve seconds, per-edit microseconds,
// and each solve's status, objective and iterations.
#include "mipx/barrier.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/pdlp.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace mipx;

namespace {

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void run(const char* name, LpSolver& s, const LpProblem& lp, int rounds, int edits) {
    s.load(lp);
    double t0 = now();
    LpResult r = s.solve();
    std::printf("%s round 0: solve %.4f s status %d obj %.12g iters %d\n", name, now() - t0,
                static_cast<int>(r.status), r.objective, r.iterations);
    std::vector<Real> obj = lp.obj;
    double edit_s = 0.0, solve_s = 0.0;
    long n_edits = 0;
    for (int k = 1; k <= rounds; ++k) {
        t0 = now();
        for (int e = 0; e < edits; ++e) {
            const Index j = static_cast<Index>((static_cast<long>(e) * 7919 + k) % lp.num_cols);
            obj[j] *= (k % 2 == 1) ? 1.001 : 1.0 / 1.001;
            s.setColObjective(j, obj[j]);
            const Index i = static_cast<Index>((static_cast<long>(e) * 104729 + k) % lp.num_rows);
            s.setRowBounds(i, lp.row_lower[i], lp.row_upper[i]);
        }
        const double te = now() - t0;
        t0 = now();
        r = s.solve();
        const double ts = now() - t0;
        edit_s += te;
        solve_s += ts;
        n_edits += 2L * edits;
        std::printf("%s round %d: edits %.4f s solve %.4f s status %d obj %.12g iters %d\n", name,
                    k, te, ts, static_cast<int>(r.status), r.objective, r.iterations);
    }
    std::printf("%s total: %ld edits %.4f s (%.1f us/edit), solves %.4f s, edit/solve %.2f\n", name,
                n_edits, edit_s, 1e6 * edit_s / static_cast<double>(n_edits), solve_s,
                edit_s / solve_s);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(
            stderr,
            "usage: mipx-lp-edit-bench LP.mps ROUNDS EDITS [barrier-p|barrier-np|pdlp]...\n");
        return 2;
    }
    const LpProblem lp = readMps(argv[1]);
    const int rounds = std::atoi(argv[2]);
    const int edits = std::atoi(argv[3]);
    std::printf("%s: %d rows, %d cols, %d nnz\n", argv[1], lp.num_rows, lp.num_cols,
                lp.matrix.numNonzeros());
    for (int a = 4; a < argc || a == 4; ++a) {
        const std::string arm = a < argc ? argv[a] : "all";
        if (arm == "barrier-p" || arm == "barrier-np" || arm == "all") {
            for (const bool pre : {true, false}) {
                if ((arm == "barrier-p" && !pre) || (arm == "barrier-np" && pre)) {
                    continue;
                }
                BarrierSolver b;
                BarrierOptions bo;
                bo.verbose = false;
                bo.presolve = pre;
                b.setOptions(bo);
                run(pre ? "barrier-p" : "barrier-np", b, lp, rounds, edits);
            }
        }
        if (arm == "pdlp" || arm == "all") {
            PdlpSolver p;
            PdlpOptions po;
            po.verbose = false;
            p.setOptions(po);
            run("pdlp", p, lp, rounds, edits);
        }
        if (a >= argc) {
            break;
        }
    }
    return 0;
}
