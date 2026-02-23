#pragma once

#include <string>
#include <vector>

#include "mipx/lp_problem.h"

namespace mipx {

/// Read an MPS file (fixed or free format). Detects .gz for gzip.
LpProblem readMps(const std::string& filename);

/// Write an MPS file (free format).
void writeMps(const std::string& filename, const LpProblem& problem);

/// Read a CPLEX-style LP file.
LpProblem readLp(const std::string& filename);

/// Entry from a .solu file.
struct SoluEntry {
    std::string name;
    Real value;
    bool is_infeasible = false;
};

/// Read a .solu file with known optimal values.
std::vector<SoluEntry> readSolu(const std::string& filename);

}  // namespace mipx
