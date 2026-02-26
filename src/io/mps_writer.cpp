#include "mipx/io.h"

#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>

namespace mipx {

void writeMps(const std::string& filename, const LpProblem& input) {
    LpProblem linearized;
    const LpProblem* problem = &input;
    if (hasAdvancedModelFeatures(input)) {
        linearized = linearizeModelFeatures(input);
        problem = &linearized;
    }
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    out << "NAME          " << problem->name << '\n';

    // ROWS section.
    out << "ROWS\n";
    out << " N  obj\n";
    for (Index i = 0; i < problem->num_rows; ++i) {
        Real lo = problem->row_lower[i];
        Real up = problem->row_upper[i];
        char sense;
        if (lo == up) {
            sense = 'E';
        } else if (lo <= -kInf) {
            sense = 'L';
        } else if (up >= kInf) {
            sense = 'G';
        } else {
            // Ranged — write as L with RANGES.
            sense = 'L';
        }
        const std::string& name =
            i < static_cast<Index>(problem->row_names.size())
                ? problem->row_names[i]
                : std::format("R{}", i);
        out << " " << sense << "  " << name << '\n';
    }

    // COLUMNS section.
    out << "COLUMNS\n";
    bool in_int = false;
    for (Index j = 0; j < problem->num_cols; ++j) {
        bool is_int = (j < static_cast<Index>(problem->col_type.size())) &&
                      (problem->col_type[j] == VarType::Integer ||
                       problem->col_type[j] == VarType::Binary);

        if (is_int && !in_int) {
            out << "    INTMARK   'MARKER'                 'INTORG'\n";
            in_int = true;
        } else if (!is_int && in_int) {
            out << "    INTMARK   'MARKER'                 'INTEND'\n";
            in_int = false;
        }

        const std::string& col_name =
            j < static_cast<Index>(problem->col_names.size())
                ? problem->col_names[j]
                : std::format("C{}", j);

        // Objective coefficient.
        if (j < static_cast<Index>(problem->obj.size()) &&
            problem->obj[j] != 0.0) {
            out << "    " << col_name << "  obj  " << problem->obj[j] << '\n';
        }

        // Matrix coefficients for this column.
        // Need column access — iterate rows.
        for (Index i = 0; i < problem->num_rows; ++i) {
            Real val = problem->matrix.coeff(i, j);
            if (val != 0.0) {
                const std::string& row_name =
                    i < static_cast<Index>(problem->row_names.size())
                        ? problem->row_names[i]
                        : std::format("R{}", i);
                out << "    " << col_name << "  " << row_name << "  " << val
                    << '\n';
            }
        }
    }
    if (in_int) {
        out << "    INTMARK   'MARKER'                 'INTEND'\n";
    }

    // RHS section.
    out << "RHS\n";
    for (Index i = 0; i < problem->num_rows; ++i) {
        Real lo = problem->row_lower[i];
        Real up = problem->row_upper[i];
        Real rhs;
        if (lo == up) {
            rhs = lo;
        } else if (lo <= -kInf) {
            rhs = up;
        } else if (up >= kInf) {
            rhs = lo;
        } else {
            // Ranged L row — RHS is upper bound.
            rhs = up;
        }
        if (rhs != 0.0) {
            const std::string& name =
                i < static_cast<Index>(problem->row_names.size())
                    ? problem->row_names[i]
                    : std::format("R{}", i);
            out << "    rhs  " << name << "  " << rhs << '\n';
        }
    }

    // RANGES section.
    bool has_ranges = false;
    for (Index i = 0; i < problem->num_rows; ++i) {
        Real lo = problem->row_lower[i];
        Real up = problem->row_upper[i];
        if (lo > -kInf && up < kInf && lo != up) {
            if (!has_ranges) {
                out << "RANGES\n";
                has_ranges = true;
            }
            const std::string& name =
                i < static_cast<Index>(problem->row_names.size())
                    ? problem->row_names[i]
                    : std::format("R{}", i);
            out << "    rng  " << name << "  " << (up - lo) << '\n';
        }
    }

    // BOUNDS section.
    bool has_bounds = false;
    for (Index j = 0; j < problem->num_cols; ++j) {
        Real lo = j < static_cast<Index>(problem->col_lower.size())
                      ? problem->col_lower[j]
                      : 0.0;
        Real up = j < static_cast<Index>(problem->col_upper.size())
                      ? problem->col_upper[j]
                      : kInf;
        VarType type = j < static_cast<Index>(problem->col_type.size())
                           ? problem->col_type[j]
                           : VarType::Continuous;

        const std::string& col_name =
            j < static_cast<Index>(problem->col_names.size())
                ? problem->col_names[j]
                : std::format("C{}", j);

        bool default_bounds = (lo == 0.0 && up == kInf);
        if (type == VarType::Binary) {
            if (!has_bounds) {
                out << "BOUNDS\n";
                has_bounds = true;
            }
            out << " BV bnd  " << col_name << '\n';
        } else if (lo == up) {
            if (!has_bounds) {
                out << "BOUNDS\n";
                has_bounds = true;
            }
            out << " FX bnd  " << col_name << "  " << lo << '\n';
        } else if (!default_bounds) {
            if (!has_bounds) {
                out << "BOUNDS\n";
                has_bounds = true;
            }
            if (lo == -kInf && up == kInf) {
                out << " FR bnd  " << col_name << '\n';
            } else {
                if (lo != 0.0) {
                    out << " LO bnd  " << col_name << "  " << lo << '\n';
                }
                if (up != kInf) {
                    out << " UP bnd  " << col_name << "  " << up << '\n';
                }
            }
        }
    }

    out << "ENDATA\n";
}

}  // namespace mipx
