#include "mipx/io.h"

#include <cassert>
#include <charconv>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <zlib.h>

namespace mipx {

namespace {

/// RAII wrapper for gzFile.
class GzFileReader {
public:
    explicit GzFileReader(const std::string& filename) {
        file_ = gzopen(filename.c_str(), "rb");
        if (!file_) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
    }

    ~GzFileReader() {
        if (file_) gzclose(file_);
    }

    GzFileReader(const GzFileReader&) = delete;
    GzFileReader& operator=(const GzFileReader&) = delete;

    bool getline(std::string& line) {
        char buf[4096];
        if (!gzgets(file_, buf, sizeof(buf))) return false;
        line = buf;
        // Remove trailing newline.
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        return true;
    }

private:
    gzFile file_ = nullptr;
};

/// Line reader that works with both gzip and plain files.
class LineReader {
public:
    explicit LineReader(const std::string& filename) {
        if (filename.size() >= 3 &&
            filename.substr(filename.size() - 3) == ".gz") {
            gz_ = std::make_unique<GzFileReader>(filename);
        } else {
            plain_ = std::make_unique<std::ifstream>(filename);
            if (!plain_->is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }
        }
    }

    bool getline(std::string& line) {
        if (gz_) {
            return gz_->getline(line);
        }
        if (std::getline(*plain_, line)) {
            while (!line.empty() &&
                   (line.back() == '\r' || line.back() == '\n')) {
                line.pop_back();
            }
            return true;
        }
        return false;
    }

private:
    std::unique_ptr<GzFileReader> gz_;
    std::unique_ptr<std::ifstream> plain_;
};

/// Split a line into whitespace-delimited tokens.
std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

Real parseReal(const std::string& s) {
    Real val = 0.0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec != std::errc{}) {
        throw std::runtime_error("Invalid number: " + s);
    }
    return val;
}

bool isSection(const std::string& line) {
    if (line.empty()) return false;
    // Section headers start at column 0 (no leading space).
    return !std::isspace(static_cast<unsigned char>(line[0]));
}

enum class Section { None, Name, Rows, Columns, Rhs, Ranges, Bounds, Endata };

Section parseSection(const std::string& line) {
    auto tokens = tokenize(line);
    if (tokens.empty()) return Section::None;
    const auto& s = tokens[0];
    if (s == "NAME") return Section::Name;
    if (s == "ROWS") return Section::Rows;
    if (s == "COLUMNS") return Section::Columns;
    if (s == "RHS") return Section::Rhs;
    if (s == "RANGES") return Section::Ranges;
    if (s == "BOUNDS") return Section::Bounds;
    if (s == "ENDATA") return Section::Endata;
    return Section::None;
}

}  // namespace

LpProblem readMps(const std::string& filename) {
    LineReader reader(filename);

    LpProblem prob;

    // Maps for name → index.
    std::unordered_map<std::string, Index> row_map;
    std::unordered_map<std::string, Index> col_map;

    // Row data during parsing.
    std::vector<char> row_sense;  // 'N', 'L', 'G', 'E'
    std::string obj_row_name;

    // Triplets for constraint matrix.
    std::vector<Triplet> triplets;

    // RHS and range values (indexed by row).
    std::vector<Real> rhs_values;
    std::vector<Real> range_values;

    bool in_integer_section = false;
    Section section = Section::None;

    auto getOrCreateCol = [&](const std::string& name) -> Index {
        auto it = col_map.find(name);
        if (it != col_map.end()) return it->second;
        Index idx = prob.num_cols++;
        col_map[name] = idx;
        prob.col_names.push_back(name);
        prob.obj.push_back(0.0);
        prob.col_lower.push_back(0.0);
        prob.col_upper.push_back(kInf);
        prob.col_type.push_back(VarType::Continuous);
        return idx;
    };

    std::string line;
    while (reader.getline(line)) {
        // Skip empty lines and comments.
        if (line.empty()) continue;
        if (line[0] == '*' || line[0] == '$') continue;

        if (isSection(line)) {
            section = parseSection(line);
            if (section == Section::Name) {
                auto tokens = tokenize(line);
                if (tokens.size() >= 2) {
                    prob.name = tokens[1];
                }
            }
            if (section == Section::Endata) break;
            continue;
        }

        auto tokens = tokenize(line);
        if (tokens.empty()) continue;

        switch (section) {
            case Section::Rows: {
                if (tokens.size() < 2) break;
                char sense = tokens[0][0];
                const std::string& name = tokens[1];
                if (sense == 'N') {
                    obj_row_name = name;
                    // Still add to row_map for coefficient parsing, but mark
                    // as objective.
                    row_map[name] = -1;
                } else {
                    Index idx = prob.num_rows++;
                    row_map[name] = idx;
                    prob.row_names.push_back(name);
                    row_sense.push_back(sense);
                }
                break;
            }

            case Section::Columns: {
                // Check for integer markers.
                if (tokens.size() >= 3 && tokens[1] == "'MARKER'") {
                    if (tokens[2] == "'INTORG'") {
                        in_integer_section = true;
                    } else if (tokens[2] == "'INTEND'") {
                        in_integer_section = false;
                    }
                    break;
                }

                if (tokens.size() < 3) break;
                const std::string& col_name = tokens[0];
                Index col_idx = getOrCreateCol(col_name);

                if (in_integer_section) {
                    prob.col_type[col_idx] = VarType::Integer;
                }

                // Process pairs: (row_name, value).
                for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
                    const std::string& row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);

                    auto it = row_map.find(row_name);
                    if (it == row_map.end()) {
                        throw std::runtime_error(
                            "MPS: unknown row '" + row_name + "'");
                    }
                    if (it->second == -1) {
                        // Objective row.
                        prob.obj[col_idx] = val;
                    } else {
                        triplets.push_back({it->second, col_idx, val});
                    }
                }
                break;
            }

            case Section::Rhs: {
                // First token is RHS name (ignored), then pairs.
                if (tokens.size() < 3) break;
                for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
                    const std::string& row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);
                    auto it = row_map.find(row_name);
                    if (it == row_map.end()) continue;
                    if (it->second == -1) {
                        // Objective offset (RHS of N row).
                        prob.obj_offset = -val;
                        continue;
                    }
                    Index idx = it->second;
                    if (idx >= static_cast<Index>(rhs_values.size())) {
                        rhs_values.resize(idx + 1, 0.0);
                    }
                    rhs_values[idx] = val;
                }
                break;
            }

            case Section::Ranges: {
                if (tokens.size() < 3) break;
                for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
                    const std::string& row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);
                    auto it = row_map.find(row_name);
                    if (it == row_map.end() || it->second == -1) continue;
                    Index idx = it->second;
                    if (idx >= static_cast<Index>(range_values.size())) {
                        range_values.resize(idx + 1, 0.0);
                    }
                    range_values[idx] = val;
                }
                break;
            }

            case Section::Bounds: {
                if (tokens.size() < 3) break;
                const std::string& bound_type = tokens[0];
                // tokens[1] is bound name (ignored).
                const std::string& col_name = tokens[2];
                Index col_idx = getOrCreateCol(col_name);

                if (bound_type == "LO") {
                    prob.col_lower[col_idx] = parseReal(tokens[3]);
                } else if (bound_type == "UP") {
                    prob.col_upper[col_idx] = parseReal(tokens[3]);
                } else if (bound_type == "FX") {
                    Real v = parseReal(tokens[3]);
                    prob.col_lower[col_idx] = v;
                    prob.col_upper[col_idx] = v;
                } else if (bound_type == "FR") {
                    prob.col_lower[col_idx] = -kInf;
                    prob.col_upper[col_idx] = kInf;
                } else if (bound_type == "MI") {
                    prob.col_lower[col_idx] = -kInf;
                } else if (bound_type == "PL") {
                    prob.col_upper[col_idx] = kInf;
                } else if (bound_type == "BV") {
                    prob.col_lower[col_idx] = 0.0;
                    prob.col_upper[col_idx] = 1.0;
                    prob.col_type[col_idx] = VarType::Binary;
                } else if (bound_type == "LI") {
                    prob.col_lower[col_idx] = parseReal(tokens[3]);
                    prob.col_type[col_idx] = VarType::Integer;
                } else if (bound_type == "UI") {
                    prob.col_upper[col_idx] = parseReal(tokens[3]);
                    prob.col_type[col_idx] = VarType::Integer;
                }
                break;
            }

            default:
                break;
        }
    }

    // Build constraint matrix.
    prob.matrix = SparseMatrix(prob.num_rows, prob.num_cols, std::move(triplets));

    // Convert row_sense + rhs + ranges to row_lower/row_upper.
    rhs_values.resize(prob.num_rows, 0.0);
    range_values.resize(prob.num_rows, 0.0);
    prob.row_lower.resize(prob.num_rows);
    prob.row_upper.resize(prob.num_rows);

    for (Index i = 0; i < prob.num_rows; ++i) {
        Real rhs = rhs_values[i];
        Real range = range_values[i];
        char sense = row_sense[i];

        switch (sense) {
            case 'L':
                prob.row_upper[i] = rhs;
                if (range != 0.0) {
                    prob.row_lower[i] = rhs - std::abs(range);
                } else {
                    prob.row_lower[i] = -kInf;
                }
                break;
            case 'G':
                prob.row_lower[i] = rhs;
                if (range != 0.0) {
                    prob.row_upper[i] = rhs + std::abs(range);
                } else {
                    prob.row_upper[i] = kInf;
                }
                break;
            case 'E':
                if (range != 0.0) {
                    if (range > 0.0) {
                        prob.row_lower[i] = rhs;
                        prob.row_upper[i] = rhs + range;
                    } else {
                        prob.row_lower[i] = rhs + range;
                        prob.row_upper[i] = rhs;
                    }
                } else {
                    prob.row_lower[i] = rhs;
                    prob.row_upper[i] = rhs;
                }
                break;
            default:
                prob.row_lower[i] = -kInf;
                prob.row_upper[i] = kInf;
                break;
        }
    }

    return prob;
}

}  // namespace mipx
