#include "mipx/io.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include <zlib.h>

#ifdef MIPX_HAS_BZIP2
#include <bzlib.h>
#endif

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

#ifdef MIPX_HAS_BZIP2
/// RAII wrapper for bzip2 files.
class BzFileReader {
public:
    explicit BzFileReader(const std::string& filename) {
        fp_ = fopen(filename.c_str(), "rb");
        if (!fp_) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        int bzerror = BZ_OK;
        bz_ = BZ2_bzReadOpen(&bzerror, fp_, 0, 0, nullptr, 0);
        if (bzerror != BZ_OK) {
            fclose(fp_);
            throw std::runtime_error("BZ2_bzReadOpen failed: " + filename);
        }
    }

    ~BzFileReader() {
        if (bz_) {
            int bzerror;
            BZ2_bzReadClose(&bzerror, bz_);
        }
        if (fp_) fclose(fp_);
    }

    BzFileReader(const BzFileReader&) = delete;
    BzFileReader& operator=(const BzFileReader&) = delete;

    bool getline(std::string& line) {
        line.clear();
        while (true) {
            // Return buffered characters up to next newline.
            while (buf_pos_ < buf_len_) {
                char c = buf_[buf_pos_++];
                if (c == '\n') {
                    // Strip trailing \r.
                    while (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    return true;
                }
                line.push_back(c);
            }
            // Refill buffer.
            if (eof_) {
                return !line.empty();
            }
            int bzerror = BZ_OK;
            buf_len_ = BZ2_bzRead(&bzerror, bz_, buf_, sizeof(buf_));
            buf_pos_ = 0;
            if (bzerror == BZ_STREAM_END) {
                eof_ = true;
            } else if (bzerror != BZ_OK) {
                return !line.empty();
            }
            if (buf_len_ == 0) {
                eof_ = true;
                return !line.empty();
            }
        }
    }

private:
    FILE* fp_ = nullptr;
    BZFILE* bz_ = nullptr;
    char buf_[4096]{};
    int buf_pos_ = 0;
    int buf_len_ = 0;
    bool eof_ = false;
};
#endif

/// Line reader that works with gzip, bzip2, and plain files.
class LineReader {
public:
    explicit LineReader(const std::string& filename) {
        if (filename.size() >= 4 &&
            filename.substr(filename.size() - 4) == ".bz2") {
#ifdef MIPX_HAS_BZIP2
            bz_ = std::make_unique<BzFileReader>(filename);
#else
            throw std::runtime_error(
                "bzip2 support not compiled in (need -DMIPX_USE_BZIP2=ON): " +
                filename);
#endif
        } else if (filename.size() >= 3 &&
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
#ifdef MIPX_HAS_BZIP2
        if (bz_) {
            return bz_->getline(line);
        }
#endif
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
#ifdef MIPX_HAS_BZIP2
    std::unique_ptr<BzFileReader> bz_;
#endif
    std::unique_ptr<std::ifstream> plain_;
};

/// Zero-allocation tokenizer. MPS lines have at most 6 fields.
struct Tokens {
    std::array<std::string_view, 6> t;
    int count = 0;
    std::string_view operator[](int i) const { return t[i]; }
    int size() const { return count; }
    bool empty() const { return count == 0; }
};

Tokens tokenize(std::string_view line) {
    Tokens tok;
    std::string_view::size_type pos = 0;
    const auto len = line.size();
    while (tok.count < 6 && pos < len) {
        // Skip whitespace.
        while (pos < len && (line[pos] == ' ' || line[pos] == '\t')) ++pos;
        if (pos >= len) break;
        auto start = pos;
        while (pos < len && line[pos] != ' ' && line[pos] != '\t') ++pos;
        tok.t[tok.count++] = line.substr(start, pos - start);
    }
    return tok;
}

Real parseReal(std::string_view s) {
    const char* begin = s.data();
    const char* end = s.data() + s.size();
    // std::from_chars may not accept a leading '+' on some implementations.
    if (begin < end && *begin == '+') ++begin;
    Real val = 0.0;
    auto [ptr, ec] = std::from_chars(begin, end, val);
    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid number: " + std::string(s));
    }
    return val;
}

bool isSection(const std::string& line) {
    if (line.empty()) return false;
    // Section headers start at column 0 (no leading space).
    return !std::isspace(static_cast<unsigned char>(line[0]));
}

enum class Section { None, Name, Rows, Columns, Rhs, Ranges, Bounds, Endata };

Section parseSection(std::string_view line) {
    auto tokens = tokenize(line);
    if (tokens.empty()) return Section::None;
    const auto s = tokens[0];
    if (s == "NAME") return Section::Name;
    if (s == "ROWS") return Section::Rows;
    // LAZYCONS / USERCUTS are CPLEX extensions that define additional rows.
    if (s == "LAZYCONS" || s == "USERCUTS") return Section::Rows;
    if (s == "COLUMNS") return Section::Columns;
    if (s == "RHS") return Section::Rhs;
    if (s == "RANGES") return Section::Ranges;
    if (s == "BOUNDS") return Section::Bounds;
    if (s == "ENDATA") return Section::Endata;
    return Section::None;
}

/// Transparent hash/equal for heterogeneous unordered_map lookup.
struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const {
        return std::hash<std::string_view>{}(s);
    }
    size_t operator()(const std::string& s) const {
        return std::hash<std::string_view>{}(s);
    }
};

struct StringEqual {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const {
        return a == b;
    }
};

}  // namespace

LpProblem readMps(const std::string& filename) {
    LineReader reader(filename);

    LpProblem prob;

    // Maps for name -> index (transparent lookup avoids temporary std::string).
    std::unordered_map<std::string, Index, StringHash, StringEqual> row_map;
    std::unordered_map<std::string, Index, StringHash, StringEqual> col_map;

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

    // Column name cache for scattered column accumulation.
    std::string_view cached_col_name;
    Index cached_col_idx = -1;

    auto getOrCreateCol = [&](std::string_view name) -> Index {
        // Fast path: MPS groups columns contiguously.
        if (name == cached_col_name && cached_col_idx >= 0) {
            return cached_col_idx;
        }
        auto it = col_map.find(name);
        if (it != col_map.end()) {
            cached_col_idx = it->second;
            // cached_col_name will point into col_map key after insert,
            // but for lookup hits the name sv still works next iteration.
            cached_col_name = it->first;
            return it->second;
        }
        Index idx = prob.num_cols++;
        auto [ins_it, _] = col_map.emplace(std::string(name), idx);
        cached_col_name = ins_it->first;
        cached_col_idx = idx;
        prob.col_names.emplace_back(name);
        prob.obj.push_back(0.0);
        prob.col_lower.push_back(0.0);
        prob.col_upper.push_back(kInf);
        prob.col_type.push_back(VarType::Continuous);
        prob.col_semi_lower.push_back(0.0);
        return idx;
    };

    std::string line;
    while (reader.getline(line)) {
        // Skip empty lines and full-line comments.
        if (line.empty()) continue;
        if (line[0] == '*' || line[0] == '$') continue;

        // Strip inline '$' comments: '$' preceded by whitespace marks the
        // start of a comment.  A '$' embedded inside a name (e.g. "x$foo")
        // must not be treated as a comment delimiter.
        for (std::string::size_type pos = 1; pos < line.size(); ++pos) {
            if (line[pos] == '$' &&
                std::isspace(static_cast<unsigned char>(line[pos - 1]))) {
                line.erase(pos);
                break;
            }
        }

        if (isSection(line)) {
            section = parseSection(line);
            if (section == Section::Name) {
                auto tokens = tokenize(line);
                if (tokens.size() >= 2) {
                    prob.name = std::string(tokens[1]);
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
                auto name = tokens[1];
                if (sense == 'N') {
                    obj_row_name = std::string(name);
                    // Still add to row_map for coefficient parsing, but mark
                    // as objective.
                    row_map[std::string(name)] = -1;
                } else {
                    Index idx = prob.num_rows++;
                    row_map[std::string(name)] = idx;
                    prob.row_names.emplace_back(name);
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
                auto col_name = tokens[0];
                Index col_idx = getOrCreateCol(col_name);

                if (in_integer_section) {
                    prob.col_type[col_idx] = VarType::Integer;
                }

                // Process pairs: (row_name, value).
                for (int i = 1; i + 1 < tokens.size(); i += 2) {
                    auto row_name = tokens[i];
                    Real val = parseReal(tokens[i + 1]);

                    auto it = row_map.find(row_name);
                    if (it == row_map.end()) {
                        throw std::runtime_error(
                            "MPS: unknown row '" + std::string(row_name) + "'");
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
                for (int i = 1; i + 1 < tokens.size(); i += 2) {
                    auto row_name = tokens[i];
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
                for (int i = 1; i + 1 < tokens.size(); i += 2) {
                    auto row_name = tokens[i];
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
                auto bound_type = tokens[0];
                // tokens[1] is bound name (ignored).
                auto col_name = tokens[2];
                Index col_idx = getOrCreateCol(col_name);
                const bool has_value = tokens.size() >= 4;

                if (bound_type == "LO" && has_value) {
                    prob.col_lower[col_idx] = parseReal(tokens[3]);
                } else if (bound_type == "UP" && has_value) {
                    prob.col_upper[col_idx] = parseReal(tokens[3]);
                } else if (bound_type == "FX" && has_value) {
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
                } else if (bound_type == "LI" && has_value) {
                    prob.col_lower[col_idx] = parseReal(tokens[3]);
                    prob.col_type[col_idx] = VarType::Integer;
                } else if (bound_type == "UI" && has_value) {
                    prob.col_upper[col_idx] = parseReal(tokens[3]);
                    prob.col_type[col_idx] = VarType::Integer;
                } else if (bound_type == "SC") {
                    const Real semi_lb = (tokens.size() >= 4) ? parseReal(tokens[3]) : 1.0;
                    prob.col_type[col_idx] = VarType::SemiContinuous;
                    prob.col_lower[col_idx] = 0.0;
                    prob.col_semi_lower[col_idx] = std::max<Real>(0.0, semi_lb);
                } else if (bound_type == "SI") {
                    const Real semi_lb = (tokens.size() >= 4) ? parseReal(tokens[3]) : 1.0;
                    prob.col_type[col_idx] = VarType::SemiInteger;
                    prob.col_lower[col_idx] = 0.0;
                    prob.col_semi_lower[col_idx] = std::max<Real>(0.0, semi_lb);
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
