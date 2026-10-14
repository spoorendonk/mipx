#include "mipx/io.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace mipx {

namespace {

enum class LpSection { None, Objective, Constraints, Bounds, General, Binary, End };

std::string toLower(std::string s) {
    for (auto& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

LpSection parseLpSection(const std::string& line) {
    std::string lower = toLower(line);
    // Remove trailing colon and whitespace.
    while (!lower.empty() && (lower.back() == ':' || lower.back() == ' ')) {
        lower.pop_back();
    }

    if (lower == "minimize" || lower == "minimum" || lower == "min") {
        return LpSection::Objective;
    }
    if (lower == "maximize" || lower == "maximum" || lower == "max") {
        return LpSection::Objective;
    }
    if (lower == "subject to" || lower == "such that" || lower == "st" || lower == "s.t." ||
        lower == "subject to:") {
        return LpSection::Constraints;
    }
    if (lower == "bounds") {
        return LpSection::Bounds;
    }
    if (lower == "general" || lower == "generals" || lower == "gen") {
        return LpSection::General;
    }
    if (lower == "binary" || lower == "binaries" || lower == "bin") {
        return LpSection::Binary;
    }
    if (lower == "end") {
        return LpSection::End;
    }
    return LpSection::None;
}

// CPLEX LP also spells the inequalities "=<" and "=>".
bool isLessEqual(const std::string& s) {
    return s == "<=" || s == "<" || s == "=<";
}

bool isGreaterEqual(const std::string& s) {
    return s == ">=" || s == ">" || s == "=>";
}

bool isSenseIndicator(const std::string& s) {
    return isLessEqual(s) || isGreaterEqual(s) || s == "=";
}

/// Parse a linear expression like "2 x1 + 3 x2 - x3" into (name, coeff)
/// pairs.
std::vector<std::pair<std::string, Real>> parseExpression(const std::vector<std::string>& tokens,
                                                          size_t& pos) {
    std::vector<std::pair<std::string, Real>> terms;
    Real sign = 1.0;

    while (pos < tokens.size()) {
        const auto& tok = tokens[pos];
        if (isSenseIndicator(tok)) {
            break;
        }

        if (tok == "+") {
            sign = 1.0;
            ++pos;
            continue;
        }
        if (tok == "-") {
            sign = -1.0;
            ++pos;
            continue;
        }

        // Try to parse as number.
        Real coeff;
        auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), coeff);
        if (ec == std::errc{} && ptr == tok.data() + tok.size()) {
            // Pure number — next token should be variable name.
            coeff *= sign;
            sign = 1.0;
            if (pos + 1 < tokens.size() && !isSenseIndicator(tokens[pos + 1]) &&
                tokens[pos + 1] != "+" && tokens[pos + 1] != "-") {
                ++pos;
                terms.emplace_back(tokens[pos], coeff);
            }
            // else: constant term, skip for now
            ++pos;
        } else {
            // Variable name (coefficient is 1).
            terms.emplace_back(tok, sign * 1.0);
            sign = 1.0;
            ++pos;
        }
    }
    return terms;
}

/// Parse a bound value: a number, optionally signed with '+', or an infinity
/// spelled the way std::from_chars accepts it ("inf", "-infinity", ...).
/// Returns nullopt for anything else, rather than the 0.0 an unchecked
/// from_chars would leave behind.
std::optional<Real> parseBoundValue(std::string_view tok) {
    if (tok.size() > 1 && tok.front() == '+') {
        tok.remove_prefix(1);
    }
    Real value = 0.0;
    auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), value);
    // NaN is no bound, so "nan" stays available as a column name.
    if (ec != std::errc{} || ptr != tok.data() + tok.size() || std::isnan(value)) {
        return std::nullopt;
    }
    return value;
}

/// A column name in a bounds line: anything that is neither a number nor an
/// operator.
bool isBoundName(const std::string& tok) {
    return !isSenseIndicator(tok) && !parseBoundValue(tok);
}

struct BoundLine {
    std::string name;
    std::optional<Real> lower;
    std::optional<Real> upper;
};

/// Recognise one line of the Bounds section, or return nullopt when it matches
/// none of the forms:
///   x free | x <= u | x >= l | x = v | l <= x | u >= x | v = x | l <= x <= u
/// ('<' and '=<' are read as '<=', '>' and '=>' as '>=').
std::optional<BoundLine> parseBoundLine(const std::vector<std::string>& tokens) {
    const auto isLe = isLessEqual;
    const auto isGe = isGreaterEqual;

    if (tokens.size() == 2 && isBoundName(tokens[0]) && toLower(tokens[1]) == "free") {
        return BoundLine{tokens[0], -kInf, kInf};
    }
    if (tokens.size() == 3) {
        const std::string& op = tokens[1];
        if (isBoundName(tokens[0])) {
            // x op value
            const auto value = parseBoundValue(tokens[2]);
            if (!value) {
                return std::nullopt;
            }
            if (isLe(op)) {
                return BoundLine{tokens[0], std::nullopt, *value};
            }
            if (isGe(op)) {
                return BoundLine{tokens[0], *value, std::nullopt};
            }
            if (op == "=") {
                return BoundLine{tokens[0], *value, *value};
            }
            return std::nullopt;
        }
        // value op x
        const auto value = parseBoundValue(tokens[0]);
        if (!value || !isBoundName(tokens[2])) {
            return std::nullopt;
        }
        if (isLe(op)) {
            return BoundLine{tokens[2], *value, std::nullopt};
        }
        if (isGe(op)) {
            return BoundLine{tokens[2], std::nullopt, *value};
        }
        if (op == "=") {
            return BoundLine{tokens[2], *value, *value};
        }
        return std::nullopt;
    }
    if (tokens.size() == 5 && isLe(tokens[1]) && isLe(tokens[3]) && isBoundName(tokens[2])) {
        const auto lower = parseBoundValue(tokens[0]);
        const auto upper = parseBoundValue(tokens[4]);
        if (!lower || !upper) {
            return std::nullopt;
        }
        return BoundLine{tokens[2], *lower, *upper};
    }
    return std::nullopt;
}

}  // namespace

LpProblem readLp(const std::string& filename, ReadDiagnostics* diag) {
    if (diag != nullptr) {
        *diag = ReadDiagnostics{};
    }

    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    LpProblem prob;
    std::unordered_map<std::string, Index> col_map;

    auto getOrCreateCol = [&](const std::string& name) -> Index {
        auto it = col_map.find(name);
        if (it != col_map.end()) {
            return it->second;
        }
        Index idx = prob.num_cols++;
        col_map[name] = idx;
        prob.col_names.push_back(name);
        prob.obj.push_back(0.0);
        prob.col_lower.push_back(0.0);
        prob.col_upper.push_back(kInf);
        prob.col_type.push_back(VarType::Continuous);
        return idx;
    };

    // Read all lines and join continuation lines. Each kept line carries its
    // 1-based line number in the file, for error messages.
    std::vector<std::pair<size_t, std::string>> lines;
    std::string line;
    size_t line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        // Recorded on the raw line, before comments are stripped: this answers
        // "was there anything in the file", not "was there anything we
        // understood".
        if (diag != nullptr && !line.empty()) {
            diag->saw_content = true;
        }
        // Remove comments.
        auto bslash = line.find('\\');
        if (bslash != std::string::npos) {
            line.resize(bslash);
        }
        // Trim trailing whitespace.
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.emplace_back(line_number, std::move(line));
        }
    }

    LpSection section = LpSection::None;
    prob.sense = Sense::Minimize;

    std::vector<Triplet> triplets;

    for (size_t li = 0; li < lines.size(); ++li) {
        const auto& [l_number, l] = lines[li];

        // Check for section header.
        LpSection new_section = parseLpSection(l);
        if (new_section != LpSection::None) {
            // Bounds is excluded: MPS spells a section BOUNDS as well, so an
            // MPS file fed to this reader would otherwise look like LP.
            if (diag != nullptr && new_section != LpSection::Bounds) {
                diag->saw_format_section = true;
            }
            if (new_section == LpSection::Objective) {
                std::string lower = toLower(l);
                if (lower.starts_with("max")) {
                    prob.sense = Sense::Maximize;
                } else {
                    prob.sense = Sense::Minimize;
                }
            }
            section = new_section;
            if (section == LpSection::End) {
                break;
            }
            continue;
        }

        // Tokenize.
        std::istringstream iss(l);
        std::vector<std::string> tokens;
        std::string tok;
        while (iss >> tok) {
            tokens.push_back(std::move(tok));
        }
        if (tokens.empty()) {
            continue;
        }

        // Skip constraint name (label: ).
        size_t start = 0;
        if (tokens.size() > 1 && tokens[0].back() == ':') {
            start = 1;
        }

        switch (section) {
            case LpSection::Objective: {
                size_t pos = start;
                auto terms = parseExpression(tokens, pos);
                for (auto& [name, coeff] : terms) {
                    Index idx = getOrCreateCol(name);
                    prob.obj[idx] += coeff;
                }
                break;
            }

            case LpSection::Constraints: {
                size_t pos = start;
                auto lhs_terms = parseExpression(tokens, pos);

                if (pos >= tokens.size()) {
                    break;
                }
                std::string sense_str = tokens[pos++];

                Real rhs_val = 0.0;
                if (pos < tokens.size()) {
                    auto [ptr, ec] = std::from_chars(
                        tokens[pos].data(), tokens[pos].data() + tokens[pos].size(), rhs_val);
                    if (ec != std::errc{}) {
                        rhs_val = 0.0;
                    }
                }

                Index row = prob.num_rows++;
                // Extract constraint name if present.
                if (start > 0) {
                    std::string name = tokens[0];
                    name.pop_back();  // remove ':'
                    prob.row_names.push_back(std::move(name));
                } else {
                    prob.row_names.push_back("R" + std::to_string(row));
                }

                for (auto& [name, coeff] : lhs_terms) {
                    Index col = getOrCreateCol(name);
                    triplets.push_back({row, col, coeff});
                }

                if (isLessEqual(sense_str)) {
                    prob.row_lower.push_back(-kInf);
                    prob.row_upper.push_back(rhs_val);
                } else if (isGreaterEqual(sense_str)) {
                    prob.row_lower.push_back(rhs_val);
                    prob.row_upper.push_back(kInf);
                } else {
                    prob.row_lower.push_back(rhs_val);
                    prob.row_upper.push_back(rhs_val);
                }
                break;
            }

            case LpSection::Bounds: {
                // Recognise the whole line before touching the model, so a
                // malformed line cannot leave a phantom column behind.
                const auto bound = parseBoundLine(tokens);
                if (!bound) {
                    throw ModelFormatError(
                        "Cannot read '" + filename + "' as CPLEX LP: line " +
                        std::to_string(l_number) +
                        " is not a recognised bound (expected "
                        "'x free', 'x <= u', 'x >= l', 'x = v', 'l <= x', 'u >= x', "
                        "'v = x' or 'l <= x <= u', or a section header this reader knows): " +
                        l);
                }
                const Index idx = getOrCreateCol(bound->name);
                if (bound->lower) {
                    prob.col_lower[idx] = *bound->lower;
                }
                if (bound->upper) {
                    prob.col_upper[idx] = *bound->upper;
                }
                break;
            }

            case LpSection::General: {
                for (size_t i = start; i < tokens.size(); ++i) {
                    Index idx = getOrCreateCol(tokens[i]);
                    prob.col_type[idx] = VarType::Integer;
                }
                break;
            }

            case LpSection::Binary: {
                for (size_t i = start; i < tokens.size(); ++i) {
                    Index idx = getOrCreateCol(tokens[i]);
                    prob.col_type[idx] = VarType::Binary;
                    prob.col_lower[idx] = 0.0;
                    prob.col_upper[idx] = 1.0;
                }
                break;
            }

            default:
                break;
        }
    }

    prob.matrix = SparseMatrix(prob.num_rows, prob.num_cols, std::move(triplets));

    return prob;
}

}  // namespace mipx
