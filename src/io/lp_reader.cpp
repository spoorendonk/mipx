#include "mipx/io.h"

#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mipx {

namespace {

enum class LpSection {
    None,
    Objective,
    Constraints,
    Bounds,
    General,
    Binary,
    End
};

std::string toLower(std::string s) {
    for (auto& c : s)
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    return s;
}

LpSection parseLpSection(const std::string& line) {
    std::string lower = toLower(line);
    // Remove trailing colon and whitespace.
    while (!lower.empty() && (lower.back() == ':' || lower.back() == ' '))
        lower.pop_back();

    if (lower == "minimize" || lower == "minimum" || lower == "min")
        return LpSection::Objective;
    if (lower == "maximize" || lower == "maximum" || lower == "max")
        return LpSection::Objective;
    if (lower == "subject to" || lower == "such that" || lower == "st" ||
        lower == "s.t." || lower == "subject to:")
        return LpSection::Constraints;
    if (lower == "bounds") return LpSection::Bounds;
    if (lower == "general" || lower == "generals" || lower == "gen")
        return LpSection::General;
    if (lower == "binary" || lower == "binaries" || lower == "bin")
        return LpSection::Binary;
    if (lower == "end") return LpSection::End;
    return LpSection::None;
}

bool isSenseIndicator(const std::string& s) {
    return s == "<=" || s == ">=" || s == "=" || s == "<" || s == ">";
}

/// Parse a linear expression like "2 x1 + 3 x2 - x3" into (name, coeff)
/// pairs.
std::vector<std::pair<std::string, Real>> parseExpression(
    const std::vector<std::string>& tokens, size_t& pos) {
    std::vector<std::pair<std::string, Real>> terms;
    Real sign = 1.0;

    while (pos < tokens.size()) {
        const auto& tok = tokens[pos];
        if (isSenseIndicator(tok)) break;

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
        auto [ptr, ec] =
            std::from_chars(tok.data(), tok.data() + tok.size(), coeff);
        if (ec == std::errc{} &&
            ptr == tok.data() + tok.size()) {
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

}  // namespace

LpProblem readLp(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    LpProblem prob;
    std::unordered_map<std::string, Index> col_map;

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

    // Read all lines and join continuation lines.
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        // Remove comments.
        auto bslash = line.find('\\');
        if (bslash != std::string::npos) line.resize(bslash);
        // Trim trailing whitespace.
        while (!line.empty() &&
               std::isspace(static_cast<unsigned char>(line.back())))
            line.pop_back();
        if (!line.empty()) lines.push_back(std::move(line));
    }

    LpSection section = LpSection::None;
    prob.sense = Sense::Minimize;

    std::vector<Triplet> triplets;

    for (size_t li = 0; li < lines.size(); ++li) {
        const auto& l = lines[li];

        // Check for section header.
        LpSection new_section = parseLpSection(l);
        if (new_section != LpSection::None) {
            if (new_section == LpSection::Objective) {
                std::string lower = toLower(l);
                if (lower.starts_with("max"))
                    prob.sense = Sense::Maximize;
                else
                    prob.sense = Sense::Minimize;
            }
            section = new_section;
            if (section == LpSection::End) break;
            continue;
        }

        // Tokenize.
        std::istringstream iss(l);
        std::vector<std::string> tokens;
        std::string tok;
        while (iss >> tok) tokens.push_back(std::move(tok));
        if (tokens.empty()) continue;

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

                if (pos >= tokens.size()) break;
                std::string sense_str = tokens[pos++];

                Real rhs_val = 0.0;
                if (pos < tokens.size()) {
                    auto [ptr, ec] = std::from_chars(
                        tokens[pos].data(),
                        tokens[pos].data() + tokens[pos].size(), rhs_val);
                    if (ec != std::errc{}) rhs_val = 0.0;
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

                if (sense_str == "<=" || sense_str == "<") {
                    prob.row_lower.push_back(-kInf);
                    prob.row_upper.push_back(rhs_val);
                } else if (sense_str == ">=" || sense_str == ">") {
                    prob.row_lower.push_back(rhs_val);
                    prob.row_upper.push_back(kInf);
                } else {
                    prob.row_lower.push_back(rhs_val);
                    prob.row_upper.push_back(rhs_val);
                }
                break;
            }

            case LpSection::Bounds: {
                // Forms: lb <= x <= ub, x >= lb, x <= ub, x = val, x free
                if (tokens.size() >= 2 && toLower(tokens.back()) == "free") {
                    Index idx = getOrCreateCol(tokens[0]);
                    prob.col_lower[idx] = -kInf;
                    prob.col_upper[idx] = kInf;
                } else if (tokens.size() >= 5 &&
                           (tokens[1] == "<=" || tokens[1] == "<") &&
                           (tokens[3] == "<=" || tokens[3] == "<")) {
                    // lb <= x <= ub
                    Real lb = 0.0;
                    std::from_chars(tokens[0].data(),
                                    tokens[0].data() + tokens[0].size(), lb);
                    Index idx = getOrCreateCol(tokens[2]);
                    Real ub = 0.0;
                    std::from_chars(tokens[4].data(),
                                    tokens[4].data() + tokens[4].size(), ub);
                    prob.col_lower[idx] = lb;
                    prob.col_upper[idx] = ub;
                } else if (tokens.size() >= 3) {
                    // x >= lb, x <= ub, x = val
                    Index idx = getOrCreateCol(tokens[0]);
                    Real val = 0.0;
                    std::from_chars(tokens[2].data(),
                                    tokens[2].data() + tokens[2].size(), val);
                    if (tokens[1] == ">=" || tokens[1] == ">") {
                        prob.col_lower[idx] = val;
                    } else if (tokens[1] == "<=" || tokens[1] == "<") {
                        prob.col_upper[idx] = val;
                    } else if (tokens[1] == "=") {
                        prob.col_lower[idx] = val;
                        prob.col_upper[idx] = val;
                    }
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

    prob.matrix =
        SparseMatrix(prob.num_rows, prob.num_cols, std::move(triplets));

    return prob;
}

}  // namespace mipx
