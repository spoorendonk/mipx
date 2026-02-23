#include "mipx/io.h"

#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mipx {

std::vector<SoluEntry> readSolu(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<SoluEntry> entries;
    std::string line;

    while (std::getline(in, line)) {
        // Skip empty lines and comments.
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string tag, name;
        iss >> tag >> name;

        if (tag == "=inf=") {
            entries.push_back({std::move(name), 0.0, true});
        } else if (tag == "=opt=") {
            Real val = 0.0;
            std::string val_str;
            iss >> val_str;
            if (!val_str.empty()) {
                std::from_chars(val_str.data(),
                                val_str.data() + val_str.size(), val);
            }
            entries.push_back({std::move(name), val, false});
        }
        // Skip lines that don't match the expected format.
    }

    return entries;
}

}  // namespace mipx
