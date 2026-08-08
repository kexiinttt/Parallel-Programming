#include <vector>
#include <string>
#include <iostream>
#include <fstream>

inline std::vector<float> readVectorFromFile(const std::string &fileName) {
    std::ifstream inputFile(fileName);
    if (!inputFile) {
        throw std::runtime_error("Could not open input file: " + fileName);
    }

    std::vector<float> values;
    float value;
    while (inputFile >> value) {
        values.push_back(value);
    }

    if (!inputFile.eof()) {
        throw std::runtime_error("Invalid floating-point value in file: " + fileName);
    }

    return values;
};


inline int compare(
    const std::vector<float>& output,
    const std::vector<float>& expected,
    float tolerance = 1e-5f
) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(output[i] - expected[i]) > tolerance) {
            return 1;
        }
    }
    return 0;
}
