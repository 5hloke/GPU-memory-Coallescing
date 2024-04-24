#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <climits>
#include <fstream>
#include <sstream>

using namespace std;

// Define memory trace type
using MemoryTrace = std::map<size_t,                             // static id
                        std::map<size_t,                         // dynamic id
                            std::map<size_t,                     // tidy
                                std::map<size_t,                 // tidz
                                    std::map<size_t, long long>>>>>; // tidx => addr

// Function to sort a vector by increasing value
void sortByIncreasingValue(vector<long long>& vec) {
    sort(vec.begin(), vec.end());
}



void memoryTraceAnalysis(const MemoryTrace &memoryTrace, int W, size_t dataTypeSize) {
    for (const auto& [static_id, dynamicMap] : memoryTrace) {
        long long minStride = LLONG_MAX;
        long long maxStride = 0;
        long long avgStride = 0;
        long long allStr = 0;
        vector<long long> allAddrs;

        // Iterate over dynamic ids
        for (const auto& [dynamic_id, tidyMap] : dynamicMap) {
            // Build linearized list of memory addresses accessed
            vector<long long> v;
            for (const auto& [tidy, tidzMap] : tidyMap) {
                for (const auto& [tidz, tidxMap] : tidzMap) {
                    for (const auto& [tidx, addr] : tidxMap) {
                        v.push_back(addr);
                    }
                }
            }

            // Sort W consecutive elements of V
            for (size_t c = 0; c < v.size(); c += W) {
                sortByIncreasingValue(v);
                size_t end = min(c + W, v.size());
                vector<long long>::iterator first = v.begin() + c;
                vector<long long>::iterator last = v.begin() + end;
                sort(first, last);

                // Calculate stride and update min, max, avg strides
                for (size_t j = 1; j < end - c; ++j) {
                    long long stride = v[c + j] - v[c + j - 1];
                    minStride = min(minStride, stride);
                    maxStride = max(maxStride, stride);
                    avgStride += stride;
                }
                avgStride = avgStride / (end - c);
            }

            // Concatenate memory addresses to AllAddrs
            allAddrs.insert(allAddrs.end(), v.begin(), v.end());
        }

        // Check if entire memory space accessed is contiguous
        sortByIncreasingValue(allAddrs);
        for (size_t i = 1; i < allAddrs.size(); ++i) {
            allStr = max(allStr, allAddrs[i] - allAddrs[i - 1]);
        }

        // TODO: Get load/store type based on how memory trace is provided
        cout << "Static id: " << static_id << ", Load/Store type: ";
        cout << "Min Stride: " << minStride << ", Max Stride: " << maxStride << ", Avg Stride: " << avgStride << endl;

        if (maxStride <= dataTypeSize) {
            cout << "Coalesced" << endl;
            // insert counter here
        } else {
            cout << "Uncoalesced" << endl;

            if (allStr > dataTypeSize) {
                cout << "Accesses cannot be all coalesced" << endl;
            } else {
                cout << "Suggest thread geometry transformations" << endl;
                // TODO: Adjust once we know load/store from memory trace
                // if (instruction == "load") {
                //     cout << "Suggest shared memory usage" << endl;
                // }
            }
        }
    }
}

void readTracesFromFile(MemoryTrace& memoryTrace, const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }

    while (getline(file, line)) {
        std::istringstream iss(line);
        size_t grid_x, grid_y, grid_z, tid_x, tid_y, tid_z, is_ld, current_kept_static_id, dynamic_id, address;
        if (!(iss >> grid_x >> grid_y >> grid_z >> tid_x >> tid_y >> tid_z >> is_ld >> current_kept_static_id >> dynamic_id >> address)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }

        memoryTrace[current_kept_static_id][dynamic_id][tid_y][tid_z][tid_x] = address;
    }

    file.close();
    std::cout << "Finished reading all lines." << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    MemoryTrace memoryTrace; // maximum size is 2^64 -1 so we should be fine
    readTracesFromFile(memoryTrace, argv[1]);

    int W = 16; // Warp scheduling size
    size_t dataTypeSize = sizeof(long long); 

    memoryTraceAnalysis(memoryTrace, W, dataTypeSize);

    return 0;
}

// int main() {
//     MemoryTrace memoryTrace;
//     memoryTrace[31][0][0][0][0] = 30066082304;
//     memoryTrace[34][0][0][0][0] = 30066081792;
//     memoryTrace[31][1][0][0][0] = 30066082320;
//     memoryTrace[34][1][0][0][0] = 30066081796;
//     memoryTrace[31][2][0][0][0] = 30066082336;
//     memoryTrace[34][2][0][0][0] = 30066081800;
//     memoryTrace[31][0][0][1][0] = 30066082320;
//     memoryTrace[34][0][0][1][0] = 30066081796;
//     memoryTrace[31][1][0][1][0] = 30066082336;
//     memoryTrace[34][1][0][1][0] = 30066081800;
//     memoryTrace[31][2][0][1][0] = 30066082352;
//     memoryTrace[34][2][0][1][0] = 30066081804;
    // int W = 16; // Warp scheduling size
    // size_t dataTypeSize = sizeof(long long); 

    // memoryTraceAnalysis(memoryTrace, W, dataTypeSize);

//     return 0;
// }