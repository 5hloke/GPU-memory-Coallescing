#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Define memory trace type - Really ugly right now, but I tried to match something that would be simlar to the psuedocode
using MemoryTrace = vector<vector<vector<vector<vector<long long>>>>>;

// Function to sort a vector by increasing value
void sortByIncreasingValue(vector<long long>& vec) {
    sort(vec.begin(), vec.end());
}


void memoryTraceAnalysis(const MemoryTrace &memoryTrace, int W, size_t dataTypeSize) {
    for (size_t static_id = 0; static_id < memoryTrace.size(); ++static_id) {
        long long minStride = LLONG_MAX, maxStride = 0, avgStride = 0, allStr = 0;
        vector<long long> allAddrs;

        // Iterate over dynamic ids
        for (size_t dynamic_id = 0; dynamic_id < memoryTrace[static_id].size(); ++dynamic_id) {
            // Build linearized list of memory addresses accessed
            vector<long long> v;
            for (size_t tidy = 0; tidy < memoryTrace[static_id][dynamic_id].size(); ++tidy) {
                for (size_t tidz = 0; tidz < memoryTrace[static_id][dynamic_id][tidy].size(); ++tidz) {
                    for (size_t tidx = 0; tidx < memoryTrace[static_id][dynamic_id][tidy][tidz].size(); ++tidx) {
                        v.push_back(memoryTrace[static_id][dynamic_id][tidy][tidz][tidx]);
                    }
                }
            }

            // Sort W consecutive elements of V
            for (size_t c = 0; c < v.size(); c += W) {
                // Creatin necessary subrange of vector, while ensuring it stays within bounds
                sortByIncreasingValue(v.begin() + c, v.begin() + min(c + W, v.size()));

                // Calculate stride and update min, max, avg strides
                for (size_t j = 1; j < W; ++j) {
                    long long stride = v[c + j] - v[c + j - 1];
                    minStride = min(minStride, stride);
                    maxStride = max(maxStride, stride);
                    avgStride += stride;
                }
                avgStride=avgStride / W;
            }

            // Concatenate memory addresses to AllAddrs
            allAddrs.insert(allAddrs.end(), v.begin(), v.end());
        }

        // Check if entire memory space accessed is contiguous
        sortByIncreasingValue(allAddrs);
        for (size_t i = 1; i < allAddrs.size(); ++i) {
            allStr = max(allStr, allAddrs[i] - allAddrs[i - 1]);
        }

// TODO:: Get load/store type based on how memory trace is provided 
        cout << "Static id: " << static_id << ", Load/Store type: "; 
        cout << "Min Stride: " << minStride << ", Max Stride: " << maxStride << ", Avg Stride: " << avgStride  << endl;
        
        if (maxStride <= dataTypeSize) {
            cout << "Coalesced" << endl;
        } else {
            cout << "Uncoalesced" << endl;
        
            if (allStr > dataTypeSize) {
                cout << "Accesses cannot be all coalesced" << endl;
            } else {
                cout << "Suggest thread geometry transformations" << endl;
                // TODO:: Adjust once we know load/store from memory trace
                if (instruction == "load") {
                    cout << "Suggest also shared memory usage" << endl;
                }
            }
        }
    }
}

int main() {
    MemoryTrace memoryTrace;
    int W = 16; // Warp scheduling size
    size_t dataTypeSize = sizeof(long long); 
    
    memoryTraceAnalysis(memoryTrace, W, dataTypeSize);

    return 0;
}
