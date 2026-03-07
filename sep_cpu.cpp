#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <secp256k1.h>
#include <omp.h>
#include <string.h>

// Helper to convert hex string to 32-byte array
void hexToBytes(const std::string& hex, unsigned char* bytes) {
    memset(bytes, 0, 32);
    int len = hex.length();
    for (int i = 0; i < len; i += 2) {
        std::string byteString = hex.substr(len - 2 - i, 2);
        unsigned char byte = (unsigned char)strtol(byteString.c_str(), NULL, 16);
        bytes[31 - (i / 2)] = byte;
    }
}

int main() {
    // --- SEP CONFIGURATION ---
    std::string baseHex = "22BD43C2E9354"; // Your Markov Prediction (S_base)
    int puzzleBits = 50;
    std::vector<int> lockedBits = {49, 48, 25, 12, 3}; // Your Deterministic Anchors
    int minFlips = 8;
    int maxFlips = 16;
    // -------------------------

    unsigned char basePriv[32];
    hexToBytes(baseHex, basePriv);

    std::vector<int> freeBits;
    for (int i = 0; i < puzzleBits; i++) {
        bool isLocked = false;
        for (int locked : lockedBits) {
            if (i == locked) isLocked = true;
        }
        if (!isLocked) freeBits.push_back(i);
    }
    
    int numFreeBits = freeBits.size();
    std::cout << "Locked Bits: " << lockedBits.size() << " | Free Bits: " << numFreeBits << "\n";

    // 1. Generate exact permutations using Gosper's Hack
    std::cout << "Generating exact permutations...\n";
    std::vector<uint64_t> flipMasks;
    
    for (int k = minFlips; k <= maxFlips; k++) {
        uint64_t set = (1ULL << k) - 1;
        uint64_t limit = (1ULL << numFreeBits);
        while (set < limit) {
            flipMasks.push_back(set);
            uint64_t c = set & -set;
            uint64_t r = set + c;
            set = (((r ^ set) >> 2) / c) | r;
        }
    }
    
    size_t totalPermutations = flipMasks.size();
    std::cout << "Targeted Space Volume: " << totalPermutations << " candidates.\n";

    // 2. OpenMP Evaluation Loop
    std::cout << "Igniting EPYC cores...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // Each thread gets its own context to avoid locks
        secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
        unsigned char testPriv[32];
        secp256k1_pubkey pubkey;
        
        #pragma omp for schedule(dynamic, 10000)
        for (size_t i = 0; i < totalPermutations; i++) {
            memcpy(testPriv, basePriv, 32);
            uint64_t mask = flipMasks[i];
            
            // Map the generic mask to the actual free bit positions (XOR mutation)
            for (int bit = 0; bit < numFreeBits; bit++) {
                if ((mask >> bit) & 1) {
                    int pos = freeBits[bit];
                    int byteIdx = 31 - (pos / 8);
                    int bitIdx = pos % 8;
                    testPriv[byteIdx] ^= (1 << bitIdx); // XOR flip
                }
            }

            // Test the key
            if (secp256k1_ec_pubkey_create(ctx, &pubkey, testPriv)) {
                // ADD YOUR TARGET HASH CHECKING HERE
                // For this example, we just simulate the CPU load
            }
        }
        secp256k1_context_destroy(ctx);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "Finished in " << diff.count() << " seconds.\n";
    std::cout << "Throughput: " << (totalPermutations / diff.count() / 1000000.0) << " MK/s\n";

    return 0;
}
