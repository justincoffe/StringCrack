#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <string.h>
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

// --- Base58 Decode Helper ---
static const char* b58chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
bool DecodeBase58Address(const char *str, unsigned char *outHash160) {
    unsigned char buf[25] = {0};
    while (*str) {
        const char *p = strchr(b58chars, *str);
        if (!p) return false; // Invalid character
        int carry = p - b58chars;
        for (int i = 24; i >= 0; --i) {
            carry += 58 * buf[i];
            buf[i] = carry % 256;
            carry /= 256;
        }
        str++;
    }
    memcpy(outHash160, buf + 1, 20); // Skip the 1-byte version prefix
    return true;
}

// --- Hex Parser Helper ---
void hexToBytes(std::string hex, unsigned char* bytes) {
    memset(bytes, 0, 32);
    if (hex.length() % 2 != 0) hex = "0" + hex; 
    int len = hex.length();
    for (int i = 0; i < len; i += 2) {
        std::string byteString = hex.substr(len - 2 - i, 2);
        unsigned char byte = (unsigned char)strtol(byteString.c_str(), NULL, 16);
        bytes[31 - (i / 2)] = byte;
    }
}

// --- Byte to Hex Printer ---
void printHex(const unsigned char* bytes, int len) {
    for (int i = 0; i < len; i++) {
        printf("%02X", bytes[i]);
    }
    printf("\n");
}

int main() {
    // ========================================================================
    // === INDEX 71: STRATIFIED ENTROPY PERMUTATION (SEP) STRIKE            ===
    // ========================================================================
    
    std::string targetAddress = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU";
    
    // The Base Hex: 101101001010110011010100101001101011 (shifted up 35 bits)
    // We pad the bottom 35 bits with 0s for the base prediction.
    std::string baseHex = "5A566A53500000000"; 
    
    int puzzleBits = 71;                    
    
    // The 8 Indestructible Anchors (Deterministically Locked)
    // Plus the bottom 35 bits if you are doing this pure CPU, 
    // OR just the 8 anchors if passing to the GPU SCPRO muscle.
    std::vector<int> lockedBits = {70, 69, 66, 62, 54, 52, 46, 41}; 
    
    // The Binomial Boundaries for the 28 Volatile Bits
    int minFlips = 11;
    int maxFlips = 23;
    
    // ========================================================================

    unsigned char targetHash160[20];
    if (!DecodeBase58Address(targetAddress.c_str(), targetHash160)) {
        std::cout << "Error: Invalid Target Bitcoin Address!\n";
        return 1;
    }

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
    std::cout << "Target Address : " << targetAddress << "\n";
    std::cout << "Locked Bits    : " << lockedBits.size() << " | Free Bits: " << numFreeBits << "\n";

    // 1. Generate exact permutations
    std::cout << "Generating exact combinatorial permutations...\n";
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
    std::cout << "Keyspace Volume: " << totalPermutations << " candidates.\n";
    std::cout << "Igniting EPYC cores...\n\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool found = false;

    // 2. OpenMP Heavy Evaluation Loop
    #pragma omp parallel
    {
        // Each thread requires its own context to avoid locks
        secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
        
        unsigned char testPriv[32];
        secp256k1_pubkey pubkey;
        unsigned char pub[33];
        size_t publen;
        unsigned char sha[SHA256_DIGEST_LENGTH];
        unsigned char rmd[RIPEMD160_DIGEST_LENGTH];

        #pragma omp for schedule(dynamic, 10000)
        for (size_t i = 0; i < totalPermutations; i++) {
            if (found) continue; // Early exit if another thread found it

            // Copy base string and apply XOR mask
            memcpy(testPriv, basePriv, 32);
            uint64_t mask = flipMasks[i];
            
            for (int bit = 0; bit < numFreeBits; bit++) {
                if ((mask >> bit) & 1) {
                    int pos = freeBits[bit];
                    int byteIdx = 31 - (pos / 8);
                    int bitIdx = pos % 8;
                    testPriv[byteIdx] ^= (1 << bitIdx); 
                }
            }

            // EC Math: Compute Public Key
            if (!secp256k1_ec_pubkey_create(ctx, &pubkey, testPriv)) continue;

            // Serialize Compressed (33 bytes)
            publen = 33;
            secp256k1_ec_pubkey_serialize(ctx, pub, &publen, &pubkey, SECP256K1_EC_COMPRESSED);

            // Hash: SHA256 -> RIPEMD160
            SHA256(pub, publen, sha);
            RIPEMD160(sha, SHA256_DIGEST_LENGTH, rmd);

            // Check against Target
            if (memcmp(rmd, targetHash160, 20) == 0) {
                #pragma omp critical
                {
                    found = true;
                    std::cout << "=================================================\n";
                    std::cout << "[SUCCESS] MATCH FOUND!\n";
                    std::cout << "Target Address : " << targetAddress << "\n";
                    std::cout << "Private Key Hex: 0x";
                    printHex(testPriv, 32);
                    std::cout << "=================================================\n";
                }
            }
        }
        secp256k1_context_destroy(ctx);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "\nFinished in " << diff.count() << " seconds.\n";
    std::cout << "Throughput: " << (totalPermutations / diff.count() / 1000000.0) << " MK/s\n";

    return 0;
}
