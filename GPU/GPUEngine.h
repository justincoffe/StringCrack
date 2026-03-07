/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include <string>
#include <cstdint>
#include "../SECP256k1.h"
#include "../Int.h"
#include <cuda_runtime.h>

#define SEARCH_COMPRESSED 0
#define SEARCH_UNCOMPRESSED 1
#define SEARCH_BOTH 2

// Number of thread per block
#define ITEM_SIZE 28
#define ITEM_SIZE32 (ITEM_SIZE/4)
#define _64K 65536

// Maximum number of locked bit positions for StringCrack Bit Injection
#define MAX_LOCKED_BITS 128

static const char *searchModes[] = {"Compressed","Uncompressed","Compressed or Uncompressed"};

typedef uint16_t address_t;
typedef uint32_t addressl_t;

typedef struct {
  uint32_t thId;
  int16_t  incr;
  int16_t  endo;
  uint8_t  *hash;
  bool mode;
} ITEM;

// =====================================================================================
// StringCrack Configuration for Bit Injection + Popcount Filtering
// =====================================================================================

typedef struct {
    int position;
    int value;
} LockedBit;

typedef struct {
    bool enabled;
    int numLockedBits;
    LockedBit lockedBits[MAX_LOCKED_BITS];
    int numFreeBits;
    int freeBitPositions[256];
    int popcountTarget;
    int popcountMin;
    int popcountMax;
    int puzzleBits;
    uint64_t lockMask[4];
    uint64_t lockVals[4];
    
    // Precomputed base point (sum of locked bits * G)
    uint64_t basePointX[4];
    uint64_t basePointY[4];
    
    // Precomputed G table for ONLY free bits (compact)
    uint64_t free_GX[256][4];
    uint64_t free_GY[256][4];
    
    // Locked popcount (precomputed for O(1) filter)
    int lockedPopcount;

    // Seed offset: -start HEX sets this as the starting seed
    // -range N sets the power-of-two size of the seed space
    // -end N (optional) limits the scan to a subset of the range
    uint64_t seedOffset;        // Starting seed (from -start)
    uint64_t seedCount;         // Total seeds to scan (2^range or computed)
    int endBits;                // Sub-range size in bits (from -end), -1 if not set
    
    // Full 256-bit versions for CPU calculations
    Int seedOffsetInt;          // Full 256-bit seed offset
    Int seedCountInt;           // Full 256-bit seed count
    Int seedEndInt;             // Full 256-bit end offset (start + 2^endBits)
    
    // Weak bits for Fault-Tolerant Blast Radius
    int weakBits[10];           // Array to hold the weak bit positions (max 10)
    int numWeakBits;            // Number of weak bits provided
    int weakMaxHD;              // Max simultaneous flips for weak bits (default: 4)
    int autoHD;            // Auto-Hamming Distance limit (0 = Off, 1 = HD1, 2 = HD2)
    
    // SEP (Stratified Entropy Permutation) Mode
    bool useSEP;               // Enable SEP mode
    char centerString[256];    // Human-readable center string
    int sepMin;                // SEP mutation range min
    int sepMax;                // SEP mutation range max
    uint64_t sepRawTarget[4];  // 256-bit raw center string (physical layout)
} StringCrackConfig;

// Second level lookup
typedef struct {
  address_t sAddress;
  std::vector<addressl_t> lAddresses;
} LADDRESS;

class GPUEngine {

public:

  GPUEngine(int gpuId, uint32_t maxFound, int smMultiplier = 1024);
  ~GPUEngine();
  void FreeGPUEngine();
  void SetAddress(std::vector<address_t> addresses);
  void SetAddress(std::vector<LADDRESS> addresses,uint32_t totalAddress);
  bool SetKeys(Point *p);
  bool SetRandomJump(Point p);
  void SetSearchMode(int searchMode);
  void SetSearchType(int searchType);
  void SetPattern(const char *pattern);
  bool Launch(std::vector<ITEM> &addressFound, bool spinWait=false, uint64_t ks_start_lo=0, uint64_t step_thread_lo=0, uint32_t launch_idx=0, int upper_hd=0, int upper_abs_pop=0);
  int GetNbThread();
  int GetGroupSize();
  int GetStepSize();

  // StringCrack: Configure and launch the Bit Injection + Popcount kernel
  bool SetStringCrackConfig(Secp256K1* secp, const StringCrackConfig *config);
  bool LaunchOpenClaw(std::vector<ITEM> &addressFound, uint64_t batchOffsetLo, uint64_t batchOffsetHi, bool spinWait=false);

  // Asynchronous double-buffered StringCrack
  void LaunchOpenClawAsync(uint64_t batchOffsetLo, uint64_t batchOffsetHi);
  uint32_t SyncAndGetResult(int stepToSync, std::vector<ITEM> &addressFound);

  bool Check(Secp256K1 *secp);
  std::string deviceName;

  static void PrintCudaInfo();
  static void GenerateCode(Secp256K1 *secp, int size);
  static void PrecomputeStringCrackMasks(StringCrackConfig *config);
  static void ComputeBasePoint(Secp256K1 *secp, StringCrackConfig *config);
  static bool ComputeWindowTables(Secp256K1 *secp, StringCrackConfig *config);

private:

  bool callKernel(uint64_t ks_start_lo = 0, uint64_t step_thread_lo = 0, uint32_t launch_idx = 0, int upper_hd = 0, int upper_abs_pop = 0);
  bool callOpenClawKernel(uint64_t batchOffsetLo, uint64_t batchOffsetHi, uint32_t* d_out, cudaStream_t stream);
  static void ComputeIndex(std::vector<int> &s, int depth, int n);
  static void Browse(FILE *f,int depth, int max, int s);
  bool CheckHash(uint8_t *h, std::vector<ITEM>& found, int tid, int incr, int endo, int *ok);

  int nbThread;
  uint64_t *sub;
  uint64_t *d_dx;
  address_t *inputAddress;
  address_t *inputAddressPinned;
  uint32_t *inputAddressLookUp;
  uint32_t *inputAddressLookUpPinned;
  uint64_t *inputKey;
  uint64_t *inputKeyPinned;
  uint32_t *outputBuffer;
  uint32_t *outputBufferPinned;
  bool initialised;
  uint32_t searchMode;
  uint32_t searchType;
  bool littleEndian;
  bool lostWarning;
  bool rekey;
  uint32_t maxFound;
  uint32_t outputSize;
  std::string pattern;
  bool hasPattern;

  // StringCrack state
  bool stringCrackEnabled;
  StringCrackConfig scConfig;

  // Asynchronous double-buffered streams
  cudaStream_t streams[2];
  uint32_t* d_output[2];
  uint32_t* h_outputPinned[2];
public:
  int currentStep;
};

#endif // GPUENGINEH
