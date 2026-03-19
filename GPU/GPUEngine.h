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
#define ITEM_SIZE32_WARP 8  // For warp-packed kernel: walk_id(1) + lane(1) + step(1) + hash160(5)
#define _64K 65536

// Maximum number of locked bit positions for StringCrack Bit Injection
#define MAX_LOCKED_BITS 128

static const char *searchModes[] = {"Compressed","Uncompressed","Compressed or Uncompressed"};

typedef uint16_t address_t;
typedef uint32_t addressl_t;

typedef struct {
  uint64_t thId;
  uint32_t endo;
  uint32_t incr;
  bool mode;
  uint8_t* hash;
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
    
    // SEP (Stratified Entropy Permutation) Configuration
    bool useSEP;
    char centerString[256];
    uint64_t targetSeedLo;
    uint64_t targetSeedHi;
    uint64_t seedMaskLo;        // Mask for valid seed bits
    int sepMin;
    int sepMax;
    
    // SEP3: Radius mode — CPU Gosper + GPU batch hybrid
    bool useRadius;
    int radius;                     // Max Hamming distance from center
    
    // SEP7: Revolving Door EC Walker
    bool useRevDoor;
    bool useMitm;     // <-- ADD THIS LINE
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
  bool Launch(std::vector<ITEM> &addressFound,bool spinWait=false);
  int GetNbThread();
  int GetGroupSize();
  int GetStepSize();

  // StringCrack: Configure and launch the Bit Injection + Popcount kernel
  bool SetStringCrackConfig(Secp256K1* secp, const StringCrackConfig *config);
  bool LaunchOpenClaw(std::vector<ITEM> &addressFound, uint64_t batchOffsetLo, uint64_t batchOffsetHi, bool spinWait=false);

  // Asynchronous double-buffered StringCrack
  void LaunchOpenClawAsync(uint64_t batchOffsetLo, uint64_t batchOffsetHi);
  uint32_t SyncAndGetResult(int stepToSync, std::vector<ITEM> &addressFound);

  // SEP3: Radius mode — CPU Gosper + GPU batch (legacy)
  bool SetupRadiusBuffers();
  void LaunchRadiusBatchAsync(uint64_t* h_seedsLo, uint64_t* h_seedsHi, int count);
  uint32_t SyncRadiusBatch(int stepToSync, std::vector<ITEM> &addressFound);

  // SEP4: GPU-native Gosper — combinatorial unranking in registers
  void LaunchGosperAsync(int hamming_h, uint64_t batchOffset, uint64_t totalCombs);
  uint32_t SyncGosperBatch(int stepToSync, std::vector<ITEM> &addressFound);

  // SEP7: Revolving Door Walker with Thread-Local Batch Inversion
  bool ComputeGfreeTables(Secp256K1* secp, StringCrackConfig* config);
  void LaunchGosperWalkAsync(int hamming_h, uint64_t base_rank_offset,
                             uint64_t totalCombs, int chunk_size, int numWalks);

  // SEP7: Revolving Door EC Walker
  bool ComputeDTable(Secp256K1* secp, StringCrackConfig* config);
  void LaunchRevDoorAsync(int hamming_h, uint64_t base_pos,
                          uint64_t totalCombs, int chunk_size, int numWalks);
  uint32_t SyncRevDoorBatch(int stepToSync, std::vector<ITEM> &addressFound);

  // SEP7: Coset Revolving Door
  void UploadQiArray(uint64_t* h_Qi, uint64_t size);
  void LaunchRevDoorAsync(int L_bits, int k2, int B_top, int k1, uint64_t base_pos, uint64_t totalCombs, int chunk_size, int numBlocks, int qi_chunks, uint64_t W);

  // SEP7: Warp-Packed Coset Revolving Door (32 <= W < 128)
  void LaunchWarpPackedRevDoorAsync(
      int L_bits, int k2, int B_top, int k1,
      uint64_t base_pos, uint64_t totalCombs,
      int chunk_size, int numBlocks, int qi_batches);
  uint32_t SyncWarpPackedRevDoorBatch(int stepToSync, std::vector<ITEM> &addressFound);

  // SEP7: Coset-Aware Gosper Walk Kernel
  void LaunchCosetGosperWalkAsync(int L_bits, int k2, int B_top, int k1, uint64_t base_pos, uint64_t L_totalCombs, int chunk_size, int total_walks, int W);

  // MITM VRAM Engine (Phase 1)
  bool BuildMITMTables(Secp256K1* secp, StringCrackConfig* config, 
                       int L_baby, int k_baby, int L_giant, int k_giant);
  void ShiftBabyTable(uint64_t bX[4], uint64_t bY[4], uint64_t bZ[4], uint64_t baby_size);
  void LaunchMITMChunkAsync(uint32_t qi_idx, uint64_t baby_size, uint64_t giant_size, uint64_t offset, uint64_t blocks, int s);
  uint32_t SyncMITMBatch(int s, std::vector<ITEM>& found);

  // MITM God Matrix pointers
  uint64_t* d_Qi_points_X;
  uint64_t* d_Qi_points_Y;

  // V2 God Matrix Launchers
  void BuildQiPoints(
      uint64_t lx0, uint64_t lx1, uint64_t lx2, uint64_t lx3,
      uint64_t ly0, uint64_t ly1, uint64_t ly2, uint64_t ly3,
      int L_bits, int B_top, int k1, uint64_t W);

  void LaunchMITMGodMatrixAsync(
      uint64_t baby_size, uint64_t giant_size, 
      int L_bits, int B_top, int k1,
      uint64_t qi_start, uint64_t qi_count, 
      uint64_t lx0, uint64_t lx1, uint64_t lx2, uint64_t lx3,
      uint64_t ly0, uint64_t ly1, uint64_t ly2, uint64_t ly3, int s);

  bool Check(Secp256K1 *secp);
  std::string deviceName;

  static void PrintCudaInfo();
  static void GenerateCode(Secp256K1 *secp, int size);
  static void PrecomputeStringCrackMasks(StringCrackConfig *config);
  static void ComputeBasePoint(Secp256K1 *secp, StringCrackConfig *config);
  static bool ComputeWindowTables(Secp256K1 *secp, StringCrackConfig *config);

private:

  bool callKernel();
  bool callOpenClawKernel(uint64_t batchOffsetLo, uint64_t batchOffsetHi, uint32_t* d_out, cudaStream_t stream);
  static void ComputeIndex(std::vector<int> &s, int depth, int n);
  static void Browse(FILE *f,int depth, int max, int s);
  bool CheckHash(uint8_t *h, std::vector<ITEM>& found, int tid, int incr, int endo, int *ok);

  int smCount;   // SM count, stored from constructor for SEP7 grid sizing

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

  // SEP3: Radius mode GPU buffers (double-buffered seed arrays)
  uint64_t* d_radiusSeedsLo[2];
  uint64_t* d_radiusSeedsHi[2];
  uint64_t* h_radiusSeedsLo[2];    // Pinned host staging buffers
  uint64_t* h_radiusSeedsHi[2];
  int* d_radiusCount[2];            // Per-stream seed count on device
  bool radiusBuffersReady;

  // SEP7: Warp-Packed Coset state
  int qi_batches_last;  // Stored for match reconstruction
  uint64_t* d_Qi_buffers[2];

  // ==========================================
  // MITM VRAM ENGINE POINTERS (Phase 1)
  // ==========================================
  uint64_t* d_mitm_baby_X;
  uint64_t* d_mitm_baby_Y;
  uint64_t* d_mitm_giant_X;
  uint64_t* d_mitm_giant_Y;
  uint64_t* d_mitm_baby_shifted_X;
  uint64_t* d_mitm_baby_shifted_Y;
  uint64_t* d_mitm_Gfree_X;
  uint64_t* d_mitm_Gfree_Y;

public:
  int currentStep;
};

#endif // GPUENGINEH
