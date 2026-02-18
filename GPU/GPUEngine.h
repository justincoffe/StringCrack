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
} StringCrackConfig;

// Second level lookup
typedef struct {
  address_t sAddress;
  std::vector<addressl_t> lAddresses;
} LADDRESS;

class GPUEngine {

public:

  GPUEngine(int gpuId, uint32_t maxFound);
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
  bool SetStringCrackConfig(const StringCrackConfig *config);
  bool LaunchOpenClaw(std::vector<ITEM> &addressFound, uint64_t batchOffset, bool spinWait=false);

  bool Check(Secp256K1 *secp);
  std::string deviceName;

  static void PrintCudaInfo();
  static void GenerateCode(Secp256K1 *secp, int size);
  static void PrecomputeStringCrackMasks(StringCrackConfig *config);

private:

  bool callKernel();
  bool callOpenClawKernel(uint64_t batchOffset);
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
};

#endif // GPUENGINEH
