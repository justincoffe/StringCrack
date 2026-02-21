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

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "../SECP256k1.h"
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"
#include "../Vanity.h"

#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUWildcard.h"
#include "GPUCompute.h"
#include <iostream>

#include <omp.h>

int _ConvertSMVer2Cores(int major, int minor) {

    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86,  128},
        {0x89,  128},
        {-1, -1} };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    return 0;

}



#define GRP_SIZE 1024
#define STEP_SIZE 128

__global__ void comp_keys(address_t* sAddress, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {


    uint64_t* startx = keys + (blockIdx.x * blockDim.x) * 8;
    uint64_t* starty = keys + (blockIdx.x * blockDim.x) * 8 + 4 * blockDim.x;


    uint64_t dx[4];  
    uint64_t px[4];
    uint64_t py[4];
    uint64_t dy[4];
    uint64_t sxn[4];
    uint64_t syn[4];
    uint64_t sx[4];
    uint64_t sy[4];
    uint64_t sx_gx[4];
    uint8_t odd_py;
    uint32_t h[5];
    uint64_t inverse[5];

    uint64_t subp[GRP_SIZE/2][4];
    

    __syncthreads();
    Load256A(sx, startx);
    Load256A(sy, starty);


    uint32_t i;

    // Check starting point
    odd_py = sy[0] & 1;
    _GetHash160Comp(sx, odd_py, (uint8_t*)h);
    CheckPoint(h, GRP_SIZE / 2, sAddress, lookup32, out);
    __syncthreads();

    ModSub256(sxn, _2Gnx, sx);
    Load256(subp[GRP_SIZE / 2 - 1], sxn);
    for (i = GRP_SIZE / 2 - 1; i > 0; i--) {
        ModSub256(syn, Gx[i], sx);
        _ModMult(sxn, syn);
        Load256(subp[i - 1], sxn);
    }

    ModSub256(inverse, Gx[0], sx);
    _ModMult(inverse, sxn);


    inverse[4] = 0;
    _ModInv(inverse);

    __syncthreads();
    
    ModNeg256(syn, sy);
    ModNeg256(sxn, sx);

    for (i = 0; i < GRP_SIZE / 2 - 1; i++) {

        __syncthreads();
        ModSub256(sx_gx, Gx[i], sxn);

        _ModMult(dx, subp[i], inverse);

        //////////////////

        ModSub256(dy, Gy[i], sy);
        _ModMult(dy, dx);
        _ModSqr(px, dy);
        ModSub256(px, sx_gx);

        ModSub256(py, sx, px);
        _ModMult(py, dy);
        ModSub256isOdd(py, sy, &odd_py);

        _GetHash160Comp(px, odd_py, (uint8_t*)h);
        CheckPoint(h, GRP_SIZE / 2 + (i + 1), sAddress, lookup32, out);

        //////////////////

        __syncthreads();

        ModSub256(dy, syn, Gy[i]);
        _ModMult(dy, dx);
        _ModSqr(px, dy);
        ModSub256(px, sx_gx);

        ModSub256(py, px, sx);
        _ModMult(py, dy);
        ModSub256isOdd(syn, py, &odd_py);

        _GetHash160Comp(px, odd_py, (uint8_t*)h);
        CheckPoint(h, GRP_SIZE / 2 - (i + 1), sAddress, lookup32, out);

        //////////////////

        ModSub256(dx, Gx[i], sx);
        _ModMult(inverse, dx);

    }

    __syncthreads();

    _ModMult(dx, subp[i], inverse);

    ModSub256(dy, syn, Gy[i]);
    _ModMult(dy, dx);
    _ModSqr(px, dy);
    ModSub256(px, sx);
    ModSub256(px, Gx[i]);

    ModSub256(py, px, sx);
    _ModMult(py, dy);
    ModSub256isOdd(syn, py, &odd_py);

    _GetHash160Comp(px, odd_py, (uint8_t*)h);
    CheckPoint(h, 0, sAddress, lookup32, out);

    //////////////////

    __syncthreads();

    ModSub256(dy, _2Gny, sy);
    ModSub256(dx, Gx[i], sx);
    _ModMult(inverse, dx);

    _ModMult(dy, inverse);
    _ModSqr(px, dy);
    ModSub256(px, sx);
    ModSub256(px, _2Gnx);

    ModSub256(py, _2Gnx, px);
    _ModMult(py, dy);
    ModSub256(py, _2Gny);               

    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);


}


// ---------------------------------------------------------------------------------------
int NB_TRHEAD_PER_GROUP;

using namespace std;

int g_gpuId;
std::string globalGPUname;



GPUEngine::GPUEngine(int gpuId, uint32_t maxFound) {

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpuId);

    NB_TRHEAD_PER_GROUP = 256;                                          //////////////////  GRID SIZE ////////////////
    int nbThreadGroup = deviceProp.multiProcessorCount * 128;

    // --- COMMENT THIS ENTIRE BLOCK OUT ---
    /*
    if (!randomMode) {
        uint64_t powerOfTwo = 1;
        while (powerOfTwo <= nbThreadGroup) {  //  GET THE CLOSEST POWER OF 2
            powerOfTwo <<= 1;
        }

        powerOfTwo >>= 1;
        nbThreadGroup = powerOfTwo;
    }
    */
    // -------------------------------------

    
    g_gpuId = gpuId;

    // Initialise CUDA
    this->rekey = rekey;
    initialised = false;
    cudaError_t err;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
        return;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("GPUEngine: There are no available device(s) that support CUDA\n");
        return;
    }

    err = cudaSetDevice(gpuId);
    if (err != cudaSuccess) {
        printf("GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

   

    this->nbThread = nbThreadGroup * NB_TRHEAD_PER_GROUP;//////////////////////////////////////////////////////////////////
    this->maxFound = maxFound;
    this->outputSize = (maxFound * ITEM_SIZE + 4);

    char tmp[512];
    sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
    gpuId,deviceProp.name,deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
    nbThread / NB_TRHEAD_PER_GROUP,
    NB_TRHEAD_PER_GROUP);

    deviceName = std::string(tmp);

    globalGPUname = deviceProp.name;

    // Prefer L1 (We do not use __shared__ at all)
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) {
        printf("GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

    //size_t stackSize = 49152;
    //err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    //if (err != cudaSuccess) {
    //  printf("GPUEngine: %s\n", cudaGetErrorString(err));
    //  return;
    //}

    /*
    size_t heapSize = ;
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      exit(0);
    }

    size_t size;
    cudaDeviceGetLimit(&size, cudaLimitStackSize);
    printf("Stack Size %lld\n", size);
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    printf("Heap Size %lld\n", size);
    */

    // Allocate memory
    err = cudaMalloc((void**)&inputAddress, _64K * 2);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate address memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&inputAddressPinned, _64K * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate address pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&inputKey, nbThread * 32 * 2);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&outputBuffer, outputSize);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&outputBufferPinned, outputSize, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }

    searchMode = SEARCH_COMPRESSED;
    searchType = P2PKH;
    initialised = true;
    pattern = "";
    hasPattern = false;
    inputAddressLookUp = NULL;
    stringCrackEnabled = false;
    memset(&scConfig, 0, sizeof(StringCrackConfig));

}

GPUEngine::~GPUEngine() {

    cudaFree(inputKey);
    cudaFree(inputAddress);
    if (inputAddressLookUp) cudaFree(inputAddressLookUp);
    cudaFreeHost(outputBufferPinned);
    cudaFree(outputBuffer);
}



void GPUEngine::PrintCudaInfo() {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);


    for (int i = 0;i < deviceCount;i++) {

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("%d , %s", i, deviceProp.name);

    }

}



int GPUEngine::GetNbThread() {
    return nbThread;
}

void GPUEngine::SetSearchMode(int searchMode) {
    this->searchMode = searchMode;
}

void GPUEngine::SetSearchType(int searchType) {
    this->searchType = searchType;
}





void GPUEngine::SetAddress(std::vector<address_t> addresses) {

    memset(inputAddressPinned, 0, _64K * 2);
    for (int i = 0;i < (int)addresses.size();i++)
        inputAddressPinned[addresses[i]] = 1;

    // Fill device memory
    cudaMemcpy(inputAddress, inputAddressPinned, _64K * 2, cudaMemcpyHostToDevice);

    // We do not need the input pinned memory anymore
    cudaFreeHost(inputAddressPinned);
    inputAddressPinned = NULL;
    lostWarning = false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetAddress: %s\n", cudaGetErrorString(err));
    }

}

void GPUEngine::SetPattern(const char* pattern) {

    strcpy((char*)inputAddressPinned, pattern);

    // Fill device memory
    cudaMemcpy(inputAddress, inputAddressPinned, _64K * 2, cudaMemcpyHostToDevice);

    // We do not need the input pinned memory anymore
    cudaFreeHost(inputAddressPinned);
    inputAddressPinned = NULL;
    lostWarning = false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetPattern: %s\n", cudaGetErrorString(err));
    }

    hasPattern = true;

}



void GPUEngine::SetAddress(std::vector<LADDRESS> addresses, uint32_t totalAddress) {

    // Allocate memory for the second level of lookup tables
    cudaError_t err = cudaMalloc((void**)&inputAddressLookUp, (_64K + totalAddress) * 4);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate address lookup memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&inputAddressLookUpPinned, (_64K + totalAddress) * 4, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate address lookup pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }

    uint32_t offset = _64K;
    memset(inputAddressPinned, 0, _64K * 2);
    memset(inputAddressLookUpPinned, 0, _64K * 4);
    for (int i = 0; i < (int)addresses.size(); i++) {
        int nbLAddress = (int)addresses[i].lAddresses.size();
        inputAddressPinned[addresses[i].sAddress] = (uint16_t)nbLAddress;
        inputAddressLookUpPinned[addresses[i].sAddress] = offset;
        for (int j = 0; j < nbLAddress; j++) {
            inputAddressLookUpPinned[offset++] = addresses[i].lAddresses[j];
        }
    }

    if (offset != (_64K + totalAddress)) {
        printf("GPUEngine: Wrong totalAddress %d!=%d!\n", offset - _64K, totalAddress);
        return;
    }

    // Fill device memory
    cudaMemcpy(inputAddress, inputAddressPinned, _64K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(inputAddressLookUp, inputAddressLookUpPinned, (_64K + totalAddress) * 4, cudaMemcpyHostToDevice);


    // We do not need the input pinned memory anymore
    cudaFreeHost(inputAddressPinned);
    inputAddressPinned = NULL;
    cudaFreeHost(inputAddressLookUpPinned);
    inputAddressLookUpPinned = NULL;
    lostWarning = false;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetAddress (large): %s\n", cudaGetErrorString(err));
    }

}

int GPUEngine::GetStepSize() {

    return STEP_SIZE;

}

int GPUEngine::GetGroupSize() {

    return GRP_SIZE;

}


bool GPUEngine::callKernel() {

   
    // Reset nbFound
    cudaMemset(outputBuffer, 0, 4);

    comp_keys << < nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >> >
        (inputAddress, inputAddressLookUp, inputKey, outputBuffer);




    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
        return false;
    }

    //cudaFree(d_dx);


    return true;

}


bool GPUEngine::SetKeys(Point* p) {

    // Sets the starting keys for each thread
    // p must contains nbThread public keys

    for (int i = 0; i < nbThread; i += NB_TRHEAD_PER_GROUP) {
        for (int j = 0; j < NB_TRHEAD_PER_GROUP; j++) {

            inputKeyPinned[8 * i + j + 0 * NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[0];
            inputKeyPinned[8 * i + j + 1 * NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[1];
            inputKeyPinned[8 * i + j + 2 * NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[2];
            inputKeyPinned[8 * i + j + 3 * NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[3];

            inputKeyPinned[8 * i + j + 4 * NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[0];
            inputKeyPinned[8 * i + j + 5 * NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[1];
            inputKeyPinned[8 * i + j + 6 * NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[2];
            inputKeyPinned[8 * i + j + 7 * NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[3];

        }
    }

    // Fill device memory

    cudaMemcpy(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice);
    // We do not need the input pinned memory anymore
    cudaFreeHost(inputKeyPinned);
    inputKeyPinned = NULL;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
    }

    return callKernel();
    //return true;

}


uint64_t new_2Gnx[4];
uint64_t new_2Gny[4];

bool GPUEngine::SetRandomJump(Point p) {


    new_2Gnx[0] = p.x.bits64[0];
    new_2Gnx[1] = p.x.bits64[1];
    new_2Gnx[2] = p.x.bits64[2];
    new_2Gnx[3] = p.x.bits64[3];

    new_2Gny[0] = p.y.bits64[0];
    new_2Gny[1] = p.y.bits64[1];
    new_2Gny[2] = p.y.bits64[2];
    new_2Gny[3] = p.y.bits64[3];

    cudaError_t err;

    err = cudaMemcpyToSymbol(_2Gnx, new_2Gnx, sizeof(new_2Gnx));
    if (err != cudaSuccess) {
        printf("GPUEngine: SetRandomJump _2Gnx: %s\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMemcpyToSymbol(_2Gny, new_2Gny, sizeof(new_2Gny));
    if (err != cudaSuccess) {
        printf("GPUEngine: SetRandomJump _2Gny: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
    //return callKernel();

}



bool GPUEngine::Launch(std::vector<ITEM>& addressFound, bool spinWait) {

    addressFound.clear();
    

    // Get the result


    if(spinWait) {

      cudaMemcpy(outputBufferPinned, outputBuffer, outputSize, cudaMemcpyDeviceToHost);

    } else {

      // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
      cudaEvent_t evt;
      cudaEventCreate(&evt);

      //cudaMemcpy(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost, 0);

      cudaEventRecord(evt, 0);
      while (cudaEventQuery(evt) == cudaErrorNotReady) {
        // Sleep 1 ms to free the CPU
        Timer::SleepMillis(1);
      }
      cudaEventDestroy(evt);

    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
      return false;
    }

    // Look for address found
    uint32_t nbFound = outputBufferPinned[0];

    if (nbFound > maxFound) {
      // address has been lost
      if (!lostWarning) {
        printf("\nWarning, %d items lost\nHint: Search with less addresses/prefixes or increase maxFound (-m) using multiple of 65536\n", (nbFound - maxFound));
        lostWarning = true;
      }
      nbFound = maxFound;
    }

    // When can perform a standard copy, the kernel is eneded
    cudaMemcpy(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE32 + 1);
        ITEM it;
        it.thId = itemPtr[0];
        int16_t* ptr = (int16_t*)&(itemPtr[1]);
        it.endo = ptr[0] & 0x7FFF;
        it.mode = (ptr[0] & 0x8000) != 0;
        it.incr = ptr[1];
        it.hash = (uint8_t*)(itemPtr + 2);
        addressFound.push_back(it);
    }

    return callKernel();

}

std::string toHex(unsigned char* data, int length) {

    string ret;
    char tmp[3];
    for (int i = 0; i < length; i++) {
        if (i && i % 4 == 0) ret.append(" ");
        sprintf(tmp, "%02hhX", (int)data[i]);
        ret.append(tmp);
    }
    return ret;

}



void GPUEngine::FreeGPUEngine() {  //free gpu for Pause function

    // Ensure all operations have completed before freeing memory
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(inputKey);
    cudaFree(inputAddress);
    if (inputAddressLookUp) {
        cudaFree(inputAddressLookUp);  // Free the lookup table memory if allocated
    }
    cudaFree(outputBuffer);

    // Free pinned memory
    cudaFreeHost(inputAddressPinned);
    cudaFreeHost(inputKeyPinned);
    cudaFreeHost(outputBufferPinned);

    // Reset the pointers to prevent dangling references
    inputAddressPinned = NULL;
    inputKeyPinned = NULL;
    outputBufferPinned = NULL;
    inputAddressLookUpPinned = NULL;
    inputAddressLookUp = NULL;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Error freeing memory: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceReset();

}


bool GPUEngine::CheckHash(uint8_t* h, vector<ITEM>& found, int tid, int incr, int endo, int* nbOK) {

    return true;
}

bool GPUEngine::Check(Secp256K1* secp) {

    return true;
}

// =====================================================================================
// StringCrack: Direct Seed Iteration - Zero expand_bits, Zero Warp Divergence
// Precomputed base point from locked bits computed on CPU
// =====================================================================================

__device__ __constant__ uint64_t d_lockMask[4];
__device__ __constant__ uint64_t d_lockVals[4];
__device__ __constant__ int      d_freeBitPos[256];
__device__ __constant__ int      d_numFreeBits;
__device__ __constant__ int      d_popcountMin;
__device__ __constant__ int      d_popcountMax;
__device__ __constant__ uint64_t d_batchOffsetLo;
__device__ __constant__ uint64_t d_batchOffsetHi;

// New Constant Memory for Direct Seed Iteration
__device__ __constant__ uint64_t d_basePointX[4];
__device__ __constant__ uint64_t d_basePointY[4];

// 20 windows (covers up to 80 free bits), 16 possible values per 4-bit window, 4 uint64_t limbs per coordinate
__device__ __constant__ uint64_t d_window_GX[20][16][4];
__device__ __constant__ uint64_t d_window_GY[20][16][4];
__device__ __constant__ int      d_lockedPopcount;
__device__ __constant__ int      d_stepSize;

// expand_bits: Map continuous seed into sparse 256-bit key via Bit Injection
// Now supports 128-bit seed (seed_lo + seed_hi)
__device__ __forceinline__ void expand_bits(uint64_t seed_lo, uint64_t seed_hi, uint64_t key[4]) {
    key[0] = d_lockVals[0];
    key[1] = d_lockVals[1];
    key[2] = d_lockVals[2];
    key[3] = d_lockVals[3];
    
    // Combine 128-bit seed into single value for bit iteration
    // Process seed_lo first (lower 64 bits), then seed_hi (upper bits)
    for (int i = 0; i < 64 && i < d_numFreeBits; i++) {
        if (seed_lo == 0ULL) break;
        int bitVal = (int)(seed_lo & 1ULL);
        seed_lo >>= 1;
        if (bitVal) {
            int pos = d_freeBitPos[i];
            int limb = pos >> 6;
            int bit  = pos & 63;
            key[limb] |= (1ULL << bit);
        }
    }
    // Continue with upper 64 bits if needed
    for (int i = 64; i < d_numFreeBits; i++) {
        if (seed_hi == 0ULL) break;
        int bitVal = (int)(seed_hi & 1ULL);
        seed_hi >>= 1;
        if (bitVal) {
            int pos = d_freeBitPos[i];
            int limb = pos >> 6;
            int bit  = pos & 63;
            key[limb] |= (1ULL << bit);
        }
    }
}

// popcount256: Count set bits in 256-bit key
__device__ __forceinline__ int popcount256(const uint64_t key[4]) {
    return __popcll(key[0]) + __popcll(key[1]) + __popcll(key[2]) + __popcll(key[3]);
}

// G_POW2 table size (currently 71 entries in GPUGroup.h, user will expand)
#define G_POW2_TABLE_SIZE 71

// Mixed Jacobian-Affine Addition (Pure Cohen/Miyaji Formula)
// Adds affine point Q(x2, y2) to Jacobian point P(X1, Y1, Z1)
// Result in Jacobian coordinates: (X3, Y3, Z3)
// This avoids modular inversions - only needs multiplications!
__device__ void jacobian_add_affine(uint64_t X1[4], uint64_t Y1[4], uint64_t Z1[4],
                                     uint64_t x2[4], uint64_t y2[4],
                                     uint64_t X3[4], uint64_t Y3[4], uint64_t Z3[4]) {
    
    // Z1Z1 = Z1^2
    uint64_t Z1Z1[4];
    _ModSqr(Z1Z1, Z1);
    
    // U2 = x2 * Z1^2
    uint64_t U2[4];
    _ModMult(U2, Z1Z1, x2);
    
    // Z1_cubed = Z1^3
    uint64_t Z1_cubed[4];
    _ModMult(Z1_cubed, Z1Z1, Z1);
    
    // S2 = y2 * Z1^3
    uint64_t S2[4];
    _ModMult(S2, Z1_cubed, y2);
    
    // H = U2 - X1
    uint64_t H[4];
    ModSub256(H, U2, X1);
    
    // R = S2 - Y1
    uint64_t R[4];
    ModSub256(R, S2, Y1);
    
    // HH = H^2
    uint64_t HH[4];
    _ModSqr(HH, H);
    
    // HHH = H^3
    uint64_t HHH[4];
    _ModMult(HHH, HH, H);
    
    // U1HH = X1 * H^2
    uint64_t U1HH[4];
    _ModMult(U1HH, X1, HH);
    
    // R_sq = R^2
    uint64_t R_sq[4];
    _ModSqr(R_sq, R);
    
    // Calculate 2 * U1HH using ModNeg256 + ModSub256 (reverted from ModAdd256 due to math bug)
    uint64_t neg_U1HH[4], two_U1HH[4];
    ModNeg256(neg_U1HH, U1HH);
    ModSub256(two_U1HH, U1HH, neg_U1HH);  // A - (-A) = 2*A
    
    // X3 = R^2 - H^3 - 2*U1HH
    uint64_t new_X[4];
    ModSub256(new_X, R_sq, HHH);
    ModSub256(new_X, new_X, two_U1HH); 
    
    // Y3 = R * (U1HH - X3) - Y1 * H^3
    uint64_t U1HH_minus_X3[4];
    ModSub256(U1HH_minus_X3, U1HH, new_X);
    
    uint64_t R_times_diff[4];
    _ModMult(R_times_diff, R, U1HH_minus_X3);
    
    uint64_t Y1_times_HHH[4];
    _ModMult(Y1_times_HHH, Y1, HHH);
    
    uint64_t new_Y[4];
    ModSub256(new_Y, R_times_diff, Y1_times_HHH); 
    
    // Z3 = Z1 * H
    uint64_t new_Z[4];
    _ModMult(new_Z, Z1, H); 
    
    // Commit the new coordinates
    Load256(X3, new_X);
    Load256(Y3, new_Y);
    Load256(Z3, new_Z);
}

// Convert Jacobian (X, Y, Z) to Affine (x, y)
__device__ void jacobian_to_affine(uint64_t X[4], uint64_t Y[4], uint64_t Z[4], uint64_t x[4], uint64_t y[4]) {
    // Check if Z == 0 (point at infinity)
    if (Z[0] == 0 && Z[1] == 0 && Z[2] == 0 && Z[3] == 0) {
        x[0] = x[1] = x[2] = x[3] = 0;
        y[0] = y[1] = y[2] = y[3] = 0;
        return;
    }
    
    // Compute Z^-1
    uint64_t Z_inv[5];
    Load256(Z_inv, Z);
    Z_inv[4] = 0;
    _ModInv(Z_inv);
    
    // Compute Z^-2
    uint64_t Z_inv_sq[4];
    _ModSqr(Z_inv_sq, Z_inv);
    
    // x = X * Z^-2
    _ModMult(x, Z_inv_sq, X);
    
    // Compute Z^-3 = Z^-2 * Z^-1
    uint64_t Z_inv_cb[4];
    _ModMult(Z_inv_cb, Z_inv_sq, Z_inv);
    
    // y = Y * Z^-3
    _ModMult(y, Z_inv_cb, Y);
}

// ec_point_mult_pow2: Compute key * G using precomputed G_POW2 table
// Uses Jacobian coordinates for fast addition (no modular inversions inside loop)
__device__ void ec_point_mult_pow2(const uint64_t key[4], uint64_t px[4], uint64_t py[4]) {
    bool pointSet = false;
    
    // Jacobian accumulator: (X, Y, Z) where Z=1 initially (affine point)
    uint64_t accX[4] = {0}, accY[4] = {0}, accZ[4] = {1,0,0,0};

    for (int i = 0; i < G_POW2_TABLE_SIZE; i++) {
        int limb = i >> 6;
        int bit  = i & 63;
        if ((key[limb] >> bit) & 1ULL) {
            // Load affine point from G_POW2 table
            uint64_t gx[4], gy[4];
            Load256(gx, (uint64_t*)G_POW2_X_EXTENDED[i]);
            Load256(gy, (uint64_t*)G_POW2_Y_EXTENDED[i]);
            
            if (!pointSet) {
                // First point: initialize Jacobian accumulator with Z=1
                Load256(accX, gx);
                Load256(accY, gy);
                accZ[0] = 1; accZ[1] = accZ[2] = accZ[3] = 0;  // Z=1
                pointSet = true;
            } else {
                // Add affine point to Jacobian accumulator using mixed addition
                uint64_t newX[4], newY[4], newZ[4];
                jacobian_add_affine(accX, accY, accZ, gx, gy, newX, newY, newZ);
                Load256(accX, newX);
                Load256(accY, newY);
                Load256(accZ, newZ);
            }
        }
    }

    if (pointSet) {
        // Convert Jacobian to affine coordinates for hashing
        jacobian_to_affine(accX, accY, accZ, px, py);
    } else {
        px[0] = px[1] = px[2] = px[3] = 0;
        py[0] = py[1] = py[2] = py[3] = 0;
    }
}

// --- 256-bit Warp Shuffle Helper Functions ---
__device__ __forceinline__ void shfl_sync_256(uint32_t mask, const uint64_t val[4], int srcLane, uint64_t out[4]) {
    out[0] = __shfl_sync(mask, val[0], srcLane);
    out[1] = __shfl_sync(mask, val[1], srcLane);
    out[2] = __shfl_sync(mask, val[2], srcLane);
    out[3] = __shfl_sync(mask, val[3], srcLane);
}

__device__ __forceinline__ void shfl_up_sync_256(uint32_t mask, const uint64_t val[4], unsigned int delta, uint64_t out[4]) {
    out[0] = __shfl_up_sync(mask, val[0], delta);
    out[1] = __shfl_up_sync(mask, val[1], delta);
    out[2] = __shfl_up_sync(mask, val[2], delta);
    out[3] = __shfl_up_sync(mask, val[3], delta);
}

__device__ __forceinline__ void shfl_down_sync_256(uint32_t mask, const uint64_t val[4], unsigned int delta, uint64_t out[4]) {
    out[0] = __shfl_down_sync(mask, val[0], delta);
    out[1] = __shfl_down_sync(mask, val[1], delta);
    out[2] = __shfl_down_sync(mask, val[2], delta);
    out[3] = __shfl_down_sync(mask, val[3], delta);
}

// The Ultimate OpenClaw Kernel (Grid-Stride Loop Optimized)
// Optimized: 5-bit windows, warp shuffle batch inversion, batch offsets as kernel args
__global__ void comp_keys_openclaw(
    address_t* sAddress, 
    uint32_t* lookup32, 
    uint32_t* out,
    uint64_t batchOffsetLo,  // Passed as kernel argument (faster than cudaMemcpyToSymbol)
    uint64_t batchOffsetHi) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    // Load base points ONCE per thread to save register/cache bandwidth
    uint64_t baseAccX[4], baseAccY[4];
    Load256(baseAccX, d_basePointX);
    Load256(baseAccY, d_basePointY);
    
    // Use kernel arguments directly instead of device globals
    uint64_t d_batchOffsetLo = batchOffsetLo;
    uint64_t d_batchOffsetHi = batchOffsetHi;

    // Process d_stepSize seeds per thread
    for (uint32_t step = 0; step < d_stepSize; step++) {
        
        // Calculate the absolute offset for this specific step
        uint64_t current_offset = (uint64_t)tid + ((uint64_t)step * (uint64_t)stride);
        
        uint64_t seed_lo = d_batchOffsetLo;
        uint64_t seed_hi = d_batchOffsetHi;
        
        asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(seed_lo) : "l"(current_offset));
        asm volatile ("addc.u64 %0, %0, 0;" : "+l"(seed_hi));

        // 1. O(1) Popcount Filtering (Masked, NO CONTINUE)
        int pc = __popcll(seed_lo) + __popcll(seed_hi) + d_lockedPopcount;
        bool valid = (pc >= d_popcountMin && pc <= d_popcountMax);

        // 2. Initialize Jacobian Accumulator
        uint64_t accX[4], accY[4], accZ[4];
        Load256(accX, baseAccX);
        Load256(accY, baseAccY);
        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0; // Z is 1 (Neutral for multiplication)

        if (valid) {
            // 3. Windowed Seed Iteration (5 bits at a time)
            int numWindows = (d_numFreeBits + 4) / 5;  // 5 bits per window
            
            uint64_t seed = seed_lo;
            int w = 0;
            #pragma unroll 1
            for (; w < 14 && w < numWindows; w++) {
                uint32_t nibble = seed & 0x1F;  // 5 bits (0-31)
                if (nibble > 0) {
                    uint64_t curGX[4], curGY[4];
                    Load256(curGX, (uint64_t*)d_window_GX[w][nibble]);
                    Load256(curGY, (uint64_t*)d_window_GY[w][nibble]);

                    uint64_t newX[4], newY[4], newZ[4];
                    jacobian_add_affine(accX, accY, accZ, curGX, curGY, newX, newY, newZ);
                    Load256(accX, newX); Load256(accY, newY); Load256(accZ, newZ);
                }
                seed >>= 5;  // Shift 5 bits
            }

            seed = seed_hi;
            #pragma unroll 1
            for (; w < numWindows; w++) {
                uint32_t nibble = seed & 0x1F;  // 5 bits (0-31)
                if (nibble > 0) {
                    uint64_t curGX[4], curGY[4];
                    Load256(curGX, (uint64_t*)d_window_GX[w][nibble]);
                    Load256(curGY, (uint64_t*)d_window_GY[w][nibble]);

                    uint64_t newX[4], newY[4], newZ[4];
                    jacobian_add_affine(accX, accY, accZ, curGX, curGY, newX, newY, newZ);
                    Load256(accX, newX); Load256(accY, newY); Load256(accZ, newZ);
                }
                seed >>= 5;  // Shift 5 bits
            }
            
            // Safety: If Z hit 0 (Point at Infinity), poison valid to prevent warp-wide div by zero
            if (accZ[0] == 0 && accZ[1] == 0 && accZ[2] == 0 && accZ[3] == 0) {
                valid = false;
                accZ[0] = 1; 
            }
        }

       // ====================================================================
        // 4. WARP-SHUFFLE MONTGOMERY BATCH INVERSION (O(1) Inversion per 32 keys)
        // ====================================================================
        uint32_t lane = threadIdx.x & 31;
        uint64_t neighbor[4];

        // A. Parallel Prefix Pass (Product of Z's before this thread)
        uint64_t P[4]; Load256(P, accZ);
        #pragma unroll
        for(int i = 1; i < 32; i *= 2) {
            shfl_up_sync_256(0xFFFFFFFF, P, i, neighbor);
            if (lane >= i) {
                uint64_t tmp[4];
                _ModMult(tmp, P, neighbor);
                Load256(P, tmp);
            }
        }
        uint64_t P_prev[4];
        shfl_up_sync_256(0xFFFFFFFF, P, 1, P_prev);
        if (lane == 0) {
            P_prev[0] = 1; P_prev[1] = 0; P_prev[2] = 0; P_prev[3] = 0;
        }

        // B. Parallel Suffix Pass (Product of Z's after this thread)
        uint64_t S[4]; Load256(S, accZ);
        #pragma unroll
        for(int i = 1; i < 32; i *= 2) {
            shfl_down_sync_256(0xFFFFFFFF, S, i, neighbor);
            if (lane < 32 - i) {
                uint64_t tmp[4];
                _ModMult(tmp, S, neighbor);
                Load256(S, tmp);
            }
        }
        uint64_t S_next[4];
        shfl_down_sync_256(0xFFFFFFFF, S, 1, S_next);
        if (lane == 31) {
            S_next[0] = 1; S_next[1] = 0; S_next[2] = 0; S_next[3] = 0;
        }

        // C. Total Warp Product (broadcast to all lanes so lane 31 has it)
        uint64_t Total[4];
        shfl_sync_256(0xFFFFFFFF, P, 31, Total);

        // D. Single _ModInv computed ONLY by lane 31
        uint64_t Shared_Inv[4];
        if (lane == 31) {
            uint64_t Inv5[5];
            Load256(Inv5, Total);
            Inv5[4] = 0; // 5th limb required as scratch space by VanitySearch ModInv
            _ModInv(Inv5);
            Load256(Shared_Inv, Inv5); // Extract the 4-limb result
        }

        // Broadcast the inverted result from lane 31 back to all lanes
        shfl_sync_256(0xFFFFFFFF, Shared_Inv, 31, Shared_Inv);

        // E. Calculate thread's individual Z_inv
        uint64_t Z_inv_tmp[4];
        _ModMult(Z_inv_tmp, Shared_Inv, P_prev);
        uint64_t Z_inv_final[4];
        _ModMult(Z_inv_final, Z_inv_tmp, S_next);

        // ====================================================================
        // 5. Affine Conversion and Hash (Masked)
        // ====================================================================
        if (valid) {
            uint64_t px[4], py[4];
            
            // Inline affine math: X = X * Z^-2, Y = Y * Z^-3
            uint64_t Z_inv_sq[4];
            _ModSqr(Z_inv_sq, Z_inv_final);
            _ModMult(px, Z_inv_sq, accX);

            uint64_t Z_inv_cb[4];
            _ModMult(Z_inv_cb, Z_inv_sq, Z_inv_final);
            _ModMult(py, Z_inv_cb, accY);

            uint8_t odd_py = (uint8_t)(py[0] & 1);
            uint32_t h[5];
            _GetHash160Comp(px, odd_py, (uint8_t*)h);
            CheckPoint(h, step, sAddress, lookup32, out);
        }
    }
}

// =====================================================================================
// Host-side StringCrack methods
// =====================================================================================

void GPUEngine::PrecomputeStringCrackMasks(StringCrackConfig *config) {
    for (int i = 0; i < 4; i++) { config->lockMask[i] = 0; config->lockVals[i] = 0; }

    for (int i = 0; i < config->numLockedBits; i++) {
        int pos = config->lockedBits[i].position;
        int val = config->lockedBits[i].value;
        int limb = pos >> 6;
        int bit  = pos & 63;
        config->lockMask[limb] |= (1ULL << bit);
        if (val) config->lockVals[limb] |= (1ULL << bit);
    }

    // Lock MSB of puzzle range to 1
    if (config->puzzleBits > 0 && config->puzzleBits <= 256) {
        int msbPos = config->puzzleBits - 1;
        int limb = msbPos >> 6;
        int bit  = msbPos & 63;
        if (!((config->lockMask[limb] >> bit) & 1)) {
            config->lockMask[limb] |= (1ULL << bit);
            config->lockVals[limb] |= (1ULL << bit);
        }
    }

    // Lock all bits above puzzleBits to 0
    if (config->puzzleBits > 0 && config->puzzleBits < 256) {
        for (int pos = config->puzzleBits; pos < 256; pos++) {
            int limb = pos >> 6;
            int bit  = pos & 63;
            config->lockMask[limb] |= (1ULL << bit);
        }
    }

    // Compute free bit positions
    config->numFreeBits = 0;
    for (int pos = 0; pos < 256; pos++) {
        int limb = pos >> 6;
        int bit  = pos & 63;
        if (!((config->lockMask[limb] >> bit) & 1)) {
            config->freeBitPositions[config->numFreeBits++] = pos;
        }
    }

    printf("[StringCrack] Locked bits: %d, Free bits: %d\n", config->numLockedBits, config->numFreeBits);
    printf("[StringCrack] Lock mask: %016llX %016llX %016llX %016llX\n",
           (unsigned long long)config->lockMask[3], (unsigned long long)config->lockMask[2],
           (unsigned long long)config->lockMask[1], (unsigned long long)config->lockMask[0]);
    printf("[StringCrack] Lock vals: %016llX %016llX %016llX %016llX\n",
           (unsigned long long)config->lockVals[3], (unsigned long long)config->lockVals[2],
           (unsigned long long)config->lockVals[1], (unsigned long long)config->lockVals[0]);
    printf("[StringCrack] Popcount target: %d-%d\n", config->popcountMin, config->popcountMax);
    printf("[StringCrack] Effective search space: 2^%d\n", config->numFreeBits);
    fflush(stdout);
}

bool GPUEngine::SetStringCrackConfig(const StringCrackConfig *config) {
    scConfig = *config;
    stringCrackEnabled = config->enabled;
    if (!stringCrackEnabled) return true;

    cudaError_t err;
    err = cudaMemcpyToSymbol(d_lockMask, config->lockMask, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("GPUEngine: d_lockMask: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_lockVals, config->lockVals, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("GPUEngine: d_lockVals: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_freeBitPos, config->freeBitPositions, sizeof(int) * 256);
    if (err != cudaSuccess) { printf("GPUEngine: d_freeBitPos: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_numFreeBits, &config->numFreeBits, sizeof(int));
    if (err != cudaSuccess) { printf("GPUEngine: d_numFreeBits: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_popcountMin, &config->popcountMin, sizeof(int));
    if (err != cudaSuccess) { printf("GPUEngine: d_popcountMin: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_popcountMax, &config->popcountMax, sizeof(int));
    if (err != cudaSuccess) { printf("GPUEngine: d_popcountMax: %s\n", cudaGetErrorString(err)); return false; }
    
    // Upload precomputed base point and free G table to GPU
    err = cudaMemcpyToSymbol(d_basePointX, config->basePointX, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("GPUEngine: d_basePointX: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_basePointY, config->basePointY, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("GPUEngine: d_basePointY: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_window_GX, config->window_GX, sizeof(uint64_t) * 20 * 16 * 4);
    if (err != cudaSuccess) { printf("GPUEngine: d_window_GX: %s\n", cudaGetErrorString(err)); return false; }
    
    err = cudaMemcpyToSymbol(d_window_GY, config->window_GY, sizeof(uint64_t) * 20 * 16 * 4);
    if (err != cudaSuccess) { printf("GPUEngine: d_window_GY: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_lockedPopcount, &config->lockedPopcount, sizeof(int));
    if (err != cudaSuccess) { printf("GPUEngine: d_lockedPopcount: %s\n", cudaGetErrorString(err)); return false; }
    
    // Upload stepSize to GPU
    err = cudaMemcpyToSymbol(d_stepSize, &config->stepSize, sizeof(int));
    if (err != cudaSuccess) { printf("GPUEngine: d_stepSize: %s\n", cudaGetErrorString(err)); return false; }

    printf("[StringCrack] GPU configuration uploaded\n"); fflush(stdout);
    return true;
}

// Compute the base point and free G table on CPU for Direct Seed Iteration
void GPUEngine::ComputeBasePoint(Secp256K1 *secp, StringCrackConfig *config) {
    // 1. Calculate the true locked popcount (cross-platform safe)
    config->lockedPopcount = 0;
    for(int i = 0; i < 4; i++) {
        uint64_t v = config->lockVals[i];
        while (v) {
            v &= (v - 1);
            config->lockedPopcount++;
        }
    }
    
    // 2. Precompute the Base Point (sum of all locked bits)
    Int lockedKey;
    lockedKey.SetInt32(0);
    lockedKey.bits64[0] = config->lockVals[0];
    lockedKey.bits64[1] = config->lockVals[1];
    lockedKey.bits64[2] = config->lockVals[2];
    lockedKey.bits64[3] = config->lockVals[3];

    Point basePoint = secp->ComputePublicKey(&lockedKey);
    memcpy(config->basePointX, basePoint.x.bits64, 32);
    memcpy(config->basePointY, basePoint.y.bits64, 32);

    // 3. Compact the G table for 5-bit Windows
    int numWindows = (config->numFreeBits + 4) / 5;  // 5 bits per window
    
    // Clear index 0 (Point at Infinity) for all windows to prevent garbage data
    memset(config->window_GX, 0, sizeof(config->window_GX));
    memset(config->window_GY, 0, sizeof(config->window_GY));

    for(int w = 0; w < numWindows; w++) {
        // Calculate all 31 non-zero combinations for this 5-bit window
        for(int val = 1; val < 32; val++) {
            Int windowKey;
            windowKey.SetInt32(0);
            
            for(int bit = 0; bit < 5; bit++) {
                if ((val >> bit) & 1) {
                    int pos_idx = w * 5 + bit;
                    if (pos_idx < config->numFreeBits) {
                        int actualPos = config->freeBitPositions[pos_idx];
                        windowKey.bits64[actualPos >> 6] |= (1ULL << (actualPos & 63));
                    }
                }
            }
            Point p = secp->ComputePublicKey(&windowKey);
            memcpy(config->window_GX[w][val], p.x.bits64, 32);
            memcpy(config->window_GY[w][val], p.y.bits64, 32);
        }
    }
    
    printf("[StringCrack] Base point computed (locked popcount: %d)\n", config->lockedPopcount);
    printf("[StringCrack]   Base X: %016llX %016llX %016llX %016llX\n",
           (unsigned long long)config->basePointX[3], (unsigned long long)config->basePointX[2],
           (unsigned long long)config->basePointX[1], (unsigned long long)config->basePointX[0]);
    printf("[StringCrack]   Base Y: %016llX %016llX %016llX %016llX\n",
           (unsigned long long)config->basePointY[3], (unsigned long long)config->basePointY[2],
           (unsigned long long)config->basePointY[1], (unsigned long long)config->basePointY[0]);
    printf("[StringCrack] Window tables: %d windows (5-bit each, 32 values per window)\n", numWindows);
    fflush(stdout);
}

bool GPUEngine::callOpenClawKernel(uint64_t batchOffsetLo, uint64_t batchOffsetHi) {
    cudaMemset(outputBuffer, 0, 4);
    // Pass batch offsets as kernel arguments (faster than cudaMemcpyToSymbol)
    // No need to copy to device globals - they're passed directly to the kernel

    comp_keys_openclaw<<<nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP>>>(
        inputAddress, inputAddressLookUp, outputBuffer, batchOffsetLo, batchOffsetHi);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("GPUEngine: OpenClaw Kernel: %s\n", cudaGetErrorString(err)); return false; }
    return true;
}

bool GPUEngine::LaunchOpenClaw(std::vector<ITEM> &addressFound, uint64_t batchOffsetLo, uint64_t batchOffsetHi, bool spinWait) {
    addressFound.clear();
    if (!callOpenClawKernel(batchOffsetLo, batchOffsetHi)) return false;

    if (spinWait) {
        cudaMemcpy(outputBufferPinned, outputBuffer, outputSize, cudaMemcpyDeviceToHost);
    } else {
        cudaEvent_t evt;
        cudaEventCreate(&evt);
        cudaMemcpyAsync(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost, 0);
        cudaEventRecord(evt, 0);
        while (cudaEventQuery(evt) == cudaErrorNotReady) Timer::SleepMillis(1);
        cudaEventDestroy(evt);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("GPUEngine: LaunchOpenClaw: %s\n", cudaGetErrorString(err)); return false; }

    uint32_t nbFound = outputBufferPinned[0];
    if (nbFound > maxFound) { nbFound = maxFound; }
    cudaMemcpy(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE32 + 1);
        ITEM it;
        it.thId = itemPtr[0];
        int16_t* ptr = (int16_t*)&(itemPtr[1]);
        it.endo = ptr[0] & 0x7FFF;
        it.mode = (ptr[0] & 0x8000) != 0;
        it.incr = ptr[1];
        it.hash = (uint8_t*)(itemPtr + 2);
        addressFound.push_back(it);
    }
    return true;
}