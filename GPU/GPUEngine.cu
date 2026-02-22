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
#define STEP_SIZE GRP_SIZE*1

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



GPUEngine::GPUEngine(int gpuId, uint32_t maxFound, int smMultiplier) {

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpuId);

    NB_TRHEAD_PER_GROUP = 256;                                          //////////////////  GRID SIZE ////////////////
    int nbThreadGroup = deviceProp.multiProcessorCount * smMultiplier;

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

    // Initialize asynchronous double-buffered streams
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_output[i], outputSize);
        cudaMallocHost(&h_outputPinned[i], outputSize);
    }
    currentStep = 0;

}

GPUEngine::~GPUEngine() {

    // Cleanup asynchronous double-buffered streams
    for (int i = 0; i < 2; i++) {
        cudaStreamDestroy(streams[i]);
        if (d_output[i]) cudaFree(d_output[i]);
        if (h_outputPinned[i]) cudaFreeHost(h_outputPinned[i]);
    }

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

// Global read-only pointers for window tables (use __ldg() in kernel)
__device__ uint64_t* d_window_GX;
__device__ uint64_t* d_window_GY;

// Keep basepoint in constant memory (small, frequently accessed)
__device__ __constant__ uint64_t d_basePointX[4];
__device__ __constant__ uint64_t d_basePointY[4];

__device__ __constant__ int      d_lockedPopcount;

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
    
    // Minimal register footprint: Only 4 temporary 256-bit variables
    uint64_t T1[4], T2[4], T3[4], T4[4];
    
    _ModSqr(T1, Z1);               // T1 = Z1^2
    _ModMult(T2, T1, x2);          // T2 = U2 = x2 * Z1^2
    _ModMult(T3, T1, Z1);          // T3 = Z1^3
    _ModMult(T4, T3, y2);          // T4 = S2 = y2 * Z1^3
    
    ModSub256(T2, T2, X1);         // T2 = H = U2 - X1
    ModSub256(T4, T4, Y1);         // T4 = R = S2 - Y1
    
    _ModSqr(T1, T2);               // T1 = HH = H^2
    _ModMult(T3, T1, T2);          // T3 = HHH = H^3
    _ModMult(T1, X1, T1);          // T1 = U1HH = X1 * H^2
    
    _ModSqr(X3, T4);               // X3 = R^2
    ModSub256(X3, X3, T3);         // X3 = R^2 - H^3
    
    uint64_t tmp[4], tmp2[4];      // Two extra arrays for safe subtraction
    ModNeg256(tmp, T1);
    ModSub256(tmp2, T1, tmp);      // tmp2 = 2 * U1HH
    ModSub256(X3, X3, tmp2);       // X3 = R^2 - H^3 - 2*U1HH
    
    ModSub256(Y3, T1, X3);         // Y3 = U1HH - X3
    _ModMult(Y3, T4, Y3);          // Y3 = R * (U1HH - X3)
    
    _ModMult(tmp, Y1, T3);         // tmp = Y1 * HHH
    ModSub256(Y3, Y3, tmp);        // Y3 = R * (U1HH - X3) - Y1 * HHH
    
    _ModMult(Z3, Z1, T2);          // Z3 = Z1 * H
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

// comp_keys_openclaw: StringCrack kernel - Stream Compaction + Direct Seed Iteration + EC Math
__global__ void comp_keys_openclaw(
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    uint64_t d_batchOffsetLo, uint64_t d_batchOffsetHi)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_in_block = threadIdx.x;
    
    // ==========================================================
    // PHASE 1: STREAM COMPACTION QUEUE ALLOCATION
    // Requires ~9.7 KB of shared memory per block
    // ==========================================================
    __shared__ uint64_t s_seed_lo[256];
    __shared__ uint64_t s_seed_hi[256];
    __shared__ uint8_t  s_orig_tid[256];
    __shared__ uint32_t s_hash[256][5];
    __shared__ bool     s_passed[256];
    __shared__ int      s_count;

    // Initialize shared tracking
    s_passed[tid_in_block] = false;
    if (tid_in_block == 0) s_count = 0;
    __syncthreads();
    
    // 128-bit Seed Generation with carry propagation
    uint64_t seed_lo = d_batchOffsetLo; 
    uint64_t seed_hi = d_batchOffsetHi; 
    
    // Add tid to lower 64 bits
    asm volatile ("add.cc.u64 %0, %0, %1;" 
                 : "+l"(seed_lo) 
                 : "l"((uint64_t)tid));
    // Add carry to upper 64 bits
    asm volatile ("addc.u64 %0, %0, 0;" 
                 : "+l"(seed_hi));

    // O(1) Popcount Filtering
    int pc = __popcll(seed_lo) + __popcll(seed_hi) + d_lockedPopcount;
    
    // 1. Evaluate filter condition
    bool is_valid = (pc >= d_popcountMin && pc <= d_popcountMax);
    
    // 2. Synchronize the warp and get a bitmask of all passing threads
    // 0xFFFFFFFF means all 32 threads in the warp participate in the ballot
    unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, is_valid);
    
    if (is_valid) {
        // Get the thread's index within its specific warp (0 to 31)
        int lane_id = threadIdx.x & 31;
        
        // 3. Calculate how many passing threads are *before* this one in the warp.
        // We mask off the bits at and above the current lane_id, then count the remaining 1s.
        int warp_offset = __popc(warp_mask & ((1u << lane_id) - 1));
        
        int base_offset = 0;
        
        // 4. Leader Election: Only the first passing thread in the warp executes the atomic.
        if (warp_offset == 0) {
            // Add the total number of passing threads in this warp (__popc(warp_mask)) 
            // to the global shared counter, returning the starting index for this warp.
            base_offset = atomicAdd(&s_count, __popc(warp_mask));
        }
        
        // 5. Broadcast the base_offset from the leader to all other passing threads.
        // __ffs(warp_mask) - 1 gives the lane_id of the leader.
        base_offset = __shfl_sync(warp_mask, base_offset, __ffs(warp_mask) - 1);
        
        // 6. Calculate the exact, unique queue index for this specific thread
        int q_idx = base_offset + warp_offset;
        
        // Push data to the dense shared queue
        s_seed_lo[q_idx] = seed_lo;
        s_seed_hi[q_idx] = seed_hi;
        s_orig_tid[q_idx] = tid_in_block;
        s_passed[tid_in_block] = true; // Mark original thread as alive
    }
    
    // Wait for all threads in the block to finish filtering
    __syncthreads();

    // --- Fast-Path Early Exit ---
    // If no threads survived the filter, drop the entire block instantly
    if (s_count == 0) return;

    // ==========================================================
    // PHASE 2: 100% EFFICIENT MATH EXECUTION
    // Only threads 0 to s_count execute this, completely packed
    // ==========================================================
    if (tid_in_block < s_count) {
        // Pull dense seed from queue
        uint64_t my_seed_lo = s_seed_lo[tid_in_block];
        uint64_t my_seed_hi = s_seed_hi[tid_in_block];
        uint8_t orig_tid = s_orig_tid[tid_in_block];

        // Initialize Jacobian Accumulator directly with the CPU Base Point
        uint64_t accX[4], accY[4], accZ[4];
        Load256(accX, d_basePointX);
        Load256(accY, d_basePointY);
        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0; // Z is always 1

        // Direct Seed Iteration (8-Bit Windows)
        int num_windows = (d_numFreeBits + 7) / 8;
        
        // --- Process lower 64 bits (Up to 8 windows) ---
        uint64_t seed = my_seed_lo;
        for (int w = 0; w < 8 && w < num_windows; w++) {
            int byte_val = seed & 0xFF;
            
            if (byte_val != 0) {
                int idx = (w * 256 + byte_val) * 4; 
                
                uint64_t curGX[4], curGY[4];
                curGX[0] = __ldg(&d_window_GX[idx + 0]);
                curGX[1] = __ldg(&d_window_GX[idx + 1]);
                curGX[2] = __ldg(&d_window_GX[idx + 2]);
                curGX[3] = __ldg(&d_window_GX[idx + 3]);
                curGY[0] = __ldg(&d_window_GY[idx + 0]);
                curGY[1] = __ldg(&d_window_GY[idx + 1]);
                curGY[2] = __ldg(&d_window_GY[idx + 2]);
                curGY[3] = __ldg(&d_window_GY[idx + 3]);

                uint64_t newX[4], newY[4], newZ[4];
                jacobian_add_affine(accX, accY, accZ, curGX, curGY, newX, newY, newZ);
                
                Load256(accX, newX);
                Load256(accY, newY);
                Load256(accZ, newZ);
            }
            seed >>= 8;
        }

        // --- Process upper 64 bits (Windows 8 to 15) ---
        seed = my_seed_hi;
        for (int w = 8; w < 16 && w < num_windows; w++) {
            int byte_val = seed & 0xFF;
            
            if (byte_val != 0) {
                int idx = (w * 256 + byte_val) * 4; 
                
                uint64_t curGX[4], curGY[4];
                curGX[0] = __ldg(&d_window_GX[idx + 0]);
                curGX[1] = __ldg(&d_window_GX[idx + 1]);
                curGX[2] = __ldg(&d_window_GX[idx + 2]);
                curGX[3] = __ldg(&d_window_GX[idx + 3]);
                curGY[0] = __ldg(&d_window_GY[idx + 0]);
                curGY[1] = __ldg(&d_window_GY[idx + 1]);
                curGY[2] = __ldg(&d_window_GY[idx + 2]);
                curGY[3] = __ldg(&d_window_GY[idx + 3]);

                uint64_t newX[4], newY[4], newZ[4];
                jacobian_add_affine(accX, accY, accZ, curGX, curGY, newX, newY, newZ);
                
                Load256(accX, newX);
                Load256(accY, newY);
                Load256(accZ, newZ);
            }
            seed >>= 8;
        }

        // Convert back to Affine and Hash
        uint64_t px[4], py[4];
        jacobian_to_affine(accX, accY, accZ, px, py);

        uint8_t odd_py = (uint8_t)(py[0] & 1);
        uint32_t h[5];
        _GetHash160Comp(px, odd_py, (uint8_t*)h);
        
        // Save the resulting hash back to the original thread's slot!
        s_hash[orig_tid][0] = h[0];
        s_hash[orig_tid][1] = h[1];
        s_hash[orig_tid][2] = h[2];
        s_hash[orig_tid][3] = h[3];
        s_hash[orig_tid][4] = h[4];
    }

    // Wait for math threads to finish writing hashes
    __syncthreads();

    // ==========================================================
    // PHASE 3: THE UNPACK
    // Original threads wake up and submit their own answers
    // ==========================================================
    if (s_passed[tid_in_block]) {
        uint32_t h[5];
        h[0] = s_hash[tid_in_block][0];
        h[1] = s_hash[tid_in_block][1];
        h[2] = s_hash[tid_in_block][2];
        h[3] = s_hash[tid_in_block][3];
        h[4] = s_hash[tid_in_block][4];
        
        // Because the original thread calls this, thId logic remains 100% intact
        CheckPoint(h, 0, sAddress, lookup32, out);
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

bool GPUEngine::SetStringCrackConfig(Secp256K1* secp, const StringCrackConfig *config) {
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
    
    // Upload precomputed base point to GPU (window tables now use global memory via __ldg)
    err = cudaMemcpyToSymbol(d_basePointX, config->basePointX, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("GPUEngine: d_basePointX: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_basePointY, config->basePointY, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("GPUEngine: d_basePointY: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_lockedPopcount, &config->lockedPopcount, sizeof(int));
    if (err != cudaSuccess) { printf("GPUEngine: d_lockedPopcount: %s\n", cudaGetErrorString(err)); return false; }

    // Compute and upload window tables to global memory
    ComputeWindowTables(secp, (StringCrackConfig*)config);

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

    // 3. Compact the G table for ONLY the free bits
    for(int i = 0; i < config->numFreeBits; i++) {
        int pos = config->freeBitPositions[i];
        Int bitKey;
        bitKey.SetInt32(0);
        bitKey.bits64[pos >> 6] = (1ULL << (pos & 63));
        Point p = secp->ComputePublicKey(&bitKey);
        memcpy(config->free_GX[i], p.x.bits64, 32);
        memcpy(config->free_GY[i], p.y.bits64, 32);
    }
    
    printf("[StringCrack] Base point computed (locked popcount: %d)\n", config->lockedPopcount);
    printf("[StringCrack]   Base X: %016llX %016llX %016llX %016llX\n",
           (unsigned long long)config->basePointX[3], (unsigned long long)config->basePointX[2],
           (unsigned long long)config->basePointX[1], (unsigned long long)config->basePointX[0]);
    printf("[StringCrack]   Base Y: %016llX %016llX %016llX %016llX\n",
           (unsigned long long)config->basePointY[3], (unsigned long long)config->basePointY[2],
           (unsigned long long)config->basePointY[1], (unsigned long long)config->basePointY[0]);
    printf("[StringCrack] Free G table: %d entries precomputed\n", config->numFreeBits);
    fflush(stdout);
}

// Compute 8-Bit Window Tables for accelerated lookup
void GPUEngine::ComputeWindowTables(Secp256K1 *secp, StringCrackConfig *config) {
    int num_windows = (config->numFreeBits + 7) / 8;
    
    // Allocate host memory (1D arrays to represent [window][256][4])
    uint64_t* h_window_GX = (uint64_t*)malloc(num_windows * 256 * 4 * sizeof(uint64_t));
    uint64_t* h_window_GY = (uint64_t*)malloc(num_windows * 256 * 4 * sizeof(uint64_t));
    memset(h_window_GX, 0, num_windows * 256 * 4 * sizeof(uint64_t));
    memset(h_window_GY, 0, num_windows * 256 * 4 * sizeof(uint64_t));

    for (int w = 0; w < num_windows; w++) {
        // Find which free bits belong to this window (up to 8 bits)
        int bits_in_window = 0;
        int window_bit_positions[8];
        
        for(int b = 0; b < 8; b++) {
            int global_bit_idx = (w * 8) + b;
            if (global_bit_idx < config->numFreeBits) {
                window_bit_positions[b] = config->freeBitPositions[global_bit_idx];
                bits_in_window++;
            }
        }

        // Generate all non-zero combinations for this byte
        int max_val = (1 << bits_in_window);
        for (int val = 1; val < max_val; val++) {
            Int bitKey;
            bitKey.SetInt32(0);
            
            // Map the bits of 'val' to their global physical positions
            for (int b = 0; b < bits_in_window; b++) {
                if ((val >> b) & 1) {
                    int pos = window_bit_positions[b];
                    bitKey.bits64[pos >> 6] |= (1ULL << (pos & 63));
                }
            }

            // Compute the point
            Point p = secp->ComputePublicKey(&bitKey);
            
            // Store flattened
            int idx = (w * 256 + val) * 4;
            memcpy(&h_window_GX[idx], p.x.bits64, 32);
            memcpy(&h_window_GY[idx], p.y.bits64, 32);
        }
    }

    // Allocate device memory and copy from host
    size_t tableSize = num_windows * 256 * 4 * sizeof(uint64_t);
    uint64_t* d_table_GX;
    uint64_t* d_table_GY;
    cudaMalloc((void**)&d_table_GX, tableSize);
    cudaMalloc((void**)&d_table_GY, tableSize);
    cudaMemcpy(d_table_GX, h_window_GX, tableSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_GY, h_window_GY, tableSize, cudaMemcpyHostToDevice);
    
    // Copy device pointers to device symbols
    cudaMemcpyToSymbol(d_window_GX, &d_table_GX, sizeof(uint64_t*));
    cudaMemcpyToSymbol(d_window_GY, &d_table_GY, sizeof(uint64_t*));
    
    // Free host memory
    free(h_window_GX);
    free(h_window_GY);
    
    printf("[StringCrack] Window tables: %d windows (%d KB)\n", num_windows, (num_windows * 256 * 32) / 1024);
    fflush(stdout);
}

bool GPUEngine::callOpenClawKernel(uint64_t batchOffsetLo, uint64_t batchOffsetHi, uint32_t* d_out, cudaStream_t stream) {
    cudaMemsetAsync(d_out, 0, 4, stream);
    
    // The synchronous cudaMemcpyToSymbol lines have been deleted from here!

    // Launch the kernel and pass the offsets directly via the stream
    comp_keys_openclaw<<<nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP, 0, stream>>>(
        inputAddress, inputAddressLookUp, d_out, batchOffsetLo, batchOffsetHi);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("GPUEngine: OpenClaw Kernel: %s\n", cudaGetErrorString(err)); return false; }
    return true;
}

bool GPUEngine::LaunchOpenClaw(std::vector<ITEM> &addressFound, uint64_t batchOffsetLo, uint64_t batchOffsetHi, bool spinWait) {
    addressFound.clear();
    if (!callOpenClawKernel(batchOffsetLo, batchOffsetHi, outputBuffer, 0)) return false;

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

// Asynchronous double-buffered launch
void GPUEngine::LaunchOpenClawAsync(uint64_t batchOffsetLo, uint64_t batchOffsetHi) {
    int s = currentStep % 2;

    // Reset the found counter for this stream asynchronously
    cudaMemsetAsync(d_output[s], 0, 4, streams[s]);

    // Launch the math kernel on this stream
    callOpenClawKernel(batchOffsetLo, batchOffsetHi, d_output[s], streams[s]);

    // Queue the result transfer back to the CPU asynchronously
    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize, cudaMemcpyDeviceToHost, streams[s]);

    currentStep++;
}

// Synchronize and get result for a specific stream
uint32_t GPUEngine::SyncAndGetResult(int stepToSync, std::vector<ITEM> &addressFound) {
    int s = stepToSync % 2;

    // Wait ONLY for this specific stream to finish
    cudaStreamSynchronize(streams[s]);

    uint32_t nbFound = h_outputPinned[s][0];
    if (nbFound > maxFound) { nbFound = maxFound; }

    addressFound.clear();
    if (nbFound > 0) {
        for (uint32_t i = 0; i < nbFound; i++) {
            uint32_t* itemPtr = h_outputPinned[s] + (i * ITEM_SIZE32 + 1);
            ITEM it;
            it.thId = itemPtr[0];
            int16_t* ptr = (int16_t*)&(itemPtr[1]);
            it.endo = ptr[0] & 0x7FFF;
            it.mode = (ptr[0] & 0x8000) != 0;
            it.incr = ptr[1];
            it.hash = (uint8_t*)(itemPtr + 2);
            addressFound.push_back(it);
        }
    }
    return nbFound;
}