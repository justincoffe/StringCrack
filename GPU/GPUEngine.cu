/*
 * Stratified Entropy Permutation Tool.
 * Copyright (c) 2026 AlleSerOjje.
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
    this->smCount = deviceProp.multiProcessorCount;

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
    int maxItemSize = (ITEM_SIZE32_WARP > ITEM_SIZE32) ? (ITEM_SIZE32_WARP * 4) : ITEM_SIZE;
    this->outputSize = (maxFound * maxItemSize + 4);

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
        cudaMalloc(&d_Qi_buffers[i], 4096 * sizeof(uint64_t));
    }
    currentStep = 0;
    qi_batches_last = 0;

    // SEP3: Initialize radius buffer pointers
    radiusBuffersReady = false;
    for (int i = 0; i < 2; i++) {
        d_radiusSeedsLo[i] = nullptr;
        d_radiusSeedsHi[i] = nullptr;
        h_radiusSeedsLo[i] = nullptr;
        h_radiusSeedsHi[i] = nullptr;
        d_radiusCount[i] = nullptr;
    }

    // =========================================================================
    // MITM VRAM ENGINE ALLOCATION (VRAM-AWARE)
    //
    // Runtime sizing: query free VRAM, reserve 1.5 GB for OS/driver/other
    // buffers, then allocate MITM tables to fill the remaining space.
    //
    // Budget breakdown per element:
    //   Baby  X+Y: 64 bytes (2 × 32-byte coordinates)
    //   Giant X+Y: 64 bytes
    //   Baby  seedpc: 1 byte
    //   Giant seedpc: 1 byte
    //   T2 perm:   4 bytes
    //   Total: 134 bytes per element
    //
    // On 16 GB GPU: ~14.5 GB available → ~108M elements max
    // On 24 GB GPU: ~22.5 GB available → ~167M elements max
    // Capped at C(27,13)=20,058,300 — beyond that the per-(k_b,k_g)
    // sub-round tables won't exceed this for practical puzzle sizes.
    // =========================================================================
    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);
    
    // Reserve 1.5 GB for driver, streams, address tables, other allocations
    size_t vramReserve = (size_t)1536 * 1024 * 1024;
    size_t vramBudget = (vramFree > vramReserve) ? (vramFree - vramReserve) : (size_t)512 * 1024 * 1024;
    
    // Each element costs 134 bytes across all MITM buffers
    // (baby_X + baby_Y + giant_X + giant_Y + baby_seedpc + giant_seedpc + t2_perm)
    // = 4*8*4 + 4*8*4 + 1 + 1 + 4 = 64 + 64 + 1 + 1 + 4 = 134 bytes
    size_t bytes_per_element = 4 * sizeof(uint64_t) * 4 + 2 * sizeof(uint8_t) + sizeof(uint32_t);
    uint64_t max_mitm_from_vram = vramBudget / bytes_per_element;
    
    // Cap at C(27,13) = 20,058,300 — practical maximum for half-sizes up to 27 bits
    uint64_t max_mitm_cap = 20058300ULL;
    // Floor at C(25,12) = 5,200,300 — minimum for reasonable MITM
    uint64_t max_mitm_floor = 5200300ULL;
    
    uint64_t max_mitm_elements = max_mitm_from_vram;
    if (max_mitm_elements > max_mitm_cap) max_mitm_elements = max_mitm_cap;
    if (max_mitm_elements < max_mitm_floor) max_mitm_elements = max_mitm_floor;
    
    size_t mitm_bytes = max_mitm_elements * 4 * sizeof(uint64_t); // 32 bytes per coordinate
    
    printf("[MITM-VRAM] GPU VRAM: %.1f GB total, %.1f GB free, %.1f GB budget\n",
           (double)vramTotal / (1024.0*1024.0*1024.0),
           (double)vramFree / (1024.0*1024.0*1024.0),
           (double)vramBudget / (1024.0*1024.0*1024.0));
    printf("[MITM-VRAM] MITM table capacity: %llu elements (%.1f MB per table)\n",
           (unsigned long long)max_mitm_elements,
           (double)mitm_bytes / (1024.0*1024.0));
    fflush(stdout);

    cudaMalloc((void**)&d_mitm_baby_X, mitm_bytes);
    cudaMalloc((void**)&d_mitm_baby_Y, mitm_bytes);
    cudaMalloc((void**)&d_mitm_giant_X, mitm_bytes);
    cudaMalloc((void**)&d_mitm_giant_Y, mitm_bytes);
    // Shifted buffers removed — unused in God Matrix v2 pipeline
    d_mitm_baby_shifted_X = nullptr;
    d_mitm_baby_shifted_Y = nullptr;
    
    // Allocate space for the 64 G_free points to pass to the builder
    cudaMalloc((void**)&d_mitm_Gfree_X, 64 * 4 * sizeof(uint64_t));
    cudaMalloc((void**)&d_mitm_Gfree_Y, 64 * 4 * sizeof(uint64_t));

    // MITM popcount pre-filter arrays (1 byte per table entry)
    cudaMalloc((void**)&d_mitm_baby_seedpc, max_mitm_elements * sizeof(uint8_t));
    cudaMalloc((void**)&d_mitm_giant_seedpc, max_mitm_elements * sizeof(uint8_t));
    cudaMalloc((void**)&d_mitm_qi_seedpc, 262144 * sizeof(uint8_t));

    // Sorted T2 popcount infrastructure
    cudaMalloc((void**)&d_mitm_t2_perm, max_mitm_elements * sizeof(uint32_t));

    // Allocate space for up to 262K Q_i points
    if (cudaMalloc((void**)&d_Qi_points_X, 262144 * 4 * sizeof(uint64_t)) != cudaSuccess) {
        printf("Failed to allocate d_Qi_points_X\n");
    }
    if (cudaMalloc((void**)&d_Qi_points_Y, 262144 * 4 * sizeof(uint64_t)) != cudaSuccess) {
        printf("Failed to allocate d_Qi_points_Y\n");
    }

}

GPUEngine::~GPUEngine() {

    // Cleanup asynchronous double-buffered streams
    for (int i = 0; i < 2; i++) {
        cudaStreamDestroy(streams[i]);
        if (d_output[i]) cudaFree(d_output[i]);
        if (h_outputPinned[i]) cudaFreeHost(h_outputPinned[i]);
        if (d_Qi_buffers[i]) cudaFree(d_Qi_buffers[i]);
    }

    // SEP3: Cleanup radius buffers
    for (int i = 0; i < 2; i++) {
        if (d_radiusSeedsLo[i]) cudaFree(d_radiusSeedsLo[i]);
        if (d_radiusSeedsHi[i]) cudaFree(d_radiusSeedsHi[i]);
        if (h_radiusSeedsLo[i]) cudaFreeHost(h_radiusSeedsLo[i]);
        if (h_radiusSeedsHi[i]) cudaFreeHost(h_radiusSeedsHi[i]);
    }

    // MITM VRAM ENGINE cleanup
    if (d_mitm_baby_X) cudaFree(d_mitm_baby_X);
    if (d_mitm_baby_Y) cudaFree(d_mitm_baby_Y);
    if (d_mitm_giant_X) cudaFree(d_mitm_giant_X);
    if (d_mitm_giant_Y) cudaFree(d_mitm_giant_Y);
    if (d_mitm_baby_shifted_X) cudaFree(d_mitm_baby_shifted_X);
    if (d_mitm_baby_shifted_Y) cudaFree(d_mitm_baby_shifted_Y);
    if (d_mitm_Gfree_X) cudaFree(d_mitm_Gfree_X);
    if (d_mitm_Gfree_Y) cudaFree(d_mitm_Gfree_Y);

    if (d_mitm_baby_seedpc) cudaFree(d_mitm_baby_seedpc);
    if (d_mitm_giant_seedpc) cudaFree(d_mitm_giant_seedpc);
    if (d_mitm_qi_seedpc) cudaFree(d_mitm_qi_seedpc);
    if (d_mitm_t2_perm) cudaFree(d_mitm_t2_perm);

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
__device__ __constant__ uint64_t d_seedMaskLo;
__device__ __constant__ uint64_t d_seedMaskHi;

// SEP (Stratified Entropy Permutation) device symbols
__device__ __constant__ uint64_t d_targetSeedLo;
__device__ __constant__ uint64_t d_targetSeedHi;
__device__ __constant__ bool     d_useSEP;
__device__ __constant__ int      d_sepMin;
__device__ __constant__ int      d_sepMax;

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

// Mixed Jacobian-Affine Addition (In-Place)
__device__ __forceinline__ void jacobian_add_affine_inplace(uint64_t X1[4], uint64_t Y1[4], uint64_t Z1[4],
                                            const uint64_t x2[4], const uint64_t y2[4]) {
    
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
    
    uint64_t X3[4];
    _ModSqr(X3, T4);               // X3 = R^2
    ModSub256(X3, X3, T3);         // X3 = R^2 - H^3
    
    // Fast 2*U1HH (Mathematically safe, natively branchless via our ModSub256)
    uint64_t two_U1HH[4];
    uint64_t tmp_neg[4];
    
    // T1 - (-T1 mod P) mod P = 2 * T1 mod P
    ModNeg256(tmp_neg, T1);
    ModSub256(two_U1HH, T1, tmp_neg);   
    
    ModSub256(X3, X3, two_U1HH);   // X3 = R^2 - H^3 - 2*U1HH
    
    uint64_t Y3[4];
    ModSub256(Y3, T1, X3);         // Y3 = U1HH - X3
    _ModMult(Y3, T4, Y3);          // Y3 = R * (U1HH - X3)
    
    uint64_t tmp[4];
    _ModMult(tmp, Y1, T3);         // tmp = Y1 * HHH
    ModSub256(Y3, Y3, tmp);        // Y3 = R * (U1HH - X3) - Y1 * HHH
    
    _ModMult(Z1, Z1, T2);          // Z1 = Z1 * H (Updated in-place)
    
    // Update X1 and Y1 in-place
    X1[0] = X3[0]; X1[1] = X3[1]; X1[2] = X3[2]; X1[3] = X3[3];
    Y1[0] = Y3[0]; Y1[1] = Y3[1]; Y1[2] = Y3[2]; Y1[3] = Y3[3];
}

// Convert Jacobian (X, Y, Z) to Affine (x, y)
__device__ void jacobian_to_affine(uint64_t X[4], uint64_t Y[4], uint64_t Z[4], uint64_t x[4], uint64_t y[4]) {
    uint64_t Z_inv[5];
    Load256(Z_inv, Z);
    Z_inv[4] = 0;
    _ModInv(Z_inv);
    
    uint64_t Z_inv_sq[4];
    _ModSqr(Z_inv_sq, Z_inv);
    _ModMult(x, Z_inv_sq, X);
    
    uint64_t Z_inv_cb[4];
    _ModMult(Z_inv_cb, Z_inv_sq, Z_inv);
    _ModMult(y, Z_inv_cb, Y);

    // Branchless zeroing for point at infinity
    bool is_inf = (Z[0] | Z[1] | Z[2] | Z[3]) == 0;
    if (is_inf) {
        x[0] = x[1] = x[2] = x[3] = 0;
        y[0] = y[1] = y[2] = y[3] = 0;
    }
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
__global__ __launch_bounds__(256, 2)
void comp_keys_openclaw(
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    uint64_t d_batchOffsetLo, uint64_t d_batchOffsetHi)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_in_block = threadIdx.x;
    
    // ==========================================================
    // PHASE 1: STREAM COMPACTION QUEUE ALLOCATION
    // ==========================================================
    __shared__ uint64_t s_seed_lo[256];
    __shared__ uint64_t s_seed_hi[256];
    
    // Upgraded to uint32_t to completely eliminate 4-way Bank Conflicts
    __shared__ uint32_t s_orig_tid[256]; 
    __shared__ uint32_t s_h0[256];
    __shared__ uint32_t s_h1[256];
    __shared__ uint32_t s_h2[256];
    __shared__ uint32_t s_h3[256];
    __shared__ uint32_t s_h4[256];
    __shared__ uint32_t s_hit[256];
    __shared__ int      s_count;

    // Initialize shared tracking
    s_hit[tid_in_block] = 0;
    if (tid_in_block == 0) s_count = 0;
    __syncthreads();
    
    uint64_t seed_lo = d_batchOffsetLo; 
    uint64_t seed_hi = d_batchOffsetHi; 
    
    asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(seed_lo) : "l"((uint64_t)tid));
    asm volatile ("addc.u64 %0, %0, 0;" : "+l"(seed_hi));

    // Fix: Mask the seeds BEFORE counting so the safe bits enter Phase 2's queue
    seed_lo &= d_seedMaskLo;
    seed_hi &= d_seedMaskHi;

    // 1. Always calculate the Absolute Density (Total 1s)
    int pc_abs = __popcll(seed_lo) + __popcll(seed_hi) + d_lockedPopcount;
    
    bool is_valid = false;
    
    if (d_useSEP) {
        // 2. Calculate the Mutations (Hamming Distance)
        int pc_mut = __popcll(seed_lo ^ d_targetSeedLo) + __popcll(seed_hi ^ d_targetSeedHi);
        
        // 3. DUAL FILTER: Must pass absolute density AND mutation count
        is_valid = (pc_mut >= d_sepMin && pc_mut <= d_sepMax) &&
                   (pc_abs >= d_popcountMin && pc_abs <= d_popcountMax);
    } else {
        // Standard StringCrack mode (Absolute Popcount only)
        is_valid = (pc_abs >= d_popcountMin && pc_abs <= d_popcountMax);
    }
    
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
                
                // Keep the highly-efficient vectorized memory loads
                ulonglong2 vec_GX_lo = __ldg((ulonglong2*)&d_window_GX[idx]);
                ulonglong2 vec_GX_hi = __ldg((ulonglong2*)&d_window_GX[idx + 2]);
                ulonglong2 vec_GY_lo = __ldg((ulonglong2*)&d_window_GY[idx]);
                ulonglong2 vec_GY_hi = __ldg((ulonglong2*)&d_window_GY[idx + 2]);

                uint64_t curGX[4] = {vec_GX_lo.x, vec_GX_lo.y, vec_GX_hi.x, vec_GX_hi.y};
                uint64_t curGY[4] = {vec_GY_lo.x, vec_GY_lo.y, vec_GY_hi.x, vec_GY_hi.y};

                // Call it directly on accX, accY, accZ. No need for newX/newY arrays or copying back!
                jacobian_add_affine_inplace(accX, accY, accZ, curGX, curGY);
            }
            seed >>= 8;
        }

        // --- Process upper 64 bits (Windows 8 to 15) ---
        seed = my_seed_hi;
        for (int w = 8; w < 16 && w < num_windows; w++) {
            int byte_val = seed & 0xFF;
            
            if (byte_val != 0) {
                int idx = (w * 256 + byte_val) * 4; 
                
                ulonglong2 vec_GX_lo = __ldg((ulonglong2*)&d_window_GX[idx]);
                ulonglong2 vec_GX_hi = __ldg((ulonglong2*)&d_window_GX[idx + 2]);
                ulonglong2 vec_GY_lo = __ldg((ulonglong2*)&d_window_GY[idx]);
                ulonglong2 vec_GY_hi = __ldg((ulonglong2*)&d_window_GY[idx + 2]);

                uint64_t curGX[4] = {vec_GX_lo.x, vec_GX_lo.y, vec_GX_hi.x, vec_GX_hi.y};
                uint64_t curGY[4] = {vec_GY_lo.x, vec_GY_lo.y, vec_GY_hi.x, vec_GY_hi.y};

                // Call it directly on accX, accY, accZ. No need for newX/newY arrays or copying back!
                jacobian_add_affine_inplace(accX, accY, accZ, curGX, curGY);
            }
            seed >>= 8;
        }

        // Convert back to Affine and Hash
        uint64_t px[4], py[4];
        jacobian_to_affine(accX, accY, accZ, px, py);

        uint8_t odd_py = (uint8_t)(py[0] & 1);
        uint32_t h[5];
        _GetHash160Comp(px, odd_py, (uint8_t*)h);
        
        // --- The Early Bloom Filter ---
        // Pre-check the global bloom filter. 
        // This prevents 99.999% of shared memory writes and Phase 3 thread wakeups.
        if (sAddress[h[0] & 0xFFFF] != 0) {
            // Write to shared memory ONLY if there is a potential hit
            s_h0[orig_tid] = h[0];
            s_h1[orig_tid] = h[1];
            s_h2[orig_tid] = h[2];
            s_h3[orig_tid] = h[3];
            s_h4[orig_tid] = h[4];
            s_hit[orig_tid] = 1; // Used 1 instead of true 
        }
    }

    // Wait for math threads to finish writing any potential hits
    __syncthreads();

    // ==========================================================
    // PHASE 3: THE UNPACK
    // Original threads wake up ONLY if their seed produced a hit
    // ==========================================================
    if (s_hit[tid_in_block] != 0) {
        uint32_t h[5];
        h[0] = s_h0[tid_in_block];
        h[1] = s_h1[tid_in_block];
        h[2] = s_h2[tid_in_block];
        h[3] = s_h3[tid_in_block];
        h[4] = s_h4[tid_in_block];
        
        // Final verification via standard CheckPoint
        CheckPoint(h, 0, sAddress, lookup32, out);
    }
}

// =====================================================================================
// SEP4: Precomputed C(n,k) table in constant memory for GPU unranking
// Supports n up to 128, k up to 128. Table is C[129][129].
// We only need the lower-left triangle but store the full row for simplicity.
// =====================================================================================

// Device global memory: C(n,k) table for combinatorial unranking
// Stored as uint64_t — saturates to UINT64_MAX on overflow (safe for comparison)
// Too large for constant memory (129*129*8 = 130KB > 64KB limit), so we use
// global memory with __ldg() for read-only texture cache access.
#define COMB_TABLE_N 129
#define COMB_TABLE_K 129
__device__ uint64_t* d_combTable;

// =====================================================================================
// SEP4: Combinatorial unranking — convert rank to flip-mask in GPU registers
// Given rank (0-indexed) among all C(n,k) combinations, produces a 128-bit
// bitmask with exactly k bits set, representing the rank-th combination
// in colexicographic (reverse lexicographic) order.
// =====================================================================================
__device__ __forceinline__ void unrank_combination(
    uint64_t rank, int n, int k,
    uint64_t &mask_lo, uint64_t &mask_hi)
{
    mask_lo = 0;
    mask_hi = 0;
    
    // Standard combinatorial unranking (colex order):
    // Greedy from the top: for each bit position i = n-1 down to 0,
    // if C(i, k) <= rank, include position i in the combination.
    int remaining = k;
    for (int i = n - 1; i >= 0 && remaining > 0; i--) {
        // C(i, remaining) from the precomputed table via read-only cache
        uint64_t c = __ldg(&d_combTable[i * COMB_TABLE_K + remaining]);
        if (rank >= c) {
            rank -= c;
            // Set bit i in the mask
            if (i < 64) {
                mask_lo |= (1ULL << i);
            } else {
                mask_hi |= (1ULL << (i - 64));
            }
            remaining--;
        }
    }
}

// --- INJECT SEP7 KERNELS (after device definitions) ---
#include "SEP7-GosperWalk-v2.cu"
#include "SEP7-RevolvingDoor.cu"
#include "MITM_Engine.cu"

// =====================================================================================
// SEP4: GPU-native Gosper kernel — combinatorial unranking in registers
// Each thread independently computes its combination from a global rank.
// No PCIe seed transfer, no CPU bottleneck. 100% GPU utilization.
// =====================================================================================

__global__ __launch_bounds__(256, 2)
void comp_keys_gosper(
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    int hamming_h, uint64_t batchOffset, uint64_t totalCombs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute this thread's global rank in the C(n, hamming_h) enumeration
    uint64_t myRank = batchOffset + (uint64_t)tid;
    
    // Boundary check — beyond this layer's combinations, thread is idle
    if (myRank >= totalCombs) return;
    
    // Unrank: convert myRank into a flip-mask with exactly hamming_h bits set
    uint64_t flip_lo, flip_hi;
    unrank_combination(myRank, d_numFreeBits, hamming_h, flip_lo, flip_hi);
    
    // XOR flip-mask with center target to get actual seed
    uint64_t seed_lo = flip_lo ^ d_targetSeedLo;
    uint64_t seed_hi = flip_hi ^ d_targetSeedHi;
    
    // Apply seed mask for safety
    seed_lo &= d_seedMaskLo;
    seed_hi &= d_seedMaskHi;
    
    // --- Popcount pre-filter (BEFORE EC math) ---
    // If user specified -poprange, check absolute popcount before spending
    // cycles on expensive EC math. This is a register-only operation.
    int pc_abs = __popcll(seed_lo) + __popcll(seed_hi) + d_lockedPopcount;
    if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) return;
    
    // Point at Infinity Safety Check
    // If d_lockedPopcount == 0, the base point is (0,0) which is invalid
    // in Jacobian (0,0,1). We defer initialization to the first non-zero window.
    uint64_t accX[4] = {0}, accY[4] = {0}, accZ[4] = {0};
    bool pointSet = false;
    
    if (d_lockedPopcount > 0) {
        Load256(accX, d_basePointX);
        Load256(accY, d_basePointY);
        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
        pointSet = true;
    }
    
    // Direct Seed Iteration (8-Bit Windows)
    int num_windows = (d_numFreeBits + 7) / 8;
    
    // Process lower 64 bits (up to 8 windows)
    uint64_t s = seed_lo;
    for (int w = 0; w < 8 && w < num_windows; w++) {
        int byte_val = s & 0xFF;
        if (byte_val != 0) {
            int idx = (w * 256 + byte_val) * 4;
            ulonglong2 vec_GX_lo = __ldg((ulonglong2*)&d_window_GX[idx]);
            ulonglong2 vec_GX_hi = __ldg((ulonglong2*)&d_window_GX[idx + 2]);
            ulonglong2 vec_GY_lo = __ldg((ulonglong2*)&d_window_GY[idx]);
            ulonglong2 vec_GY_hi = __ldg((ulonglong2*)&d_window_GY[idx + 2]);
            
            uint64_t curGX[4] = {vec_GX_lo.x, vec_GX_lo.y, vec_GX_hi.x, vec_GX_hi.y};
            uint64_t curGY[4] = {vec_GY_lo.x, vec_GY_lo.y, vec_GY_hi.x, vec_GY_hi.y};
            
            if (!pointSet) {
                Load256(accX, curGX);
                Load256(accY, curGY);
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                pointSet = true;
            } else {
                jacobian_add_affine_inplace(accX, accY, accZ, curGX, curGY);
            }
        }
        s >>= 8;
    }
    
    // Process upper 64 bits (windows 8 to 15)
    s = seed_hi;
    for (int w = 8; w < 16 && w < num_windows; w++) {
        int byte_val = s & 0xFF;
        if (byte_val != 0) {
            int idx = (w * 256 + byte_val) * 4;
            ulonglong2 vec_GX_lo = __ldg((ulonglong2*)&d_window_GX[idx]);
            ulonglong2 vec_GX_hi = __ldg((ulonglong2*)&d_window_GX[idx + 2]);
            ulonglong2 vec_GY_lo = __ldg((ulonglong2*)&d_window_GY[idx]);
            ulonglong2 vec_GY_hi = __ldg((ulonglong2*)&d_window_GY[idx + 2]);
            
            uint64_t curGX[4] = {vec_GX_lo.x, vec_GX_lo.y, vec_GX_hi.x, vec_GX_hi.y};
            uint64_t curGY[4] = {vec_GY_lo.x, vec_GY_lo.y, vec_GY_hi.x, vec_GY_hi.y};
            
            if (!pointSet) {
                Load256(accX, curGX);
                Load256(accY, curGY);
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                pointSet = true;
            } else {
                jacobian_add_affine_inplace(accX, accY, accZ, curGX, curGY);
            }
        }
        s >>= 8;
    }
    
    // If no point was ever set (seed was 0 and no locked bits), skip
    if (!pointSet) return;
    
    // Convert Jacobian to Affine and Hash
    uint64_t px[4], py[4];
    jacobian_to_affine(accX, accY, accZ, px, py);
    
    uint8_t odd_py = (uint8_t)(py[0] & 1);
    uint32_t h[5];
    _GetHash160Comp(px, odd_py, (uint8_t*)h);
    
    // CheckPoint writes tid = blockIdx.x * blockDim.x + threadIdx.x
    // CPU reconstructs: rank = batchOffset + tid → unrank → XOR target → key
    CheckPoint(h, 0, sAddress, lookup32, out);
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

    // SEP Logic: Map the center string to targetSeedLo/Hi
    config->targetSeedLo = 0;
    config->targetSeedHi = 0;
    config->seedMaskLo = (config->numFreeBits <= 64) ? ((1ULL << config->numFreeBits) - 1ULL) : 0xFFFFFFFFFFFFFFFFULL;
    
    if (config->useSEP) {
        int len = strlen(config->centerString);
        for (int i = 0; i < config->numFreeBits; i++) {
            int pos = config->freeBitPositions[i];
            
            // Map the MSB->LSB string index to the physical bit position
            // Bit 1 (Index 0) corresponds to pos (len - 1) e.g., pos 70
            int char_index = (len - 1) - pos; 
            
            if (char_index >= 0 && char_index < len) {
                if (config->centerString[char_index] == '1') {
                    if (i < 64) {
                        config->targetSeedLo |= (1ULL << i);
                    } else {
                        config->targetSeedHi |= (1ULL << (i - 64));
                    }
                }
            }
        }
        printf("[StringCrack] SEP Mode Enabled. Center string mapped.\n");
        printf("[StringCrack] Target Seed Lo: %016llX\n", (unsigned long long)config->targetSeedLo);
        printf("[StringCrack] Target Seed Hi: %016llX\n", (unsigned long long)config->targetSeedHi);
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

    // Upload SEP (Stratified Entropy Permutation) variables
    err = cudaMemcpyToSymbol(d_targetSeedLo, &config->targetSeedLo, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("GPUEngine: d_targetSeedLo: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_targetSeedHi, &config->targetSeedHi, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("GPUEngine: d_targetSeedHi: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_seedMaskLo, &config->seedMaskLo, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("GPUEngine: d_seedMaskLo: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_useSEP, &config->useSEP, sizeof(bool));
    if (err != cudaSuccess) { printf("GPUEngine: d_useSEP: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_sepMin, &config->sepMin, sizeof(int));
    if (err != cudaSuccess) { printf("GPUEngine: d_sepMin: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_sepMax, &config->sepMax, sizeof(int));
    if (err != cudaSuccess) { printf("GPUEngine: d_sepMax: %s\n", cudaGetErrorString(err)); return false; }

    // Compute and upload Popcount Correction Masks
    uint64_t seedMaskLo = 0, seedMaskHi = 0;
    if (config->numFreeBits <= 0) {
        seedMaskLo = 0; seedMaskHi = 0;
    } else if (config->numFreeBits < 64) {
        seedMaskLo = (1ULL << config->numFreeBits) - 1ULL; 
        seedMaskHi = 0;
    } else if (config->numFreeBits == 64) {
        seedMaskLo = 0xFFFFFFFFFFFFFFFFULL; 
        seedMaskHi = 0;
    } else if (config->numFreeBits < 128) {
        seedMaskLo = 0xFFFFFFFFFFFFFFFFULL; 
        seedMaskHi = (1ULL << (config->numFreeBits - 64)) - 1ULL;
    } else {
        seedMaskLo = 0xFFFFFFFFFFFFFFFFULL; 
        seedMaskHi = 0xFFFFFFFFFFFFFFFFULL;
    }

    err = cudaMemcpyToSymbol(d_seedMaskLo, &seedMaskLo, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("GPUEngine: d_seedMaskLo: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_seedMaskHi, &seedMaskHi, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("GPUEngine: d_seedMaskHi: %s\n", cudaGetErrorString(err)); return false; }

    // Safely execute and catch memory allocation errors
    if (!ComputeWindowTables(secp, (StringCrackConfig*)config)) {
        return false;
    }

    // SEP4: Compute and upload C(n,k) Pascal's triangle for GPU unranking
    if (config->useRadius) {
        int N = COMB_TABLE_N;
        int K = COMB_TABLE_K;
        size_t tableBytes = (size_t)N * K * sizeof(uint64_t);
        uint64_t* h_combTable = (uint64_t*)calloc(N * K, sizeof(uint64_t));
        
        // Build Pascal's triangle with saturation at UINT64_MAX
        for (int i = 0; i < N; i++) {
            h_combTable[i * K + 0] = 1;  // C(i, 0) = 1
            for (int j = 1; j <= i && j < K; j++) {
                uint64_t a = h_combTable[(i - 1) * K + (j - 1)];
                uint64_t b = h_combTable[(i - 1) * K + j];
                // Saturating addition: if a + b would overflow, clamp to UINT64_MAX
                if (a > 0xFFFFFFFFFFFFFFFFULL - b) {
                    h_combTable[i * K + j] = 0xFFFFFFFFFFFFFFFFULL;
                } else {
                    h_combTable[i * K + j] = a + b;
                }
            }
        }
        
        // Allocate in global memory (too large for constant memory)
        // Free old table if it exists
        uint64_t* old_combTable = nullptr;
        cudaMemcpyFromSymbol(&old_combTable, d_combTable, sizeof(uint64_t*));
        if (old_combTable) cudaFree(old_combTable);
        
        uint64_t* d_table = nullptr;
        err = cudaMalloc((void**)&d_table, tableBytes);
        if (err != cudaSuccess) { 
            printf("GPUEngine: d_combTable alloc: %s\n", cudaGetErrorString(err)); 
            free(h_combTable);
            return false; 
        }
        
        err = cudaMemcpy(d_table, h_combTable, tableBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { 
            printf("GPUEngine: d_combTable copy: %s\n", cudaGetErrorString(err)); 
            cudaFree(d_table);
            free(h_combTable);
            return false; 
        }
        
        err = cudaMemcpyToSymbol(d_combTable, &d_table, sizeof(uint64_t*));
        if (err != cudaSuccess) { 
            printf("GPUEngine: d_combTable symbol: %s\n", cudaGetErrorString(err)); 
            cudaFree(d_table);
            free(h_combTable);
            return false; 
        }
        
        free(h_combTable);
        printf("[SEP4] C(n,k) table uploaded to GPU global memory (%d x %d = %zu KB)\n", 
               N, K, tableBytes / 1024);
    }

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
bool GPUEngine::ComputeWindowTables(Secp256K1 *secp, StringCrackConfig *config) {
    int num_windows = (config->numFreeBits + 7) / 8;
    
    // Prevent malloc(0) crash if puzzle is fully locked
    if (num_windows == 0) return true; 

    // Safely retrieve and free old pointers to prevent VRAM leak
    uint64_t* old_GX = nullptr;
    uint64_t* old_GY = nullptr;
    cudaMemcpyFromSymbol(&old_GX, d_window_GX, sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&old_GY, d_window_GY, sizeof(uint64_t*));
    if (old_GX) cudaFree(old_GX);
    if (old_GY) cudaFree(old_GY);
    
    // Allocate host memory
    uint64_t* h_window_GX = (uint64_t*)malloc(num_windows * 256 * 4 * sizeof(uint64_t));
    uint64_t* h_window_GY = (uint64_t*)malloc(num_windows * 256 * 4 * sizeof(uint64_t));
    memset(h_window_GX, 0, num_windows * 256 * 4 * sizeof(uint64_t));
    memset(h_window_GY, 0, num_windows * 256 * 4 * sizeof(uint64_t));

    for (int w = 0; w < num_windows; w++) {
        int bits_in_window = 0;
        int window_bit_positions[8] = {0}; // Fix: Safe Array initialization
        
        for(int b = 0; b < 8; b++) {
            int global_bit_idx = (w * 8) + b;
            if (global_bit_idx < config->numFreeBits) {
                window_bit_positions[b] = config->freeBitPositions[global_bit_idx];
                bits_in_window++;
            }
        }

        int max_val = (1 << bits_in_window);
        for (int val = 1; val < max_val; val++) {
            Int bitKey;
            bitKey.SetInt32(0);
            for (int b = 0; b < bits_in_window; b++) {
                if ((val >> b) & 1) {
                    int pos = window_bit_positions[b];
                    bitKey.bits64[pos >> 6] |= (1ULL << (pos & 63));
                }
            }
            Point p = secp->ComputePublicKey(&bitKey);
            int idx = (w * 256 + val) * 4;
            memcpy(&h_window_GX[idx], p.x.bits64, 32);
            memcpy(&h_window_GY[idx], p.y.bits64, 32);
        }
    }

    // Strict Error Checking for GPU Allocations
    size_t tableSize = num_windows * 256 * 4 * sizeof(uint64_t);
    uint64_t* d_table_GX = nullptr;
    uint64_t* d_table_GY = nullptr;
    cudaError_t err;

    err = cudaMalloc((void**)&d_table_GX, tableSize);
    if (err != cudaSuccess) { free(h_window_GX); free(h_window_GY); printf("GPUEngine: OOM GX\n"); return false; }

    err = cudaMalloc((void**)&d_table_GY, tableSize);
    if (err != cudaSuccess) { cudaFree(d_table_GX); free(h_window_GX); free(h_window_GY); printf("GPUEngine: OOM GY\n"); return false; }

    cudaMemcpy(d_table_GX, h_window_GX, tableSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_GY, h_window_GY, tableSize, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(d_window_GX, &d_table_GX, sizeof(uint64_t*));
    cudaMemcpyToSymbol(d_window_GY, &d_table_GY, sizeof(uint64_t*));
    
    free(h_window_GX);
    free(h_window_GY);
    
    printf("[StringCrack] Window tables: %d windows (%d KB)\n", num_windows, (int)((num_windows * 256 * 32) / 1024));
    fflush(stdout);
    return true;
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
        // Transfer ONLY the 4-byte counter first
        cudaMemcpy(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost);
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
    
    // Transfer ONLY the valid structs, skipping massive amounts of zeros
    if (nbFound > 0) {
        cudaMemcpy(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost);
    }

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

// =====================================================================================
// SEP3: Radius Mode — Buffer Setup and Launch Methods
// =====================================================================================

bool GPUEngine::SetupRadiusBuffers() {
    radiusBuffersReady = false;
    cudaError_t err;
    
    for (int i = 0; i < 2; i++) {
        d_radiusSeedsLo[i] = nullptr;
        d_radiusSeedsHi[i] = nullptr;
        h_radiusSeedsLo[i] = nullptr;
        h_radiusSeedsHi[i] = nullptr;
        d_radiusCount[i] = nullptr;
        
        // Allocate device buffers for seeds (one uint64_t per thread)
        size_t seedBytes = (size_t)nbThread * sizeof(uint64_t);
        
        err = cudaMalloc((void**)&d_radiusSeedsLo[i], seedBytes);
        if (err != cudaSuccess) { printf("GPUEngine: Radius alloc SeedsLo[%d]: %s\n", i, cudaGetErrorString(err)); return false; }
        
        err = cudaMalloc((void**)&d_radiusSeedsHi[i], seedBytes);
        if (err != cudaSuccess) { printf("GPUEngine: Radius alloc SeedsHi[%d]: %s\n", i, cudaGetErrorString(err)); return false; }
        
        // Allocate pinned host staging buffers (write-combined for optimal H2D transfer)
        err = cudaHostAlloc((void**)&h_radiusSeedsLo[i], seedBytes, cudaHostAllocWriteCombined);
        if (err != cudaSuccess) { printf("GPUEngine: Radius alloc pinned Lo[%d]: %s\n", i, cudaGetErrorString(err)); return false; }
        
        err = cudaHostAlloc((void**)&h_radiusSeedsHi[i], seedBytes, cudaHostAllocWriteCombined);
        if (err != cudaSuccess) { printf("GPUEngine: Radius alloc pinned Hi[%d]: %s\n", i, cudaGetErrorString(err)); return false; }
    }
    
    size_t totalMB = (size_t)nbThread * sizeof(uint64_t) * 4 * 2 / (1024 * 1024);
    printf("[SEP3] Radius buffers allocated: %d threads x 2 streams = %zu MB\n", nbThread, totalMB);
    fflush(stdout);
    
    radiusBuffersReady = true;
    return true;
}

void GPUEngine::LaunchRadiusBatchAsync(uint64_t* seedsLo, uint64_t* seedsHi, int count) {
    // SEP3 legacy — replaced by LaunchGosperAsync in SEP4
    // Kept as stub to satisfy the linker; should not be called in radius mode.
    printf("[SEP4] ERROR: LaunchRadiusBatchAsync is deprecated. Use LaunchGosperAsync.\n");
}

uint32_t GPUEngine::SyncRadiusBatch(int stepToSync, std::vector<ITEM> &addressFound) {
    int s = stepToSync % 2;
    
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

// =====================================================================================
// SEP4: GPU-native Gosper — Launch and Sync Methods
// =====================================================================================

void GPUEngine::LaunchGosperAsync(int hamming_h, uint64_t batchOffset, uint64_t totalCombs) {
    int s = currentStep % 2;
    
    // Reset the found counter
    cudaMemsetAsync(d_output[s], 0, 4, streams[s]);
    
    // Launch the GPU-native Gosper kernel with exact layer bound
    comp_keys_gosper<<<nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP, 0, streams[s]>>>(
        inputAddress, inputAddressLookUp, d_output[s],
        hamming_h, batchOffset, totalCombs);
    
    // Queue the result transfer back
    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize, cudaMemcpyDeviceToHost, streams[s]);
    
    currentStep++;
}

uint32_t GPUEngine::SyncGosperBatch(int stepToSync, std::vector<ITEM> &addressFound) {
    int s = stepToSync % 2;
    
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

void GPUEngine::LaunchCosetGosperWalkAsync(
    int L_bits, int k2, int B_top, int k1,
    uint64_t base_pos, uint64_t L_totalCombs,
    int chunk_size, int total_walks, int W)
{
    int s = currentStep % 2;
    cudaMemsetAsync(d_output[s], 0, 4, streams[s]);
    
    int threadsPerBlock = 128;
    int numBlocks = (total_walks + threadsPerBlock - 1) / threadsPerBlock;
    
    comp_keys_coset_gosper_walk<BATCH_N><<<numBlocks, threadsPerBlock, 0, streams[s]>>>(
        inputAddress, inputAddressLookUp, d_output[s],
        L_bits, k2, B_top, k1, base_pos, L_totalCombs, chunk_size, W, d_Qi_buffers[s]);
    
    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize, cudaMemcpyDeviceToHost, streams[s]);
    currentStep++;
}

// =====================================================================================
// SHEKINAH MATRIX: Per-block GPU reconfiguration
// Reconfigures the GPU constant memory for a new Shekinah dispatch block.
// This is called once per block, before launching the MITM God Matrix kernel.
// It updates: lock masks, free bit positions, base point, popcount ranges,
// seed mask, SEP target, and the D-table for revolving door.
// =====================================================================================
bool GPUEngine::ReconfigureForShekinahBlock(Secp256K1* secp, StringCrackConfig* config) {
    scConfig = *config;
    stringCrackEnabled = config->enabled;
    if (!stringCrackEnabled) return true;

    cudaError_t err;

    // Upload lock masks
    err = cudaMemcpyToSymbol(d_lockMask, config->lockMask, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("[Shekinah] d_lockMask: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_lockVals, config->lockVals, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("[Shekinah] d_lockVals: %s\n", cudaGetErrorString(err)); return false; }

    // Upload free bit positions
    err = cudaMemcpyToSymbol(d_freeBitPos, config->freeBitPositions, sizeof(int) * 256);
    if (err != cudaSuccess) { printf("[Shekinah] d_freeBitPos: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_numFreeBits, &config->numFreeBits, sizeof(int));
    if (err != cudaSuccess) { printf("[Shekinah] d_numFreeBits: %s\n", cudaGetErrorString(err)); return false; }

    // Upload popcount range
    err = cudaMemcpyToSymbol(d_popcountMin, &config->popcountMin, sizeof(int));
    if (err != cudaSuccess) { printf("[Shekinah] d_popcountMin: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_popcountMax, &config->popcountMax, sizeof(int));
    if (err != cudaSuccess) { printf("[Shekinah] d_popcountMax: %s\n", cudaGetErrorString(err)); return false; }

    // Upload base point
    err = cudaMemcpyToSymbol(d_basePointX, config->basePointX, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("[Shekinah] d_basePointX: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_basePointY, config->basePointY, sizeof(uint64_t) * 4);
    if (err != cudaSuccess) { printf("[Shekinah] d_basePointY: %s\n", cudaGetErrorString(err)); return false; }

    // Upload locked popcount
    err = cudaMemcpyToSymbol(d_lockedPopcount, &config->lockedPopcount, sizeof(int));
    if (err != cudaSuccess) { printf("[Shekinah] d_lockedPopcount: %s\n", cudaGetErrorString(err)); return false; }

    // Upload SEP variables
    err = cudaMemcpyToSymbol(d_targetSeedLo, &config->targetSeedLo, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("[Shekinah] d_targetSeedLo: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_targetSeedHi, &config->targetSeedHi, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("[Shekinah] d_targetSeedHi: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_useSEP, &config->useSEP, sizeof(bool));
    if (err != cudaSuccess) { printf("[Shekinah] d_useSEP: %s\n", cudaGetErrorString(err)); return false; }

    // Upload seed masks
    uint64_t seedMaskLo = 0, seedMaskHi = 0;
    if (config->numFreeBits <= 0) {
        seedMaskLo = 0; seedMaskHi = 0;
    } else if (config->numFreeBits < 64) {
        seedMaskLo = (1ULL << config->numFreeBits) - 1ULL;
        seedMaskHi = 0;
    } else if (config->numFreeBits == 64) {
        seedMaskLo = 0xFFFFFFFFFFFFFFFFULL;
        seedMaskHi = 0;
    } else if (config->numFreeBits < 128) {
        seedMaskLo = 0xFFFFFFFFFFFFFFFFULL;
        seedMaskHi = (1ULL << (config->numFreeBits - 64)) - 1ULL;
    } else {
        seedMaskLo = 0xFFFFFFFFFFFFFFFFULL;
        seedMaskHi = 0xFFFFFFFFFFFFFFFFULL;
    }
    config->seedMaskLo = seedMaskLo;

    err = cudaMemcpyToSymbol(d_seedMaskLo, &seedMaskLo, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("[Shekinah] d_seedMaskLo: %s\n", cudaGetErrorString(err)); return false; }
    err = cudaMemcpyToSymbol(d_seedMaskHi, &seedMaskHi, sizeof(uint64_t));
    if (err != cudaSuccess) { printf("[Shekinah] d_seedMaskHi: %s\n", cudaGetErrorString(err)); return false; }

    return true;
}
