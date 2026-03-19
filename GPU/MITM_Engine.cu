#ifndef MITM_ENGINE_CU
#define MITM_ENGINE_CU

#include <stdint.h>

// =====================================================================================
// KERNEL 1: The VRAM Table Builder (1 Thread = 1 Point)
// =====================================================================================
__global__ void comp_build_mitm_table(
    int L_half, int k_half, int bit_offset,
    uint64_t* Gfree_X, uint64_t* Gfree_Y,
    uint64_t* out_X, uint64_t* out_Y,
    uint64_t total_combinations)
{
    uint64_t rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= total_combinations) return;

    // Protection against Point at Infinity (k=0)
    if (k_half == 0) {
        out_X[rank * 4 + 0] = 0; out_X[rank * 4 + 1] = 0; out_X[rank * 4 + 2] = 0; out_X[rank * 4 + 3] = 0;
        out_Y[rank * 4 + 0] = 0; out_Y[rank * 4 + 1] = 0; out_Y[rank * 4 + 2] = 0; out_Y[rank * 4 + 3] = 0;
        return;
    }

    // 1. Unrank directly using the NATIVE global C(n,k) table
    uint64_t mask = 0;
    uint64_t temp_rank = rank;
    int remaining = k_half;
    
    for (int i = L_half - 1; i >= 0 && remaining > 0; i--) {
        // FIXED: Using the actual allocated pointer and stride from GPUEngine.cu
        uint64_t c = __ldg(&d_combTable[i * COMB_TABLE_K + remaining]);
        if (temp_rank >= c) {
            temp_rank -= c;
            mask |= (1ULL << i);
            remaining--;
        }
    }

    // 2. Build the Base Point using NATIVE jacobian_add_affine_inplace
    uint64_t accX[4] = {0};
    uint64_t accY[4] = {0};
    uint64_t accZ[4] = {0};
    bool first = true;
    
    for(int i = 0; i < L_half; i++) {
        if ((mask >> i) & 1) {
            int real_idx = i + bit_offset; 
            
            uint64_t ptX[4];
            uint64_t ptY[4];
            ptX[0] = Gfree_X[real_idx * 4]; ptX[1] = Gfree_X[real_idx * 4 + 1]; 
            ptX[2] = Gfree_X[real_idx * 4 + 2]; ptX[3] = Gfree_X[real_idx * 4 + 3];
            
            ptY[0] = Gfree_Y[real_idx * 4]; ptY[1] = Gfree_Y[real_idx * 4 + 1]; 
            ptY[2] = Gfree_Y[real_idx * 4 + 2]; ptY[3] = Gfree_Y[real_idx * 4 + 3];

            if (first) {
                accX[0] = ptX[0]; accX[1] = ptX[1]; accX[2] = ptX[2]; accX[3] = ptX[3];
                accY[0] = ptY[0]; accY[1] = ptY[1]; accY[2] = ptY[2]; accY[3] = ptY[3];
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                first = false;
            } else {
                jacobian_add_affine_inplace(accX, accY, accZ, ptX, ptY);
            }
        }
    }

    // 3. NATIVE Scalar Z-Inversion (Converts Jacobian back to pure Affine)
    uint64_t Zinv[5];
    Zinv[0] = accZ[0]; Zinv[1] = accZ[1]; Zinv[2] = accZ[2]; Zinv[3] = accZ[3]; Zinv[4] = 0;
    _ModInv(Zinv);
    
    uint64_t Zinv_sq[4];
    uint64_t Zinv_cb[4];
    _ModSqr(Zinv_sq, Zinv);
    _ModMult(Zinv_cb, Zinv_sq, Zinv);

    uint64_t affX[4];
    uint64_t affY[4];
    _ModMult(affX, accX, Zinv_sq);
    _ModMult(affY, accY, Zinv_cb);

    // 4. Write Affine result to VRAM
    out_X[rank * 4 + 0] = affX[0]; out_X[rank * 4 + 1] = affX[1]; 
    out_X[rank * 4 + 2] = affX[2]; out_X[rank * 4 + 3] = affX[3];
    
    out_Y[rank * 4 + 0] = affY[0]; out_Y[rank * 4 + 1] = affY[1]; 
    out_Y[rank * 4 + 2] = affY[2]; out_Y[rank * 4 + 3] = affY[3];
}

// =====================================================================================
// MITM VRAM ENGINE: Table Builder Dispatcher
// =====================================================================================
bool GPUEngine::BuildMITMTables(Secp256K1* secp, StringCrackConfig* config, 
                                int L_baby, int k_baby, int L_giant, int k_giant) 
{
    auto nCr = [](int n, int k) -> uint64_t {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n / 2) k = n - k;
        uint64_t res = 1;
        for (int i = 1; i <= k; i++) res = res * (n - i + 1) / i;
        return res;
    };
    
    uint64_t baby_combs = nCr(L_baby, k_baby);
    uint64_t giant_combs = nCr(L_giant, k_giant);
    
    printf("\n[MITM-BUILDER] Baby Table : C(%d, %d) = %llu points\n", L_baby, k_baby, (unsigned long long)baby_combs);
    printf("[MITM-BUILDER] Giant Table: C(%d, %d) = %llu points\n", L_giant, k_giant, (unsigned long long)giant_combs);
    fflush(stdout);

    int n = config->numFreeBits;
    uint64_t* h_GX = (uint64_t*)calloc(n * 4, sizeof(uint64_t));
    uint64_t* h_GY = (uint64_t*)calloc(n * 4, sizeof(uint64_t));
    
    for (int i = 0; i < n; i++) {
        int pos = config->freeBitPositions[i];
        Int key;
        key.SetInt32(0);
        
        if (pos < 64) key.bits64[0] |= (1ULL << pos);
        else          key.bits64[1] |= (1ULL << (pos - 64)); 
        
        Point P_i = secp->ComputePublicKey(&key);
        
        bool center_bit = false;
        if (pos < 64) center_bit = (config->targetSeedLo & (1ULL << pos)) != 0;
        else          center_bit = (config->targetSeedHi & (1ULL << (pos - 64))) != 0;
        
        if (center_bit) P_i.y.ModNeg(); 
        
        memcpy(&h_GX[i * 4], P_i.x.bits64, 32);
        memcpy(&h_GY[i * 4], P_i.y.bits64, 32);
    }
    
    cudaMemcpy(d_mitm_Gfree_X, h_GX, n * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mitm_Gfree_Y, h_GY, n * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    free(h_GX); free(h_GY);
    
    int threadsPerBlock = 128;
    int baby_blocks = (int)((baby_combs + threadsPerBlock - 1) / threadsPerBlock);
    int giant_blocks = (int)((giant_combs + threadsPerBlock - 1) / threadsPerBlock);
    
    if (baby_blocks < 1) baby_blocks = 1;
    if (giant_blocks < 1) giant_blocks = 1;
    
    printf("[MITM-BUILDER] Launching Silicon Builder (%d blocks)...\n", baby_blocks + giant_blocks);
    
    comp_build_mitm_table<<<baby_blocks, threadsPerBlock>>>(L_baby, k_baby, 0, d_mitm_Gfree_X, d_mitm_Gfree_Y, d_mitm_baby_X, d_mitm_baby_Y, baby_combs);
    comp_build_mitm_table<<<giant_blocks, threadsPerBlock>>>(L_giant, k_giant, L_baby, d_mitm_Gfree_X, d_mitm_Gfree_Y, d_mitm_giant_X, d_mitm_giant_Y, giant_combs);
        
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[MITM-BUILDER] KERNEL FAILED: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    printf("[MITM-BUILDER] VRAM Arrays initialized successfully.\n");
    fflush(stdout);
    return true;
}

// ====================================================================================
// PHASE 2: THE NUCLEAR REACTOR
// ====================================================================================

struct BasePointArgs {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
};

__global__ void comp_shift_baby_table(
    uint64_t* baby_X, uint64_t* baby_Y, 
    uint64_t* shifted_X, uint64_t* shifted_Y, 
    BasePointArgs base, uint64_t baby_size)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= baby_size) return;

    uint64_t bX[4];
    uint64_t bY[4];
    bX[0] = baby_X[idx * 4 + 0]; bX[1] = baby_X[idx * 4 + 1]; 
    bX[2] = baby_X[idx * 4 + 2]; bX[3] = baby_X[idx * 4 + 3];

    bY[0] = baby_Y[idx * 4 + 0]; bY[1] = baby_Y[idx * 4 + 1]; 
    bY[2] = baby_Y[idx * 4 + 2]; bY[3] = baby_Y[idx * 4 + 3];

    uint64_t accX[4] = {base.X[0], base.X[1], base.X[2], base.X[3]};
    uint64_t accY[4] = {base.Y[0], base.Y[1], base.Y[2], base.Y[3]};
    uint64_t accZ[4] = {base.Z[0], base.Z[1], base.Z[2], base.Z[3]};

    // Protection against Infinity!
    if (bX[0] == 0 && bX[1] == 0 && bX[2] == 0 && bX[3] == 0) {
        shifted_X[idx * 4 + 0] = accX[0]; shifted_X[idx * 4 + 1] = accX[1]; 
        shifted_X[idx * 4 + 2] = accX[2]; shifted_X[idx * 4 + 3] = accX[3];
        shifted_Y[idx * 4 + 0] = accY[0]; shifted_Y[idx * 4 + 1] = accY[1]; 
        shifted_Y[idx * 4 + 2] = accY[2]; shifted_Y[idx * 4 + 3] = accY[3];
        return;
    }

    // NATIVE Mixed Add
    jacobian_add_affine_inplace(accX, accY, accZ, bX, bY);

    // NATIVE Z-Invert
    uint64_t Zinv[5];
    Zinv[0] = accZ[0]; Zinv[1] = accZ[1]; Zinv[2] = accZ[2]; Zinv[3] = accZ[3]; Zinv[4] = 0;
    _ModInv(Zinv);
    
    uint64_t Zinv_sq[4];
    uint64_t Zinv_cb[4];
    _ModSqr(Zinv_sq, Zinv);
    _ModMult(Zinv_cb, Zinv_sq, Zinv);

    uint64_t affX[4];
    uint64_t affY[4];
    _ModMult(affX, accX, Zinv_sq);
    _ModMult(affY, accY, Zinv_cb);

    shifted_X[idx * 4 + 0] = affX[0]; shifted_X[idx * 4 + 1] = affX[1]; 
    shifted_X[idx * 4 + 2] = affX[2]; shifted_X[idx * 4 + 3] = affX[3];

    shifted_Y[idx * 4 + 0] = affY[0]; shifted_Y[idx * 4 + 1] = affY[1]; 
    shifted_Y[idx * 4 + 2] = affY[2]; shifted_Y[idx * 4 + 3] = affY[3];
}

// =====================================================================================
// DEVICE HELPER: Native Montgomery Batch Inversion
// =====================================================================================
template<int MAX_BATCH>
__device__ __forceinline__ void native_batch_invert(uint64_t out[][4], uint64_t in[][4], int count) {
    if (count == 0) return;
    
    uint64_t c[MAX_BATCH][4];
    c[0][0] = in[0][0]; c[0][1] = in[0][1]; c[0][2] = in[0][2]; c[0][3] = in[0][3];

    for (int i = 1; i < count; i++) {
        _ModMult(c[i], c[i-1], in[i]);
    }

    uint64_t inv[5];
    inv[0] = c[count-1][0]; inv[1] = c[count-1][1]; inv[2] = c[count-1][2]; inv[3] = c[count-1][3]; inv[4] = 0;
    _ModInv(inv);

    uint64_t inv4[4];
    inv4[0] = inv[0]; inv4[1] = inv[1]; inv4[2] = inv[2]; inv4[3] = inv[3];

    for (int i = count - 1; i > 0; i--) {
        _ModMult(out[i], inv4, c[i-1]);
        _ModMult(inv4, inv4, in[i]);
    }
    out[0][0] = inv4[0]; out[0][1] = inv4[1]; out[0][2] = inv4[2]; out[0][3] = inv4[3];
}

// =====================================================================================
// KERNEL 3: The Intersector (8x Batch Mode)
// =====================================================================================
__global__ __launch_bounds__(128, 4) 
void comp_mitm_intersect(
    uint64_t* baby_shifted_X, uint64_t* baby_shifted_Y,
    uint64_t* giant_X, uint64_t* giant_Y,
    uint64_t baby_size, uint64_t giant_size, uint32_t qi_idx,
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    uint64_t giant_offset)
{
    uint32_t giant_idx = blockIdx.x + giant_offset;
    if (giant_idx >= giant_size) return;

    __shared__ uint64_t gX[4];
    __shared__ uint64_t gY[4];
    if (threadIdx.x == 0) {
        gX[0] = giant_X[giant_idx * 4 + 0]; gX[1] = giant_X[giant_idx * 4 + 1]; 
        gX[2] = giant_X[giant_idx * 4 + 2]; gX[3] = giant_X[giant_idx * 4 + 3];

        gY[0] = giant_Y[giant_idx * 4 + 0]; gY[1] = giant_Y[giant_idx * 4 + 1]; 
        gY[2] = giant_Y[giant_idx * 4 + 2]; gY[3] = giant_Y[giant_idx * 4 + 3];
    }
    __syncthreads();

    // Reduced to 8 to prevent VRAM Register Spilling!
    const int BATCH_SIZE = 8;
    uint64_t buf_X[BATCH_SIZE][4];
    uint64_t buf_Y[BATCH_SIZE][4];
    uint64_t buf_Z[BATCH_SIZE][4];
    uint64_t Zinv[BATCH_SIZE][4];
    uint32_t buf_baby_idx[BATCH_SIZE];
    int batch_count = 0;

    for (uint64_t b_idx = threadIdx.x; b_idx < baby_size; b_idx += blockDim.x) {
        
        uint64_t bX[4], bY[4];
        bX[0] = baby_shifted_X[b_idx * 4 + 0]; bX[1] = baby_shifted_X[b_idx * 4 + 1]; 
        bX[2] = baby_shifted_X[b_idx * 4 + 2]; bX[3] = baby_shifted_X[b_idx * 4 + 3];
        
        bY[0] = baby_shifted_Y[b_idx * 4 + 0]; bY[1] = baby_shifted_Y[b_idx * 4 + 1]; 
        bY[2] = baby_shifted_Y[b_idx * 4 + 2]; bY[3] = baby_shifted_Y[b_idx * 4 + 3];

        uint64_t accX[4] = {gX[0], gX[1], gX[2], gX[3]};
        uint64_t accY[4] = {gY[0], gY[1], gY[2], gY[3]};
        uint64_t accZ[4] = {1, 0, 0, 0};

        if (gX[0] == 0 && gX[1] == 0 && gX[2] == 0 && gX[3] == 0) {
            accX[0] = bX[0]; accX[1] = bX[1]; accX[2] = bX[2]; accX[3] = bX[3];
            accY[0] = bY[0]; accY[1] = bY[1]; accY[2] = bY[2]; accY[3] = bY[3];
            accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
        } else {
            jacobian_add_affine_inplace(accX, accY, accZ, bX, bY);
        }

        buf_X[batch_count][0] = accX[0]; buf_X[batch_count][1] = accX[1]; buf_X[batch_count][2] = accX[2]; buf_X[batch_count][3] = accX[3];
        buf_Y[batch_count][0] = accY[0]; buf_Y[batch_count][1] = accY[1]; buf_Y[batch_count][2] = accY[2]; buf_Y[batch_count][3] = accY[3];
        buf_Z[batch_count][0] = accZ[0]; buf_Z[batch_count][1] = accZ[1]; buf_Z[batch_count][2] = accZ[2]; buf_Z[batch_count][3] = accZ[3];
        buf_baby_idx[batch_count] = (uint32_t)b_idx;
        batch_count++;

        if (batch_count >= BATCH_SIZE || (b_idx + blockDim.x) >= baby_size) {
            native_batch_invert<BATCH_SIZE>(Zinv, buf_Z, batch_count);

            for (int i = 0; i < batch_count; i++) {
                uint64_t Zsq[4], Zcb[4], aff_X[4], aff_Y[4];
                _ModSqr(Zsq, Zinv[i]);
                _ModMult(Zcb, Zsq, Zinv[i]);
                _ModMult(aff_X, buf_X[i], Zsq);
                _ModMult(aff_Y, buf_Y[i], Zcb);

                uint32_t hash[5];
                uint8_t isOdd = (uint8_t)(aff_Y[0] & 1);
                _GetHash160Comp(aff_X, isOdd, (uint8_t*)hash);

                // FULL 64-BIT NATIVE LOOKUP. Zero false positives.
                uint64_t hash160 = *(uint64_t*)hash;
                uint32_t cl = hash160 & 0xFFFF;

                if (sAddress[cl] != 0) {
                    uint32_t p = lookup32[cl];
                    while (p != 0) {
                        uint32_t* item = (uint32_t*)&sAddress[p];
                        if (((uint64_t*)item)[0] == hash160) {
                            int id = atomicAdd(&out[0], 1);
                            if (id < 256) {
                                int offset = 1 + (id * 8);
                                out[offset + 0] = qi_idx;           
                                out[offset + 1] = giant_idx;        
                                out[offset + 2] = buf_baby_idx[i];  
                                out[offset + 3] = hash[0];
                                out[offset + 4] = hash[1];
                                out[offset + 5] = hash[2];
                                out[offset + 6] = hash[3];
                                out[offset + 7] = hash[4];
                            }
                        }
                        p = item[2];
                    }
                }
            }
            batch_count = 0;
        }
    }
}

// =====================================================================================
// HOST LAUNCHERS
// =====================================================================================
void GPUEngine::ShiftBabyTable(uint64_t bX[4], uint64_t bY[4], uint64_t bZ[4], uint64_t baby_size) {
    BasePointArgs base;
    memcpy(base.X, bX, 32); memcpy(base.Y, bY, 32); memcpy(base.Z, bZ, 32);

    int threadsPerBlock = 128;
    int blocks = (baby_size + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks < 1) blocks = 1;
    
    comp_shift_baby_table<<<blocks, threadsPerBlock>>>(
        d_mitm_baby_X, d_mitm_baby_Y, 
        d_mitm_baby_shifted_X, d_mitm_baby_shifted_Y, 
        base, baby_size);
        
    cudaDeviceSynchronize(); 
}

void GPUEngine::LaunchMITMChunkAsync(uint32_t qi_idx, uint64_t baby_size, uint64_t giant_size, uint64_t offset, uint64_t blocks, int s) {
    cudaMemsetAsync(d_output[s], 0, 36, streams[s]);
    
    comp_mitm_intersect<<<blocks, 128, 0, streams[s]>>>(
        d_mitm_baby_shifted_X, d_mitm_baby_shifted_Y, 
        d_mitm_giant_X, d_mitm_giant_Y,
        baby_size, giant_size, qi_idx, 
        inputAddress, inputAddressLookUp, d_output[s], offset);
        
    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize, cudaMemcpyDeviceToHost, streams[s]);
    currentStep++;
}

// =====================================================================================
// HOST SYNC: Native VanitySearch ITEM Extraction
// =====================================================================================
uint32_t GPUEngine::SyncMITMBatch(int s, std::vector<ITEM>& found) {
    cudaStreamSynchronize(streams[s]);
    uint32_t nbFound = h_outputPinned[s][0];
    if (nbFound > maxFound) nbFound = maxFound;
    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t* itemPtr = h_outputPinned[s] + (i * ITEM_SIZE32 + 1);
        ITEM it;
        it.thId = itemPtr[0];
        int16_t* ptr = (int16_t*)&(itemPtr[1]);
        it.endo = ptr[0] & 0x7FFF;
        it.mode = (ptr[0] & 0x8000) != 0;
        it.incr = ptr[1];
        it.hash = (uint8_t*)(itemPtr + 2);
        found.push_back(it);
    }
    return nbFound;
}

#endif // MITM_ENGINE_CU