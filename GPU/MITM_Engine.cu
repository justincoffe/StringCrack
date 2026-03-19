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

// =====================================================================================
// DEVICE HELPER: Native Montgomery Batch Inversion
// =====================================================================================
template<int MAX_BATCH>
__device__ __forceinline__ void mitm_batch_invert_Z(uint64_t Z_buf[][4], uint64_t Zinv_buf[][4], int count) {
    if (count == 0) return;
    if (count == 1) {
        uint64_t tmp[5];
        Load256(tmp, Z_buf[0]); tmp[4] = 0;
        _ModInv(tmp);
        Load256(Zinv_buf[0], tmp);
        return;
    }
    uint64_t prefix[MAX_BATCH][4]; Load256(prefix[0], Z_buf[0]);
    for (int i = 1; i < count; i++) _ModMult(prefix[i], prefix[i-1], Z_buf[i]);
    uint64_t total_inv[5]; Load256(total_inv, prefix[count - 1]); total_inv[4] = 0;
    _ModInv(total_inv);
    for (int i = count - 1; i >= 1; i--) {
        _ModMult(Zinv_buf[i], prefix[i-1], (uint64_t*)total_inv);
        uint64_t t[4]; _ModMult(t, (uint64_t*)total_inv, Z_buf[i]); Load256(total_inv, t);
    }
    Load256(Zinv_buf[0], total_inv);
}

// =====================================================================================
// THE GOD MATRIX: Perfect 1-EC-Add ILP, Zero CPU Overhead
// Dynamically assigns threads to the smaller table to force max batching
// =====================================================================================
__global__ __launch_bounds__(128, 4) 
void comp_mitm_god_matrix(
    uint64_t* T1_X, uint64_t* T1_Y, uint64_t T1_size,
    uint64_t* T2_X, uint64_t* T2_Y, uint64_t T2_size,
    uint64_t* Gfree_X, uint64_t* Gfree_Y,
    uint64_t lx0, uint64_t lx1, uint64_t lx2, uint64_t lx3,
    uint64_t ly0, uint64_t ly1, uint64_t ly2, uint64_t ly3,
    int L_bits, int B_top, int k1,
    uint64_t qi_start, uint64_t qi_count,
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    bool t1_is_baby)
{
    int blocks_per_qi = (T1_size + 127) / 128;
    if (blocks_per_qi == 0) blocks_per_qi = 1;

    uint64_t qi_idx = qi_start + (blockIdx.x / blocks_per_qi);
    uint64_t t1_idx = (blockIdx.x % blocks_per_qi) * blockDim.x + threadIdx.x;

    if (qi_idx >= (qi_start + qi_count) || t1_idx >= T1_size) return;

    // 1. Unrank Q_i internally natively (Zero CPU Sync)
    uint64_t qi_mask_lo = 0;
    uint64_t temp_rank = qi_idx;
    int remaining = k1;
    for (int i = B_top - 1; i >= 0 && remaining > 0; i--) {
        uint64_t c = __ldg(&d_rdCombTable[i * d_rdCombStride + remaining]);
        if (temp_rank >= c) {
            temp_rank -= c;
            qi_mask_lo |= (1ULL << i);
            remaining--;
        }
    }

    // 2. Build True P_qi from Locked Base + Center String Gfree Table
    uint64_t accX[4] = {lx0, lx1, lx2, lx3};
    uint64_t accY[4] = {ly0, ly1, ly2, ly3};
    uint64_t accZ[4] = {1, 0, 0, 0};

    for (int i = 0; i < B_top; i++) {
        if ((qi_mask_lo >> i) & 1) {
            int real_idx = L_bits + i;
            uint64_t ptX[4], ptY[4];
            ptX[0] = Gfree_X[real_idx * 4]; ptX[1] = Gfree_X[real_idx * 4 + 1]; 
            ptX[2] = Gfree_X[real_idx * 4 + 2]; ptX[3] = Gfree_X[real_idx * 4 + 3];
            ptY[0] = Gfree_Y[real_idx * 4]; ptY[1] = Gfree_Y[real_idx * 4 + 1]; 
            ptY[2] = Gfree_Y[real_idx * 4 + 2]; ptY[3] = Gfree_Y[real_idx * 4 + 3];
            jacobian_add_affine_inplace(accX, accY, accZ, ptX, ptY);
        }
    }

    // 3. Add T1 Thread Element (1 EC Add)
    uint64_t t1X[4], t1Y[4];
    t1X[0] = T1_X[t1_idx * 4 + 0]; t1X[1] = T1_X[t1_idx * 4 + 1]; t1X[2] = T1_X[t1_idx * 4 + 2]; t1X[3] = T1_X[t1_idx * 4 + 3];
    t1Y[0] = T1_Y[t1_idx * 4 + 0]; t1Y[1] = T1_Y[t1_idx * 4 + 1]; t1Y[2] = T1_Y[t1_idx * 4 + 2]; t1Y[3] = T1_Y[t1_idx * 4 + 3];

    if (!(t1X[0] == 0 && t1X[1] == 0 && t1X[2] == 0 && t1X[3] == 0)) {
        jacobian_add_affine_inplace(accX, accY, accZ, t1X, t1Y);
    }

    // 4. Perfect ILP Loop over T2 (1 EC Add per Candidate)
    const int BATCH_SIZE = 8;
    uint64_t buf_X[BATCH_SIZE][4], buf_Y[BATCH_SIZE][4], buf_Z[BATCH_SIZE][4], Zinv[BATCH_SIZE][4];
    uint32_t buf_t2_idx[BATCH_SIZE];
    int batch_count = 0;

    for (uint64_t t2_idx = 0; t2_idx < T2_size; t2_idx++) {
        uint64_t t2X[4], t2Y[4];
        t2X[0] = T2_X[t2_idx * 4 + 0]; t2X[1] = T2_X[t2_idx * 4 + 1]; t2X[2] = T2_X[t2_idx * 4 + 2]; t2X[3] = T2_X[t2_idx * 4 + 3];
        t2Y[0] = T2_Y[t2_idx * 4 + 0]; t2Y[1] = T2_Y[t2_idx * 4 + 1]; t2Y[2] = T2_Y[t2_idx * 4 + 2]; t2Y[3] = T2_Y[t2_idx * 4 + 3];

        uint64_t cX[4] = {accX[0], accX[1], accX[2], accX[3]};
        uint64_t cY[4] = {accY[0], accY[1], accY[2], accY[3]};
        uint64_t cZ[4] = {accZ[0], accZ[1], accZ[2], accZ[3]};

        if (!(t2X[0] == 0 && t2X[1] == 0 && t2X[2] == 0 && t2X[3] == 0)) {
            jacobian_add_affine_inplace(cX, cY, cZ, t2X, t2Y);
        }

        buf_X[batch_count][0] = cX[0]; buf_X[batch_count][1] = cX[1]; buf_X[batch_count][2] = cX[2]; buf_X[batch_count][3] = cX[3];
        buf_Y[batch_count][0] = cY[0]; buf_Y[batch_count][1] = cY[1]; buf_Y[batch_count][2] = cY[2]; buf_Y[batch_count][3] = cY[3];
        buf_Z[batch_count][0] = cZ[0]; buf_Z[batch_count][1] = cZ[1]; buf_Z[batch_count][2] = cZ[2]; buf_Z[batch_count][3] = cZ[3];
        buf_t2_idx[batch_count] = (uint32_t)t2_idx;
        batch_count++;

        if (batch_count >= BATCH_SIZE || t2_idx == T2_size - 1) {
            mitm_batch_invert_Z<BATCH_SIZE>(buf_Z, Zinv, batch_count);

            for (int i = 0; i < batch_count; i++) {
                uint64_t Zsq[4], Zcb[4], aff_X[4], aff_Y[4];
                _ModSqr(Zsq, Zinv[i]); _ModMult(Zcb, Zsq, Zinv[i]);
                _ModMult(aff_X, buf_X[i], Zsq); _ModMult(aff_Y, buf_Y[i], Zcb);

                uint32_t hash[5]; uint8_t isOdd = (uint8_t)(aff_Y[0] & 1);
                _GetHash160Comp(aff_X, isOdd, (uint8_t*)hash);

                if (sAddress[hash[0] & 0xFFFF] != 0) {
                    uint64_t hash160 = *(uint64_t*)hash;
                    uint32_t cl = hash160 & 0xFFFF;
                    uint32_t p = lookup32[cl];
                    while (p != 0) {
                        uint32_t* item = (uint32_t*)&sAddress[p];
                        if (((uint64_t*)item)[0] == hash160) {
                            int id = atomicAdd(&out[0], 1);
                            if (id < 256) {
                                int offset = 1 + (id * 8);
                                uint32_t b_idx = t1_is_baby ? (uint32_t)t1_idx : buf_t2_idx[i];
                                uint32_t g_idx = t1_is_baby ? buf_t2_idx[i] : (uint32_t)t1_idx;
                                
                                out[offset + 0] = (uint32_t)qi_idx;           
                                out[offset + 1] = g_idx;        
                                out[offset + 2] = b_idx;  
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
// HOST LAUNCHER
// =====================================================================================
void GPUEngine::LaunchMITMGodMatrixAsync(
    uint64_t baby_size, uint64_t giant_size, 
    int L_bits, int B_top, int k1,
    uint64_t qi_start, uint64_t qi_count, 
    uint64_t lx0, uint64_t lx1, uint64_t lx2, uint64_t lx3,
    uint64_t ly0, uint64_t ly1, uint64_t ly2, uint64_t ly3, int s)
{
    cudaMemsetAsync(d_output[s], 0, 8192, streams[s]);
    
    uint64_t* T1_X; uint64_t* T1_Y; uint64_t T1_size;
    uint64_t* T2_X; uint64_t* T2_Y; uint64_t T2_size;
    bool t1_is_baby;

    // Dynamically assign Thread to the SMALLER table to maximize batch inversion looping!
    if (baby_size < giant_size) {
        T1_X = d_mitm_baby_X; T1_Y = d_mitm_baby_Y; T1_size = baby_size;
        T2_X = d_mitm_giant_X; T2_Y = d_mitm_giant_Y; T2_size = giant_size;
        t1_is_baby = true;
    } else {
        T1_X = d_mitm_giant_X; T1_Y = d_mitm_giant_Y; T1_size = giant_size;
        T2_X = d_mitm_baby_X; T2_Y = d_mitm_baby_Y; T2_size = baby_size;
        t1_is_baby = false;
    }

    int blocks_per_qi = (T1_size + 127) / 128;
    if (blocks_per_qi == 0) blocks_per_qi = 1;
    int numBlocks = qi_count * blocks_per_qi;

    comp_mitm_god_matrix<<<numBlocks, 128, 0, streams[s]>>>(
        T1_X, T1_Y, T1_size, T2_X, T2_Y, T2_size,
        d_mitm_Gfree_X, d_mitm_Gfree_Y,
        lx0, lx1, lx2, lx3, ly0, ly1, ly2, ly3,
        L_bits, B_top, k1, qi_start, qi_count,
        inputAddress, inputAddressLookUp, d_output[s], t1_is_baby);

    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize, cudaMemcpyDeviceToHost, streams[s]);
    currentStep++;
}

// =====================================================================================
// HOST SYNC
// =====================================================================================
uint32_t GPUEngine::SyncMITMBatch(int s, std::vector<ITEM>& found) {
    cudaStreamSynchronize(streams[s]);
    uint32_t nbFound = h_outputPinned[s][0];
    if (nbFound > 256) nbFound = 256;
    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t* itemPtr = &h_outputPinned[s][1 + i * 8];
        ITEM it;
        it.thId = itemPtr[0]; // EXACT qi_idx passed directly!
        it.endo = itemPtr[1]; // giant_idx
        it.incr = itemPtr[2]; // baby_idx
        it.mode = true;
        it.hash = (uint8_t*)&itemPtr[3]; 
        found.push_back(it);
    }
    return nbFound;
}

#endif // MITM_ENGINE_CU