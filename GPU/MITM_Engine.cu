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

    // 1. Unrank directly using the constant memory C(n,k) table
    uint64_t mask = 0;
    uint64_t temp_rank = rank;
    int remaining = k_half;
    
    for (int i = L_half - 1; i >= 0 && remaining > 0; i--) {
        uint64_t c = d_rdCombTable[i * d_rdCombStride + remaining];
        if (temp_rank >= c) {
            temp_rank -= c;
            mask |= (1ULL << i);
            remaining--;
        }
    }

    // 2. Build the Base Point using NATIVE jacobian_add_affine
    uint64_t accX[4] = {0}, accY[4] = {0}, accZ[4] = {0};
    uint64_t newX[4], newY[4], newZ[4];
    bool first = true;
    
    for(int i = 0; i < L_half; i++) {
        if ((mask >> i) & 1) {
            int real_idx = i + bit_offset; 
            
            uint64_t ptX[4], ptY[4];
            ptX[0] = Gfree_X[real_idx * 4]; ptX[1] = Gfree_X[real_idx * 4 + 1]; 
            ptX[2] = Gfree_X[real_idx * 4 + 2]; ptX[3] = Gfree_X[real_idx * 4 + 3];
            
            ptY[0] = Gfree_Y[real_idx * 4]; ptY[1] = Gfree_Y[real_idx * 4 + 1]; 
            ptY[2] = Gfree_Y[real_idx * 4 + 2]; ptY[3] = Gfree_Y[real_idx * 4 + 3];

            if (first) {
                accX[0]=ptX[0]; accX[1]=ptX[1]; accX[2]=ptX[2]; accX[3]=ptX[3];
                accY[0]=ptY[0]; accY[1]=ptY[1]; accY[2]=ptY[2]; accY[3]=ptY[3];
                accZ[0]=1; accZ[1]=0; accZ[2]=0; accZ[3]=0;
                first = false;
            } else {
                jacobian_add_affine(accX, accY, accZ, ptX, ptY, newX, newY, newZ);
                accX[0]=newX[0]; accX[1]=newX[1]; accX[2]=newX[2]; accX[3]=newX[3];
                accY[0]=newY[0]; accY[1]=newY[1]; accY[2]=newY[2]; accY[3]=newY[3];
                accZ[0]=newZ[0]; accZ[1]=newZ[1]; accZ[2]=newZ[2]; accZ[3]=newZ[3];
            }
        }
    }

    // 3. NATIVE Scalar Z-Inversion (Converts Jacobian back to pure Affine)
    uint64_t Zinv[5];
    Zinv[0] = accZ[0]; Zinv[1] = accZ[1]; Zinv[2] = accZ[2]; Zinv[3] = accZ[3]; Zinv[4] = 0;
    _ModInv(Zinv);
    
    uint64_t Zinv_sq[4], Zinv_cb[4];
    _ModSqr(Zinv_sq, Zinv);
    _ModMult(Zinv_cb, Zinv_sq, Zinv);

    uint64_t affX[4], affY[4];
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

// =====================================================================================
// BASE POINT STRUCT (Passes 96 bytes directly to Constant Memory for instant access)
// =====================================================================================
struct BasePointArgs {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
};

// =====================================================================================
// KERNEL 2: The Q-Shifter
// =====================================================================================
__global__ void comp_shift_baby_table(
    uint64_t* baby_X, uint64_t* baby_Y, 
    uint64_t* shifted_X, uint64_t* shifted_Y, 
    BasePointArgs base, uint64_t baby_size)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= baby_size) return;

    uint64_t bX[4], bY[4];
    bX[0] = baby_X[idx * 4 + 0]; bX[1] = baby_X[idx * 4 + 1]; 
    bX[2] = baby_X[idx * 4 + 2]; bX[3] = baby_X[idx * 4 + 3];

    bY[0] = baby_Y[idx * 4 + 0]; bY[1] = baby_Y[idx * 4 + 1]; 
    bY[2] = baby_Y[idx * 4 + 2]; bY[3] = baby_Y[idx * 4 + 3];

    uint64_t accX[4] = {base.X[0], base.X[1], base.X[2], base.X[3]};
    uint64_t accY[4] = {base.Y[0], base.Y[1], base.Y[2], base.Y[3]};
    uint64_t accZ[4] = {base.Z[0], base.Z[1], base.Z[2], base.Z[3]};
    uint64_t newX[4], newY[4], newZ[4];

    // NATIVE Mixed Add
    jacobian_add_affine(accX, accY, accZ, bX, bY, newX, newY, newZ);

    // NATIVE Z-Invert
    uint64_t Zinv[5];
    Zinv[0] = newZ[0]; Zinv[1] = newZ[1]; Zinv[2] = newZ[2]; Zinv[3] = newZ[3]; Zinv[4] = 0;
    _ModInv(Zinv);
    
    uint64_t Zinv_sq[4], Zinv_cb[4];
    _ModSqr(Zinv_sq, Zinv);
    _ModMult(Zinv_cb, Zinv_sq, Zinv);

    uint64_t affX[4], affY[4];
    _ModMult(affX, newX, Zinv_sq);
    _ModMult(affY, newY, Zinv_cb);

    shifted_X[idx * 4 + 0] = affX[0]; shifted_X[idx * 4 + 1] = affX[1]; 
    shifted_X[idx * 4 + 2] = affX[2]; shifted_X[idx * 4 + 3] = affX[3];

    shifted_Y[idx * 4 + 0] = affY[0]; shifted_Y[idx * 4 + 1] = affY[1]; 
    shifted_Y[idx * 4 + 2] = affY[2]; shifted_Y[idx * 4 + 3] = affY[3];
}

// =====================================================================================
// KERNEL 3: The Intersector (SCALAR INVERSION FOR GUARANTEED NATIVE COMPILATION)
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

    __shared__ uint64_t gX[4], gY[4];
    if (threadIdx.x == 0) {
        gX[0] = giant_X[giant_idx * 4 + 0]; gX[1] = giant_X[giant_idx * 4 + 1]; 
        gX[2] = giant_X[giant_idx * 4 + 2]; gX[3] = giant_X[giant_idx * 4 + 3];

        gY[0] = giant_Y[giant_idx * 4 + 0]; gY[1] = giant_Y[giant_idx * 4 + 1]; 
        gY[2] = giant_Y[giant_idx * 4 + 2]; gY[3] = giant_Y[giant_idx * 4 + 3];
    }
    __syncthreads();

    for (uint64_t b_idx = threadIdx.x; b_idx < baby_size; b_idx += blockDim.x) {
        
        uint64_t bX[4], bY[4];
        bX[0] = baby_shifted_X[b_idx * 4 + 0]; bX[1] = baby_shifted_X[b_idx * 4 + 1]; 
        bX[2] = baby_shifted_X[b_idx * 4 + 2]; bX[3] = baby_shifted_X[b_idx * 4 + 3];
        
        bY[0] = baby_shifted_Y[b_idx * 4 + 0]; bY[1] = baby_shifted_Y[b_idx * 4 + 1]; 
        bY[2] = baby_shifted_Y[b_idx * 4 + 2]; bY[3] = baby_shifted_Y[b_idx * 4 + 3];

        uint64_t accX[4] = {gX[0], gX[1], gX[2], gX[3]};
        uint64_t accY[4] = {gY[0], gY[1], gY[2], gY[3]};
        uint64_t accZ[4] = {1, 0, 0, 0};
        uint64_t newX[4], newY[4], newZ[4];

        // 1. NATIVE Add
        jacobian_add_affine(accX, accY, accZ, bX, bY, newX, newY, newZ);

        // 2. NATIVE Invert
        uint64_t Zinv[5];
        Zinv[0] = newZ[0]; Zinv[1] = newZ[1]; Zinv[2] = newZ[2]; Zinv[3] = newZ[3]; Zinv[4] = 0;
        _ModInv(Zinv);
        
        // 3. NATIVE Affine conversion
        uint64_t Zinv_sq[4], Zinv_cb[4], aff_X[4], aff_Y[4];
        _ModSqr(Zinv_sq, Zinv);
        _ModMult(Zinv_cb, Zinv_sq, Zinv);
        _ModMult(aff_X, newX, Zinv_sq);
        _ModMult(aff_Y, newY, Zinv_cb);

        // 4. NATIVE Hashing
        uint32_t hash[5];
        uint32_t isOdd = (uint32_t)(aff_Y[0] & 1);
        _GetHash160Comp(aff_X, isOdd, hash);

        address_t hash160 = *(address_t*)hash;
        uint32_t cl = hash160 & 0xFFFF;

        if (sAddress[cl] != 0) {
            uint32_t p = lookup32[cl];
            while (p != 0) {
                uint32_t* item = (uint32_t*)&sAddress[p];
                if (((uint64_t*)item)[0] == hash160) {
                    int id = atomicAdd(&out[8], 1);
                    int offset = 9 + (id * 9);
                    out[offset + 0] = qi_idx;           
                    out[offset + 1] = hash[0];
                    out[offset + 2] = hash[1];
                    out[offset + 3] = hash[2];
                    out[offset + 4] = hash[3];
                    out[offset + 5] = hash[4];
                    out[offset + 6] = giant_idx;        
                    out[offset + 7] = (uint32_t)b_idx;  
                    out[offset + 8] = 1;                
                }
                p = item[2];
            }
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

#endif // MITM_ENGINE_CU