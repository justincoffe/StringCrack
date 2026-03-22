// =====================================================================================
// MITM_Engine_v2.cu — ARMORED BENCHMARK KERNEL
// =====================================================================================

#ifndef MITM_ENGINE_V2_CU
#define MITM_ENGINE_V2_CU

#include <stdint.h>

// External native RevDoor tables guaranteed by g.ComputeDTable()
// (Do NOT use extern __device__ here to avoid redefinition errors with RevolvingDoor.cu)

// =====================================================================================
// SPECIALIZED AFFINE+AFFINE → JACOBIAN ADD (Z1 = 1)
// Saves 1 ModSqr + 2 ModMult = ~3 × 33 = 99 cycles per candidate (25% reduction).
// =====================================================================================
__device__ __forceinline__ void affine_add_affine_to_jacobian(
    const uint64_t x1[4], const uint64_t y1[4],
    const uint64_t x2[4], const uint64_t y2[4],
    uint64_t X3[4], uint64_t Y3[4], uint64_t Z3[4])
{
    __align__(32) uint64_t H[4];
    __align__(32) uint64_t R[4];
    ModSub256(H, x2, x1);    // H = x2 - x1
    ModSub256(R, y2, y1);    // R = y2 - y1

    __align__(32) uint64_t HH[4];
    __align__(32) uint64_t HHH[4];
    __align__(32) uint64_t U1HH[4];
    _ModSqr(HH, H);          // HH = H²
    _ModMult(HHH, HH, H);    // HHH = H³
    _ModMult(U1HH, x1, HH);  // U1HH = X1 * H²

    // X3 = R² - HHH - 2·U1HH
    _ModSqr(X3, R);
    ModSub256(X3, X3, HHH);

    __align__(32) uint64_t two_U1HH[4];
    __align__(32) uint64_t tmp_neg[4];
    ModNeg256(tmp_neg, U1HH);
    ModSub256(two_U1HH, U1HH, tmp_neg);  // 2·U1HH
    ModSub256(X3, X3, two_U1HH);

    // Y3 = R·(U1HH - X3) - Y1·HHH
    __align__(32) uint64_t diff[4];
    ModSub256(diff, U1HH, X3);
    _ModMult(Y3, R, diff);
    
    __align__(32) uint64_t y1_hhh[4];
    _ModMult(y1_hhh, y1, HHH);
    ModSub256(Y3, Y3, y1_hhh);

    // Z3 = Z1 · H = 1 · H = H
    Z3[0] = H[0]; Z3[1] = H[1]; Z3[2] = H[2]; Z3[3] = H[3];
}

// =====================================================================================
// KERNEL 1: VRAM Table Builder
// =====================================================================================
__global__ void comp_build_mitm_table(
    int L_half, int k_half, int bit_offset,
    uint64_t* Gfree_X, uint64_t* Gfree_Y,
    uint64_t* out_X, uint64_t* out_Y,
    uint8_t* out_seedpc,
    uint64_t center_slice,
    uint64_t total_combinations)
{
    uint64_t rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= total_combinations) return;

    if (k_half == 0) {
        out_X[rank * 4 + 0] = 0; out_X[rank * 4 + 1] = 0; out_X[rank * 4 + 2] = 0; out_X[rank * 4 + 3] = 0;
        out_Y[rank * 4 + 0] = 0; out_Y[rank * 4 + 1] = 0; out_Y[rank * 4 + 2] = 0; out_Y[rank * 4 + 3] = 0;
        out_seedpc[rank] = (uint8_t)__popcll(center_slice);
        return;
    }

    uint64_t mask = 0;
    uint64_t temp_rank = rank;
    int remaining = k_half;
    for (int i = L_half - 1; i >= 0 && remaining > 0; i--) {
        uint64_t c = __ldg(&d_rdCombTable[i * d_rdCombStride + remaining]);
        if (temp_rank >= c) {
            temp_rank -= c;
            mask |= (1ULL << i);
            remaining--;
        }
    }

    // Compute seed popcount for this entry
    uint64_t seed_bits = mask ^ center_slice;
    out_seedpc[rank] = (uint8_t)__popcll(seed_bits);

    __align__(32) uint64_t accX[4] = {0};
    __align__(32) uint64_t accY[4] = {0};
    __align__(32) uint64_t accZ[4] = {0};
    bool first = true;

    for (int i = 0; i < L_half; i++) {
        if ((mask >> i) & 1) {
            int real_idx = i + bit_offset;
            __align__(32) uint64_t ptX[4];
            __align__(32) uint64_t ptY[4];
            ptX[0] = __ldg(&Gfree_X[real_idx * 4]); ptX[1] = __ldg(&Gfree_X[real_idx * 4 + 1]);
            ptX[2] = __ldg(&Gfree_X[real_idx * 4 + 2]); ptX[3] = __ldg(&Gfree_X[real_idx * 4 + 3]);
            ptY[0] = __ldg(&Gfree_Y[real_idx * 4]); ptY[1] = __ldg(&Gfree_Y[real_idx * 4 + 1]);
            ptY[2] = __ldg(&Gfree_Y[real_idx * 4 + 2]); ptY[3] = __ldg(&Gfree_Y[real_idx * 4 + 3]);
            
            if (first) {
                Load256(accX, ptX); Load256(accY, ptY);
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                first = false;
            } else {
                jacobian_add_affine_inplace(accX, accY, accZ, ptX, ptY);
            }
        }
    }

    __align__(32) uint64_t Zinv[5];
    Zinv[0] = accZ[0]; Zinv[1] = accZ[1]; Zinv[2] = accZ[2]; Zinv[3] = accZ[3]; Zinv[4] = 0;
    _ModInv(Zinv);

    __align__(32) uint64_t Zinv_sq[4];
    __align__(32) uint64_t Zinv_cb[4];
    __align__(32) uint64_t affX[4];
    __align__(32) uint64_t affY[4];
    _ModSqr(Zinv_sq, Zinv);
    _ModMult(Zinv_cb, Zinv_sq, Zinv);
    _ModMult(affX, accX, Zinv_sq);
    _ModMult(affY, accY, Zinv_cb);

    out_X[rank * 4 + 0] = affX[0]; out_X[rank * 4 + 1] = affX[1]; out_X[rank * 4 + 2] = affX[2]; out_X[rank * 4 + 3] = affX[3];
    out_Y[rank * 4 + 0] = affY[0]; out_Y[rank * 4 + 1] = affY[1]; out_Y[rank * 4 + 2] = affY[2]; out_Y[rank * 4 + 3] = affY[3];
}

// =====================================================================================
// KERNEL 2: Q_i Point Builder
// =====================================================================================
__global__ void comp_build_qi_points(
    uint64_t* Gfree_X, uint64_t* Gfree_Y,
    uint64_t* out_X, uint64_t* out_Y,
    uint8_t* out_seedpc,
    uint64_t center_slice,
    uint64_t lx0, uint64_t lx1, uint64_t lx2, uint64_t lx3,
    uint64_t ly0, uint64_t ly1, uint64_t ly2, uint64_t ly3,
    int L_bits, int B_top, int k1, uint64_t W)
{
    uint64_t qi_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi_idx >= W) return;

    uint64_t qi_mask = 0;
    uint64_t temp_rank = qi_idx;
    int remaining = k1;
    for (int i = B_top - 1; i >= 0 && remaining > 0; i--) {
        uint64_t c = __ldg(&d_rdCombTable[i * d_rdCombStride + remaining]);
        if (temp_rank >= c) {
            temp_rank -= c;
            qi_mask |= (1ULL << i);
            remaining--;
        }
    }

    // Compute seed popcount for this Q_i entry
    uint64_t qi_seed_bits = qi_mask ^ center_slice;
    out_seedpc[qi_idx] = (uint8_t)__popcll(qi_seed_bits);

    __align__(32) uint64_t accX[4] = {lx0, lx1, lx2, lx3};
    __align__(32) uint64_t accY[4] = {ly0, ly1, ly2, ly3};
    __align__(32) uint64_t accZ[4] = {1, 0, 0, 0};

    for (int i = 0; i < B_top; i++) {
        if ((qi_mask >> i) & 1) {
            int real_idx = L_bits + i;
            __align__(32) uint64_t ptX[4];
            __align__(32) uint64_t ptY[4];
            ptX[0] = __ldg(&Gfree_X[real_idx * 4]); ptX[1] = __ldg(&Gfree_X[real_idx * 4 + 1]);
            ptX[2] = __ldg(&Gfree_X[real_idx * 4 + 2]); ptX[3] = __ldg(&Gfree_X[real_idx * 4 + 3]);
            ptY[0] = __ldg(&Gfree_Y[real_idx * 4]); ptY[1] = __ldg(&Gfree_Y[real_idx * 4 + 1]);
            ptY[2] = __ldg(&Gfree_Y[real_idx * 4 + 2]); ptY[3] = __ldg(&Gfree_Y[real_idx * 4 + 3]);
            jacobian_add_affine_inplace(accX, accY, accZ, ptX, ptY);
        }
    }

    __align__(32) uint64_t Zinv[5];
    Zinv[0] = accZ[0]; Zinv[1] = accZ[1]; Zinv[2] = accZ[2]; Zinv[3] = accZ[3]; Zinv[4] = 0;
    _ModInv(Zinv);

    __align__(32) uint64_t Zinv_sq[4];
    __align__(32) uint64_t Zinv_cb[4];
    __align__(32) uint64_t affX[4];
    __align__(32) uint64_t affY[4];
    _ModSqr(Zinv_sq, Zinv);
    _ModMult(Zinv_cb, Zinv_sq, Zinv);
    _ModMult(affX, accX, Zinv_sq);
    _ModMult(affY, accY, Zinv_cb);

    out_X[qi_idx * 4 + 0] = affX[0]; out_X[qi_idx * 4 + 1] = affX[1];
    out_X[qi_idx * 4 + 2] = affX[2]; out_X[qi_idx * 4 + 3] = affX[3];
    out_Y[qi_idx * 4 + 0] = affY[0]; out_Y[qi_idx * 4 + 1] = affY[1];
    out_Y[qi_idx * 4 + 2] = affY[2]; out_Y[qi_idx * 4 + 3] = affY[3];
}

// =====================================================================================
// DEVICE HELPER: Native Montgomery Batch Inversion (Armored)
// =====================================================================================
template<int MAX_BATCH>
__device__ __forceinline__ void mitm_batch_invert_Z(uint64_t Z_buf[][4], uint64_t Zinv_buf[][4], int count) {
    if (count == 0) return;
    if (count == 1) {
        __align__(32) uint64_t tmp[5];
        Load256(tmp, Z_buf[0]); tmp[4] = 0;
        _ModInv(tmp);
        Load256(Zinv_buf[0], tmp);
        return;
    }
    __align__(32) uint64_t prefix[MAX_BATCH][4]; Load256(prefix[0], Z_buf[0]);
    for (int i = 1; i < count; i++) _ModMult(prefix[i], prefix[i-1], Z_buf[i]);
    
    __align__(32) uint64_t total_inv[5]; Load256(total_inv, prefix[count - 1]); total_inv[4] = 0;
    _ModInv(total_inv);
    
    for (int i = count - 1; i >= 1; i--) {
        _ModMult(Zinv_buf[i], prefix[i-1], total_inv);
        __align__(32) uint64_t t[4]; _ModMult(t, total_inv, Z_buf[i]); Load256(total_inv, t);
    }
    Load256(Zinv_buf[0], total_inv);
}

// =====================================================================================
// KERNEL 3: THE GOD MATRIX v2
// =====================================================================================
__global__ __launch_bounds__(128, 4)
void comp_mitm_god_matrix_v2(
    uint64_t* T1_X, uint64_t* T1_Y, uint64_t T1_size,
    uint64_t* T2_X, uint64_t* T2_Y, uint64_t T2_size,
    uint64_t* Qi_X, uint64_t* Qi_Y,
    uint8_t* T1_seedpc, uint8_t* T2_seedpc, uint8_t* Qi_seedpc,
    uint64_t qi_start, uint64_t qi_count,
    uint64_t t2_start, uint64_t t2_end,
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    bool t1_is_baby)
{
    int blocks_per_qi = (T1_size + 127) / 128;
    if (blocks_per_qi == 0) blocks_per_qi = 1;

    uint64_t qi_local = blockIdx.x / blocks_per_qi;
    uint64_t t1_idx = (blockIdx.x % blocks_per_qi) * blockDim.x + threadIdx.x;

    if (qi_local >= qi_count || t1_idx >= T1_size) return;

    uint64_t qi_idx = qi_start + qi_local;

    // ═══ POPCOUNT PRE-FILTER (before any EC math) ═══
    if (d_popcountMin > 0 || d_popcountMax < 256) {
        int qi_pc = (int)Qi_seedpc[qi_local];
        int t1_pc = (int)T1_seedpc[t1_idx];
        int partial_pc = d_lockedPopcount + qi_pc + t1_pc;
        if (partial_pc > d_popcountMax) return;
        if (partial_pc + 64 < d_popcountMin) return;
    }

    // STEP 1: Load precomputed Q_i point (affine)
    __align__(32) uint64_t qiX[4];
    __align__(32) uint64_t qiY[4];
    qiX[0] = __ldg(&Qi_X[qi_local * 4 + 0]); qiX[1] = __ldg(&Qi_X[qi_local * 4 + 1]);
    qiX[2] = __ldg(&Qi_X[qi_local * 4 + 2]); qiX[3] = __ldg(&Qi_X[qi_local * 4 + 3]);
    qiY[0] = __ldg(&Qi_Y[qi_local * 4 + 0]); qiY[1] = __ldg(&Qi_Y[qi_local * 4 + 1]);
    qiY[2] = __ldg(&Qi_Y[qi_local * 4 + 2]); qiY[3] = __ldg(&Qi_Y[qi_local * 4 + 3]);

    // STEP 2: Load T1[t1_idx] point (affine)
    __align__(32) uint64_t t1X[4];
    __align__(32) uint64_t t1Y[4];
    t1X[0] = __ldg(&T1_X[t1_idx * 4 + 0]); t1X[1] = __ldg(&T1_X[t1_idx * 4 + 1]);
    t1X[2] = __ldg(&T1_X[t1_idx * 4 + 2]); t1X[3] = __ldg(&T1_X[t1_idx * 4 + 3]);
    t1Y[0] = __ldg(&T1_Y[t1_idx * 4 + 0]); t1Y[1] = __ldg(&T1_Y[t1_idx * 4 + 1]);
    t1Y[2] = __ldg(&T1_Y[t1_idx * 4 + 2]); t1Y[3] = __ldg(&T1_Y[t1_idx * 4 + 3]);

    // Preload popcount components for this (qi, t1) pair
    int _qi_pc = 0, _t1_pc = 0, _partial_pc = 0;
    bool _use_pcfilter = (d_popcountMin > 0 || d_popcountMax < 256);
    if (_use_pcfilter) {
        _qi_pc = (int)Qi_seedpc[qi_local];
        _t1_pc = (int)T1_seedpc[t1_idx];
        _partial_pc = d_lockedPopcount + _qi_pc + _t1_pc;
        if (_partial_pc > d_popcountMax) return;  // early exit entire thread
    }

    // STEP 3: Compute base = Q_i + T1
    __align__(32) uint64_t baseJX[4], baseJY[4], baseJZ[4];

    bool qi_zero = (qiX[0] == 0 && qiX[1] == 0 && qiX[2] == 0 && qiX[3] == 0);
    bool t1_zero = (t1X[0] == 0 && t1X[1] == 0 && t1X[2] == 0 && t1X[3] == 0);

    if (qi_zero && t1_zero) return;
    if (qi_zero) {
        Load256(baseJX, t1X); Load256(baseJY, t1Y);
        baseJZ[0] = 1; baseJZ[1] = 0; baseJZ[2] = 0; baseJZ[3] = 0;
    } else if (t1_zero) {
        Load256(baseJX, qiX); Load256(baseJY, qiY);
        baseJZ[0] = 1; baseJZ[1] = 0; baseJZ[2] = 0; baseJZ[3] = 0;
    } else {
        affine_add_affine_to_jacobian(qiX, qiY, t1X, t1Y, baseJX, baseJY, baseJZ);
    }

    // STEP 4: Convert base to affine (1 ModInv)
    __align__(32) uint64_t baseX[4], baseY[4];
    {
        __align__(32) uint64_t Zinv[5];
        Load256(Zinv, baseJZ); Zinv[4] = 0;
        _ModInv(Zinv);
        __align__(32) uint64_t Zsq[4], Zcb[4];
        _ModSqr(Zsq, Zinv);
        _ModMult(baseX, baseJX, Zsq);
        _ModMult(Zcb, Zsq, Zinv);
        _ModMult(baseY, baseJY, Zcb);
    }

    // STEP 5: Inner loop over T2
    const int BATCH_SIZE = 8;
    __align__(32) uint64_t buf_X[BATCH_SIZE][4], buf_Y[BATCH_SIZE][4], buf_Z[BATCH_SIZE][4];
    __align__(32) uint64_t Zinv_buf[BATCH_SIZE][4];
    __align__(32) uint32_t buf_t2_idx[BATCH_SIZE];
    int batch_count = 0;

    for (uint64_t t2_idx = t2_start; t2_idx < t2_end; t2_idx++) {

        __align__(32) uint64_t t2X[4];
        __align__(32) uint64_t t2Y[4];
        t2X[0] = __ldg(&T2_X[t2_idx * 4 + 0]); t2X[1] = __ldg(&T2_X[t2_idx * 4 + 1]);
        t2X[2] = __ldg(&T2_X[t2_idx * 4 + 2]); t2X[3] = __ldg(&T2_X[t2_idx * 4 + 3]);
        t2Y[0] = __ldg(&T2_Y[t2_idx * 4 + 0]); t2Y[1] = __ldg(&T2_Y[t2_idx * 4 + 1]);
        t2Y[2] = __ldg(&T2_Y[t2_idx * 4 + 2]); t2Y[3] = __ldg(&T2_Y[t2_idx * 4 + 3]);

        if (t2X[0] == 0 && t2X[1] == 0 && t2X[2] == 0 && t2X[3] == 0) continue;

        if (_use_pcfilter) {
            int total_pc = _partial_pc + (int)T2_seedpc[t2_idx];
            if (total_pc < d_popcountMin || total_pc > d_popcountMax) continue;
        }

        __align__(32) uint64_t cX[4], cY[4], cZ[4];
        affine_add_affine_to_jacobian(baseX, baseY, t2X, t2Y, cX, cY, cZ);

        Load256(buf_X[batch_count], cX);
        Load256(buf_Y[batch_count], cY);
        Load256(buf_Z[batch_count], cZ);
        buf_t2_idx[batch_count] = (uint32_t)t2_idx;
        batch_count++;

        if (batch_count >= BATCH_SIZE || t2_idx == T2_size - 1) {
            mitm_batch_invert_Z<BATCH_SIZE>(buf_Z, Zinv_buf, batch_count);

            for (int i = 0; i < batch_count; i++) {
                __align__(32) uint64_t Zsq[4], Zcb[4], aff_X[4], aff_Y[4];
                _ModSqr(Zsq, Zinv_buf[i]);
                _ModMult(aff_X, buf_X[i], Zsq);
                _ModMult(Zcb, Zsq, Zinv_buf[i]);
                _ModMult(aff_Y, buf_Y[i], Zcb);

                __align__(32) uint32_t hash[5];
                uint8_t isOdd = (uint8_t)(aff_Y[0] & 1);
                _GetHash160Comp(aff_X, isOdd, (uint8_t*)hash);

                uint32_t pr = hash[0] & 0xFFFF;
                if (sAddress[pr] != 0) {
                    bool reportHit = false;
                    if (lookup32 != NULL) {
                        uint32_t offset = lookup32[pr];
                        uint16_t count = sAddress[pr];
                        uint32_t la = hash[0];
                        for (uint16_t c = 0; c < count; c++) {
                            if (lookup32[offset + c] == la) { reportHit = true; break; }
                        }
                    } else {
                        reportHit = true;
                    }
                    if (reportHit) {
                        int id = atomicAdd(&out[0], 1);
                        if (id < 256) {
                            int off = 1 + (id * 8);
                            uint32_t b_idx = t1_is_baby ? (uint32_t)t1_idx : buf_t2_idx[i];
                            uint32_t g_idx = t1_is_baby ? buf_t2_idx[i] : (uint32_t)t1_idx;
                            out[off + 0] = (uint32_t)qi_idx;
                            out[off + 1] = g_idx;
                            out[off + 2] = b_idx;
                            out[off + 3] = hash[0]; out[off + 4] = hash[1];
                            out[off + 5] = hash[2]; out[off + 6] = hash[3];
                            out[off + 7] = hash[4];
                        }
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

    // printf("[MITM-v2] Baby: C(%d,%d)=%llu  Giant: C(%d,%d)=%llu  VRAM: %.0f MB\n",
    //        L_baby, k_baby, (unsigned long long)baby_combs,
    //        L_giant, k_giant, (unsigned long long)giant_combs,
    //        (baby_combs + giant_combs) * 64.0 / 1e6);
    // fflush(stdout);

    int n = config->numFreeBits;
    uint64_t* h_GX = (uint64_t*)calloc(n * 4, sizeof(uint64_t));
    uint64_t* h_GY = (uint64_t*)calloc(n * 4, sizeof(uint64_t));

    for (int i = 0; i < n; i++) {
        int pos = config->freeBitPositions[i];
        Int key; key.SetInt32(0);
        if (pos < 64) key.bits64[0] |= (1ULL << pos);
        else          key.bits64[1] |= (1ULL << (pos - 64));
        Point P_i = secp->ComputePublicKey(&key);

        // Check if free bit i is set in the center (seed-space indexed)
        bool center_bit = false;
        if (i < 64) center_bit = (config->targetSeedLo & (1ULL << i)) != 0;
        else         center_bit = (config->targetSeedHi & (1ULL << (i - 64))) != 0;
        if (center_bit) P_i.y.ModNeg();

        memcpy(&h_GX[i * 4], P_i.x.bits64, 32);
        memcpy(&h_GY[i * 4], P_i.y.bits64, 32);
    }

    cudaMemcpy(d_mitm_Gfree_X, h_GX, n * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mitm_Gfree_Y, h_GY, n * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    free(h_GX); free(h_GY);

    int tpb = 128;
    int baby_blocks = (int)((baby_combs + tpb - 1) / tpb);
    int giant_blocks = (int)((giant_combs + tpb - 1) / tpb);
    if (baby_blocks < 1) baby_blocks = 1;
    if (giant_blocks < 1) giant_blocks = 1;

    // Extract center bit slices for popcount computation
    uint64_t baby_center_slice = config->targetSeedLo & ((L_baby < 64) ? ((1ULL << L_baby) - 1) : 0xFFFFFFFFFFFFFFFFULL);
    uint64_t giant_center_slice = (config->targetSeedLo >> L_baby) & ((L_giant < 64) ? ((1ULL << L_giant) - 1) : 0xFFFFFFFFFFFFFFFFULL);

    comp_build_mitm_table<<<baby_blocks, tpb>>>(
        L_baby, k_baby, 0,
        d_mitm_Gfree_X, d_mitm_Gfree_Y,
        d_mitm_baby_X, d_mitm_baby_Y,
        d_mitm_baby_seedpc, baby_center_slice,
        baby_combs);

    comp_build_mitm_table<<<giant_blocks, tpb>>>(
        L_giant, k_giant, L_baby,
        d_mitm_Gfree_X, d_mitm_Gfree_Y,
        d_mitm_giant_X, d_mitm_giant_Y,
        d_mitm_giant_seedpc, giant_center_slice,
        giant_combs);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[MITM-v2] TABLE BUILD FAILED: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

void GPUEngine::BuildQiPoints(
    uint64_t lx0, uint64_t lx1, uint64_t lx2, uint64_t lx3,
    uint64_t ly0, uint64_t ly1, uint64_t ly2, uint64_t ly3,
    int L_bits, int B_top, int k1, uint64_t W,
    uint64_t qi_center_slice)
{
    int tpb = 128;
    int blocks = (int)((W + tpb - 1) / tpb);
    if (blocks < 1) blocks = 1;

    comp_build_qi_points<<<blocks, tpb>>>(
        d_mitm_Gfree_X, d_mitm_Gfree_Y,
        d_Qi_points_X, d_Qi_points_Y,
        d_mitm_qi_seedpc, qi_center_slice,
        lx0, lx1, lx2, lx3, ly0, ly1, ly2, ly3,
        L_bits, B_top, k1, W);

    cudaDeviceSynchronize();
}

void GPUEngine::LaunchMITMGodMatrixAsync(
    uint64_t baby_size, uint64_t giant_size,
    int L_bits, int B_top, int k1,
    uint64_t qi_start, uint64_t qi_count,
    uint64_t t2_start, uint64_t t2_end,
    uint64_t lx0, uint64_t lx1, uint64_t lx2, uint64_t lx3,
    uint64_t ly0, uint64_t ly1, uint64_t ly2, uint64_t ly3, int s)
{
    cudaMemsetAsync(d_output[s], 0, 8192, streams[s]);

    uint64_t* T1_X; uint64_t* T1_Y; uint64_t T1_size;
    uint64_t* T2_X; uint64_t* T2_Y; uint64_t T2_size;
    bool t1_is_baby;

    if (baby_size <= giant_size) {
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

    uint64_t* qi_X_offset = d_Qi_points_X + qi_start * 4;
    uint64_t* qi_Y_offset = d_Qi_points_Y + qi_start * 4;

    uint8_t* T1_seedpc;
    uint8_t* T2_seedpc;
    if (baby_size <= giant_size) {
        T1_seedpc = d_mitm_baby_seedpc;
        T2_seedpc = d_mitm_giant_seedpc;
    } else {
        T1_seedpc = d_mitm_giant_seedpc;
        T2_seedpc = d_mitm_baby_seedpc;
    }

    uint8_t* qi_seedpc_offset = d_mitm_qi_seedpc + qi_start;

    comp_mitm_god_matrix_v2<<<numBlocks, 128, 0, streams[s]>>>(
        T1_X, T1_Y, T1_size,
        T2_X, T2_Y, T2_size,
        qi_X_offset, qi_Y_offset,
        T1_seedpc, T2_seedpc, qi_seedpc_offset,
        qi_start, qi_count,
        t2_start, t2_end,
        inputAddress, inputAddressLookUp, d_output[s], t1_is_baby);

    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize,
                    cudaMemcpyDeviceToHost, streams[s]);
    currentStep++;
}

uint32_t GPUEngine::SyncMITMBatch(int s, std::vector<ITEM>& found) {
    cudaStreamSynchronize(streams[s]);
    uint32_t nbFound = h_outputPinned[s][0];
    if (nbFound > 256) nbFound = 256;
    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t* itemPtr = &h_outputPinned[s][1 + i * 8];
        ITEM it;
        it.thId = itemPtr[0];
        it.endo = itemPtr[1];
        it.incr = itemPtr[2];
        it.mode = true;
        it.hash = (uint8_t*)&itemPtr[3];
        found.push_back(it);
    }
    return nbFound;
}

#endif // MITM_ENGINE_V2_CU