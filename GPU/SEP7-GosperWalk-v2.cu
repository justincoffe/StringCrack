// =====================================================================================
// SEP7: Revolving Door Walker with Thread-Local Batch Inversion
// =====================================================================================
//
// Architecture: 1 Thread = 1 Independent Walk. Pure SIMT.
//
// Key innovations over SEP6 Gosper kernel:
//   1. INCREMENTAL EC updates via XOR-diff (avg ~3 adds vs 7 full reconstruction)
//   2. THREAD-LOCAL Montgomery batch inversion over BATCH_N accumulated steps
//      (amortizes 1 ModInv over N candidates instead of 1 per candidate)
//   3. Gosper bit-hack walk (proven correct, O(1) per step, zero state arrays)
//
// The batch inversion is NOT warp-level (which adds overhead in SIMT).
// It's within a single thread, across consecutive walk steps stored in local memory.
//
// Cost model (BATCH_N=8, Gosper walk avg 3 diff bits):
//   Per step: 31.2 (EC adds) + 6.9 (amortized inv) + 8 (affine+hash) + 2 (mgmt)
//   Total: ~48 ModMult-equiv per candidate
//   vs SEP6 Gosper: ~140 per candidate
//   Speedup: ~2.9×
//
// With true revolving door (1 diff bit, future upgrade):
//   Per step: 10.4 + 6.9 + 8 + 2 = ~27 → speedup ~5.2×
//
// =====================================================================================

#ifndef BATCH_N
#define BATCH_N 8   // Candidates buffered per thread before batch ModInv
#endif

// =====================================================================================
// Per-free-bit EC point tables (device global memory, L2-cached)
// G_free[i]    = (2^freeBitPositions[i]) * G          (for adding a flipped bit)
// negG_free[i] = -(2^freeBitPositions[i]) * G         (for removing a flipped bit)
// =====================================================================================
__device__ uint64_t* d_GfreeX;     // [numFreeBits][4]
__device__ uint64_t* d_GfreeY;     // [numFreeBits][4]
__device__ uint64_t* d_negGfreeY;  // [numFreeBits][4] (X is same as GfreeX for negation)

// =====================================================================================
// Gosper bit-hack: next combination with same popcount, lexicographic order
// GPU-friendly: branchless core, O(1) per step, zero state arrays
// =====================================================================================
__device__ __forceinline__ uint64_t gosper_next(uint64_t v) {
    uint64_t c = v & (0ULL - v);           // lowest set bit
    uint64_t r = v + c;                     // carry through lowest block
    // __ffsll returns 1-indexed position of lowest set bit, or 0 if v==0
    int shift = __ffsll((long long)v) + 1;  // ctz(v) + 2
    uint64_t m = ((r ^ v) >> shift);
    return m | r;
}

// =====================================================================================
// Load an affine G_free point by free-bit index via L2 read-only cache
// =====================================================================================
__device__ __forceinline__ void load_Gfree(int bit_idx, uint64_t gx[4], uint64_t gy[4]) {
    int idx = bit_idx * 4;
    ulonglong2 vx_lo = __ldg((ulonglong2*)&d_GfreeX[idx]);
    ulonglong2 vx_hi = __ldg((ulonglong2*)&d_GfreeX[idx + 2]);
    ulonglong2 vy_lo = __ldg((ulonglong2*)&d_GfreeY[idx]);
    ulonglong2 vy_hi = __ldg((ulonglong2*)&d_GfreeY[idx + 2]);
    gx[0] = vx_lo.x; gx[1] = vx_lo.y; gx[2] = vx_hi.x; gx[3] = vx_hi.y;
    gy[0] = vy_lo.x; gy[1] = vy_lo.y; gy[2] = vy_hi.x; gy[3] = vy_hi.y;
}

__device__ __forceinline__ void load_negGfree(int bit_idx, uint64_t gx[4], uint64_t gy[4]) {
    int idx = bit_idx * 4;
    ulonglong2 vx_lo = __ldg((ulonglong2*)&d_GfreeX[idx]);      // X unchanged
    ulonglong2 vx_hi = __ldg((ulonglong2*)&d_GfreeX[idx + 2]);
    ulonglong2 vy_lo = __ldg((ulonglong2*)&d_negGfreeY[idx]);    // negated Y
    ulonglong2 vy_hi = __ldg((ulonglong2*)&d_negGfreeY[idx + 2]);
    gx[0] = vx_lo.x; gx[1] = vx_lo.y; gx[2] = vx_hi.x; gx[3] = vx_hi.y;
    gy[0] = vy_lo.x; gy[1] = vy_lo.y; gy[2] = vy_hi.x; gy[3] = vy_hi.y;
}

// =====================================================================================
// Apply incremental EC update: subtract removed bits, add new bits
// The accumulator stays in Jacobian throughout (Z != 1 in general)
// =====================================================================================
__device__ __forceinline__ void apply_xor_diff(
    uint64_t accX[4], uint64_t accY[4], uint64_t accZ[4],
    uint64_t old_seed, uint64_t new_seed)
{
    uint64_t diff = old_seed ^ new_seed;
    uint64_t removed = diff & old_seed;   // bits that were 1, now 0 -> Subtract G_free
    uint64_t added   = diff & new_seed;   // bits that were 0, now 1 -> Add G_free

    uint64_t gx[4], gy[4];

    while (removed) {
        int bit = __ffsll((long long)removed) - 1;
        removed &= (removed - 1);
        load_negGfree(bit, gx, gy);
        jacobian_add_affine_inplace(accX, accY, accZ, gx, gy);
    }
    while (added) {
        int bit = __ffsll((long long)added) - 1;
        added &= (added - 1);
        load_Gfree(bit, gx, gy);
        jacobian_add_affine_inplace(accX, accY, accZ, gx, gy);
    }
}

// =====================================================================================
// Thread-local Montgomery batch inversion
// Given N Z-values in local memory, compute N Z^-1 values using 1 ModInv
// Cost: (N-1) + 55 + (N-1) + N*2 = 3N + 53 ModMult-equiv (vs N*55 individual)
// =====================================================================================
__device__ void batch_invert_Z(
    uint64_t Z_buf[][4],     // N Jacobian Z values (in local memory)
    uint64_t Zinv_buf[][4],  // N output Z^-1 values
    int count)               // actual number of valid entries (≤ BATCH_N)
{
    if (count == 0) return;
    if (count == 1) {
        // Single inversion, no batching needed
        uint64_t tmp[5];
        Load256(tmp, Z_buf[0]);
        tmp[4] = 0;
        _ModInv(tmp);
        Load256(Zinv_buf[0], tmp);
        return;
    }
    
    // Forward pass: build prefix products
    // prefix[0] = Z[0]
    // prefix[i] = Z[0] * Z[1] * ... * Z[i]
    uint64_t prefix[BATCH_N][4];
    Load256(prefix[0], Z_buf[0]);
    for (int i = 1; i < count; i++) {
        _ModMult(prefix[i], prefix[i-1], Z_buf[i]);
    }
    
    // Single inversion of the total product
    uint64_t total_inv[5];
    Load256(total_inv, prefix[count - 1]);
    total_inv[4] = 0;
    _ModInv(total_inv);
    
    // Backward pass: extract individual inverses
    // Z_inv[count-1] = prefix[count-2] * total_inv
    // Then total_inv = total_inv * Z[count-1] (peel off last Z)
    // Z_inv[count-2] = prefix[count-3] * total_inv
    // ...
    // Z_inv[0] = total_inv (after peeling all)
    
    for (int i = count - 1; i >= 1; i--) {
        _ModMult(Zinv_buf[i], prefix[i-1], (uint64_t*)total_inv);
        // Peel: total_inv *= Z[i] (so it becomes inv of Z[0]*...*Z[i-1])
        uint64_t t[4];
        _ModMult(t, (uint64_t*)total_inv, Z_buf[i]);
        Load256(total_inv, t);
    }
    // Z_inv[0] = total_inv (which is now inv(Z[0]))
    Load256(Zinv_buf[0], total_inv);
}

// =====================================================================================
// Modified CheckPoint that stores walk_id and step for key reconstruction
// =====================================================================================
__device__ void CheckPointWalk(
    uint32_t* h, uint32_t walk_id, uint32_t step,
    address_t* sAddress, uint32_t* lookup32, uint32_t* out)
{
    address_t pr = (address_t)(h[0] & 0xFFFF);
    if (!sAddress[pr]) return;

    if (lookup32) {
        uint32_t offset = lookup32[pr];
        uint16_t count = sAddress[pr];
        addressl_t la = (addressl_t)(h[0]);

        for (uint16_t i = 0; i < count; i++) {
            if (lookup32[offset + i] == la) {
                uint32_t pos = atomicAdd(out, 1);
                if (pos < 65536) {
                    uint32_t* item = out + 1 + pos * ITEM_SIZE32;
                    item[0] = walk_id;
                    int16_t* ptr = (int16_t*)&item[1];
                    ptr[0] = (int16_t)(step & 0x7FFF);
                    ptr[1] = (int16_t)((step >> 15) & 0x7FFF);
                    memcpy(item + 2, h, 20);
                }
                return;
            }
        }
    } else {
        // Partial lookup hit
        uint32_t pos = atomicAdd(out, 1);
        if (pos < 65536) {
            uint32_t* item = out + 1 + pos * ITEM_SIZE32;
            item[0] = walk_id;
            int16_t* ptr = (int16_t*)&item[1];
            ptr[0] = (int16_t)(step & 0x7FFF);
            ptr[1] = (int16_t)((step >> 15) & 0x7FFF);
            memcpy(item + 2, h, 20);
        }
    }
}

// =====================================================================================
// THE KERNEL
// =====================================================================================

template <int MAX_BATCH>
__global__ __launch_bounds__(32, 14)
void comp_keys_gosper_walk(
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    int hamming_h,
    uint64_t base_rank_offset,
    uint64_t totalCombs,
    int chunk_size)
{
    // Each block = 1 warp = 32 independent walks
    int lane_id = threadIdx.x;
    uint32_t walk_id = blockIdx.x * 32 + lane_id;
    
    // Walk's starting rank in C(n, hamming_h)
    uint64_t start_rank = base_rank_offset + (uint64_t)walk_id * (uint64_t)chunk_size;
    if (start_rank >= totalCombs) return;
    
    int n = d_numFreeBits;
    uint64_t valid_mask = (n < 64) ? ((1ULL << n) - 1ULL) : 0xFFFFFFFFFFFFFFFFULL;
    
    // ═══════ UNRANK: Get initial flip-mask ═══════
    uint64_t flip_lo, flip_hi;
    unrank_combination(start_rank, n, hamming_h, flip_lo, flip_hi);
    
    // For n <= 64 (puzzle 71: n=59), only flip_lo is used
    uint64_t mask = flip_lo;
    
    // ═══════ COMPUTE INITIAL EC POINT (same window method as SEP6) ═══════
    uint64_t seed = (mask ^ d_targetSeedLo) & d_seedMaskLo;
    
    uint64_t accX[4], accY[4], accZ[4];
    bool pointSet = false;
    
    if (d_lockedPopcount > 0) {
        Load256(accX, d_basePointX);
        Load256(accY, d_basePointY);
        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
        pointSet = true;
    }
    
    int num_windows = (n + 7) / 8;
    uint64_t s = seed;
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
                Load256(accX, curGX); Load256(accY, curGY);
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                pointSet = true;
            } else {
                jacobian_add_affine_inplace(accX, accY, accZ, curGX, curGY);
            }
        }
        s >>= 8;
    }
    
    if (!pointSet) return;
    
    // ═══════ BATCHED WALK LOOP ═══════
    // Process chunk_size steps in batches of BATCH_N.
    // Each batch: N walk steps (accumulate Jacobian), then 1 batch ModInv, then N hashes.
    
    // Local memory buffers for batch accumulation
    uint64_t buf_X[MAX_BATCH][4];
    uint64_t buf_Y[MAX_BATCH][4];
    uint64_t buf_Z[MAX_BATCH][4];
    uint64_t buf_masks[MAX_BATCH];  // for popcount filter and key reconstruction
    uint64_t Zinv[MAX_BATCH][4];
    
    int steps_done = 0;
    int end_step = chunk_size;
    
    // Clamp to totalCombs boundary
    uint64_t max_steps = totalCombs - start_rank;
    if ((uint64_t)end_step > max_steps) end_step = (int)max_steps;
    
    // Store initial point as batch entry 0
    int batch_count = 0;
    Load256(buf_X[0], accX);
    Load256(buf_Y[0], accY);
    Load256(buf_Z[0], accZ);
    buf_masks[0] = mask;
    batch_count = 1;
    steps_done = 1;  // step 0 = initial combination
    
    while (steps_done < end_step) {
        uint64_t old_mask = mask;
        mask = gosper_next(mask);

        if ((mask & ~valid_mask) || mask == 0) break;

        uint64_t old_seed = (old_mask ^ d_targetSeedLo) & d_seedMaskLo;
        uint64_t new_seed = (mask ^ d_targetSeedLo) & d_seedMaskLo;

        apply_xor_diff(accX, accY, accZ, old_seed, new_seed);

        Load256(buf_X[batch_count], accX);
        Load256(buf_Y[batch_count], accY);
        Load256(buf_Z[batch_count], accZ);
        buf_masks[batch_count] = mask;
        batch_count++;
        steps_done++;

        if (batch_count >= MAX_BATCH) {
            batch_invert_Z(buf_Z, Zinv, batch_count);

            for (int b = 0; b < batch_count; b++) {
                uint64_t s_check = (buf_masks[b] ^ d_targetSeedLo) & d_seedMaskLo;
                int pc_abs = __popcll(s_check) + d_lockedPopcount;
                if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) continue;

                uint64_t Zinv_sq[4], px[4], py[4], Zinv_cb[4];
                _ModSqr(Zinv_sq, Zinv[b]);
                _ModMult(px, Zinv_sq, buf_X[b]);
                _ModMult(Zinv_cb, Zinv_sq, Zinv[b]);
                _ModMult(py, Zinv_cb, buf_Y[b]);

                uint8_t odd_py = (uint8_t)(py[0] & 1);
                uint32_t h[5];
                _GetHash160Comp(px, odd_py, (uint8_t*)h);

                if (sAddress[h[0] & 0xFFFF] != 0) {
                    uint32_t step_idx = steps_done - batch_count + b;
                    CheckPointWalk(h, walk_id, step_idx, sAddress, lookup32, out);
                }
            }

            if (steps_done < end_step) {
                uint64_t Zinv_sq[4], Zinv_cb[4];
                int last = batch_count - 1;
                _ModSqr(Zinv_sq, Zinv[last]);
                _ModMult(accX, Zinv_sq, buf_X[last]);
                _ModMult(Zinv_cb, Zinv_sq, Zinv[last]);
                _ModMult(accY, Zinv_cb, buf_Y[last]);
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
            }
            batch_count = 0;
        }
    }

    // ─── FLUSH REMAINING BATCH ───
    if (batch_count > 0) {
        batch_invert_Z(buf_Z, Zinv, batch_count);

        for (int b = 0; b < batch_count; b++) {
            uint64_t s_check = (buf_masks[b] ^ d_targetSeedLo) & d_seedMaskLo;
            int pc_abs = __popcll(s_check) + d_lockedPopcount;
            if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) continue;

            uint64_t Zinv_sq[4], px[4], py[4], Zinv_cb[4];
            _ModSqr(Zinv_sq, Zinv[b]);
            _ModMult(px, Zinv_sq, buf_X[b]);
            _ModMult(Zinv_cb, Zinv_sq, Zinv[b]);
            _ModMult(py, Zinv_cb, buf_Y[b]);

            uint8_t odd_py = (uint8_t)(py[0] & 1);
            uint32_t h[5];
            _GetHash160Comp(px, odd_py, (uint8_t*)h);

            if (sAddress[h[0] & 0xFFFF] != 0) {
                uint32_t step_idx = steps_done - batch_count + b;
                CheckPointWalk(h, walk_id, step_idx, sAddress, lookup32, out);
            }
        }
    }
}



template <int MAX_BATCH>
__global__ __launch_bounds__(128, 2)
void comp_keys_coset_gosper_walk(
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    int L_bits, int k2, int B_top, int k1,
    uint64_t base_pos, uint64_t L_totalCombs,
    int chunk_size, int W, uint64_t* d_Qi_array)
{
    uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Decompose into walk_chunk and Q_i offset
    uint32_t walk_chunk_id = global_id / W;
    uint32_t qi_idx = global_id % W;
    
    uint64_t start_rank = base_pos + (uint64_t)walk_chunk_id * chunk_size;
    if (start_rank >= L_totalCombs) return;

    uint64_t qi_mask = d_Qi_array[qi_idx] << L_bits;
    uint64_t valid_mask = (L_bits < 64) ? ((1ULL << L_bits) - 1ULL) : 0xFFFFFFFFFFFFFFFFULL;

    // Unrank purely within the L_bits space
    uint64_t flip_lo, flip_hi;
    unrank_combination(start_rank, L_bits, k2, flip_lo, flip_hi);
    uint64_t mask = flip_lo;

    // Build initial EC point combining Qi and L_bits
    uint64_t full_mask = qi_mask | mask;
    uint64_t seed = (full_mask ^ d_targetSeedLo) & d_seedMaskLo;
    
    uint64_t accX[4], accY[4], accZ[4];
    bool pointSet = false;
    
    if (d_lockedPopcount > 0) {
        Load256(accX, d_basePointX); Load256(accY, d_basePointY);
        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
        pointSet = true;
    }
    
    int num_windows = (d_numFreeBits + 7) / 8;
    uint64_t s = seed;
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
                Load256(accX, curGX); Load256(accY, curGY);
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                pointSet = true;
            } else {
                jacobian_add_affine_inplace(accX, accY, accZ, curGX, curGY);
            }
        }
        s >>= 8;
    }
    
    if (!pointSet) return;
    
    // Batched Walk Loop
    uint64_t buf_X[MAX_BATCH][4], buf_Y[MAX_BATCH][4], buf_Z[MAX_BATCH][4];
    uint64_t buf_masks[MAX_BATCH], Zinv[MAX_BATCH][4];
    
    int steps_done = 0;
    int end_step = chunk_size;
    uint64_t max_steps = L_totalCombs - start_rank;
    if ((uint64_t)end_step > max_steps) end_step = (int)max_steps;
    
    int batch_count = 0;
    Load256(buf_X[0], accX); Load256(buf_Y[0], accY); Load256(buf_Z[0], accZ);
    buf_masks[0] = full_mask;
    batch_count = 1; steps_done = 1;

    // Using your reference sparse architecture packing
    uint32_t packed_walk_id = (qi_idx << 24) | (walk_chunk_id & 0xFFFFFF);
    
    while (steps_done < end_step) {
        uint64_t old_mask = mask;
        mask = gosper_next(mask);

        // Break if we overflow out of L_bits
        if ((mask & ~valid_mask) || mask == 0) break;

        uint64_t old_seed = ((qi_mask | old_mask) ^ d_targetSeedLo) & d_seedMaskLo;
        uint64_t new_seed = ((qi_mask | mask) ^ d_targetSeedLo) & d_seedMaskLo;

        apply_xor_diff(accX, accY, accZ, old_seed, new_seed);

        full_mask = qi_mask | mask;
        Load256(buf_X[batch_count], accX); Load256(buf_Y[batch_count], accY); Load256(buf_Z[batch_count], accZ);
        buf_masks[batch_count] = full_mask;
        batch_count++; steps_done++;

        if (batch_count >= MAX_BATCH) {
            batch_invert_Z(buf_Z, Zinv, batch_count);
            for (int b = 0; b < batch_count; b++) {
                uint64_t s_check = (buf_masks[b] ^ d_targetSeedLo) & d_seedMaskLo;
                int pc_abs = __popcll(s_check) + d_lockedPopcount;
                if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) continue;

                uint64_t Zinv_sq[4], px[4], py[4], Zinv_cb[4];
                _ModSqr(Zinv_sq, Zinv[b]); _ModMult(px, Zinv_sq, buf_X[b]);
                _ModMult(Zinv_cb, Zinv_sq, Zinv[b]); _ModMult(py, Zinv_cb, buf_Y[b]);
                uint8_t odd_py = (uint8_t)(py[0] & 1);
                uint32_t h[5];
                _GetHash160Comp(px, odd_py, (uint8_t*)h);

                if (sAddress[h[0] & 0xFFFF] != 0) {
                    uint32_t step_idx = steps_done - batch_count + b;
                    CheckPointWalk(h, packed_walk_id, step_idx, sAddress, lookup32, out);
                }
            }
            if (steps_done < end_step) {
                uint64_t Zinv_sq[4], Zinv_cb[4];
                int last = batch_count - 1;
                _ModSqr(Zinv_sq, Zinv[last]); _ModMult(accX, Zinv_sq, buf_X[last]);
                _ModMult(Zinv_cb, Zinv_sq, Zinv[last]); _ModMult(accY, Zinv_cb, buf_Y[last]);
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
            }
            batch_count = 0;
        }
    }

    if (batch_count > 0) {
        batch_invert_Z(buf_Z, Zinv, batch_count);
        for (int b = 0; b < batch_count; b++) {
            uint64_t s_check = (buf_masks[b] ^ d_targetSeedLo) & d_seedMaskLo;
            int pc_abs = __popcll(s_check) + d_lockedPopcount;
            if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) continue;

            uint64_t Zinv_sq[4], px[4], py[4], Zinv_cb[4];
            _ModSqr(Zinv_sq, Zinv[b]); _ModMult(px, Zinv_sq, buf_X[b]);
            _ModMult(Zinv_cb, Zinv_sq, Zinv[b]); _ModMult(py, Zinv_cb, buf_Y[b]);
            uint8_t odd_py = (uint8_t)(py[0] & 1);
            uint32_t h[5];
            _GetHash160Comp(px, odd_py, (uint8_t*)h);

            if (sAddress[h[0] & 0xFFFF] != 0) {
                uint32_t step_idx = steps_done - batch_count + b;
                CheckPointWalk(h, packed_walk_id, step_idx, sAddress, lookup32, out);
            }
        }
    }
}


// =====================================================================================
// Host-side: G_free table computation and upload
// =====================================================================================

bool GPUEngine::ComputeGfreeTables(Secp256K1* secp, StringCrackConfig* config) {
    int n = config->numFreeBits;
    size_t tableSize = n * 4 * sizeof(uint64_t);
    
    uint64_t* h_GfreeX    = (uint64_t*)calloc(n * 4, sizeof(uint64_t));
    uint64_t* h_GfreeY    = (uint64_t*)calloc(n * 4, sizeof(uint64_t));
    uint64_t* h_negGfreeY = (uint64_t*)calloc(n * 4, sizeof(uint64_t));
    
    if (!h_GfreeX || !h_GfreeY || !h_negGfreeY) {
        printf("[SEP7] G_free allocation failed\n");
        return false;
    }
    
    for (int i = 0; i < n; i++) {
        int pos = config->freeBitPositions[i];
        
        Int key;
        key.SetInt32(0);
        key.bits64[pos >> 6] |= (1ULL << (pos & 63));
        Point P = secp->ComputePublicKey(&key);
        
        int idx = i * 4;
        memcpy(&h_GfreeX[idx],    P.x.bits64, 32);
        memcpy(&h_GfreeY[idx],    P.y.bits64, 32);
        
        // Negate Y for subtraction
        P.y.ModNeg();
        memcpy(&h_negGfreeY[idx], P.y.bits64, 32);
    }
    
    uint64_t *dd_X, *dd_Y, *dd_nY;
    cudaError_t err;
    
    err = cudaMalloc((void**)&dd_X,  tableSize); if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&dd_Y,  tableSize); if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&dd_nY, tableSize); if (err != cudaSuccess) goto fail;
    
    cudaMemcpy(dd_X,  h_GfreeX,    tableSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dd_Y,  h_GfreeY,    tableSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dd_nY, h_negGfreeY, tableSize, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(d_GfreeX,    &dd_X,  sizeof(uint64_t*));
    cudaMemcpyToSymbol(d_GfreeY,    &dd_Y,  sizeof(uint64_t*));
    cudaMemcpyToSymbol(d_negGfreeY, &dd_nY, sizeof(uint64_t*));
    
    printf("[SEP7] G_free tables: %d points (%.1f KB)\n", n, (float)(tableSize * 3) / 1024.0f);
    fflush(stdout);
    
    free(h_GfreeX); free(h_GfreeY); free(h_negGfreeY);
    return true;

fail:
    printf("[SEP7] G_free GPU allocation failed\n");
    free(h_GfreeX); free(h_GfreeY); free(h_negGfreeY);
    return false;
}


// =====================================================================================
// Host-side: Dispatcher
// =====================================================================================

void GPUEngine::LaunchGosperWalkAsync(int hamming_h, uint64_t base_rank_offset,
                                       uint64_t totalCombs, int chunk_size, int numWalks) {
    int s = currentStep % 2;
    cudaMemsetAsync(d_output[s], 0, 4, streams[s]);
    
    // SEP7: Grid sized for walks, NOT for stateless kernel
    // numWalks is computed by the host dispatch loop based on SM count
    int threadsPerBlock = 32;  // 1 warp = 1 block
    int numBlocks = (numWalks + threadsPerBlock - 1) / threadsPerBlock;
    
    comp_keys_gosper_walk<BATCH_N><<<numBlocks, threadsPerBlock, 0, streams[s]>>>(
        inputAddress, inputAddressLookUp, d_output[s],
        hamming_h, base_rank_offset, totalCombs, chunk_size);
    
    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize,
                    cudaMemcpyDeviceToHost, streams[s]);
    currentStep++;
}


// =====================================================================================
// Host-side: Key Reconstruction on Hit
// =====================================================================================

void VanitySearch::reconstructGosperWalkKey(
    uint32_t walk_id, uint32_t step,
    int hamming_h, uint64_t base_rank_offset, int chunk_size,
    uint8_t* hash, StringCrackConfig* config,
    const uint64_t* h_combTable, int tableK)
{
    int n = config->numFreeBits;
    
    // 1. Starting rank for this walk
    uint64_t start_rank = base_rank_offset + (uint64_t)walk_id * (uint64_t)chunk_size;
    
    // 2. Unrank initial combination
    uint64_t mask = 0;
    {
        uint64_t rank = start_rank;
        int remaining = hamming_h;
        for (int i = n - 1; i >= 0 && remaining > 0; i--) {
            uint64_t c = h_combTable[i * tableK + remaining];
            if (rank >= c) {
                rank -= c;
                mask |= (1ULL << i);
                remaining--;
            }
        }
    }
    
    // 3. Advance by 'step' Gosper transitions
    for (uint32_t s = 0; s < step; s++) {
        uint64_t c = mask & (0ULL - mask);
        uint64_t r = mask + c;
        int shift = __builtin_ctzll(mask) + 2;
        mask = ((r ^ mask) >> shift) | r;
    }
    
    // 4. XOR with center to get seed
    uint64_t seed_lo = (mask ^ config->targetSeedLo) & config->seedMaskLo;
    
    // 5. Expand to full 256-bit key
    uint64_t keyBits[4] = {
        config->lockVals[0], config->lockVals[1],
        config->lockVals[2], config->lockVals[3]
    };
    
    uint64_t sl = seed_lo;
    for (int fb = 0; fb < n && fb < 64; fb++) {
        if (sl & 1ULL) {
            int pos = config->freeBitPositions[fb];
            keyBits[pos >> 6] |= (1ULL << (pos & 63));
        }
        sl >>= 1;
    }
    
    // 6. Verify and output
    Int privkey;
    privkey.SetInt32(0);
    privkey.bits64[0] = keyBits[0];
    privkey.bits64[1] = keyBits[1];
    privkey.bits64[2] = keyBits[2];
    privkey.bits64[3] = keyBits[3];
    
    checkAddr(*(address_t*)(hash), hash, privkey, 0, 0, true);
}
