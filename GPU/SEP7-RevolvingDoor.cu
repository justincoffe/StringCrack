// =====================================================================================
// SEP7: Revolving Door EC Walker — CORRECTED
// =====================================================================================
//
// Architecture: 1 Thread = 1 Independent Walk. Pure SIMT. 64-bit free-bit limit.
//
// CRITICAL FIX over previous version:
//   The previous revdoor_init + advance_to_first_leaf was BROKEN for parallelization.
//   It always started every thread at position 0 of the revolving door sequence,
//   regardless of start_rank. The lex-order unrank was completely disconnected from
//   the revolving-door ordering.
//
//   This version implements revdoor_unrank(): an O(n) algorithm that, given a
//   POSITION in the revolving-door sequence, simultaneously constructs the correct
//   c[] array AND the correct recursion stack. This enables true parallel walks
//   where each thread covers a disjoint chunk of the revolving-door sequence.
//
// Core innovation: Revolving Door combination listing (Nijenhuis & Wilf, 1978)
//   guarantees EXACTLY ONE element swap per step → ZERO warp divergence on EC math.
//   With a precomputed D-table D[i][j] = G_free[j] - G_free[i], each transition
//   is a SINGLE EC addition.
//
// Cost model (BATCH_N=8):
//   Per step: 10.4 (1 EC add) + 9.5 (amortized inv) + ~6 (filtered affine+hash) + 1 (mgmt)
//   Total: ~27 ModMult-equiv per candidate
//   vs SEP6 Gosper: ~140 per candidate
//   vs SEP7 GosperWalk: ~84 effective (due to warp divergence)
//   Predicted: ~1300+ MK/s on 5080
//
// =====================================================================================

#ifndef SEP7_REVOLVING_DOOR_CU
#define SEP7_REVOLVING_DOOR_CU

#ifndef BATCH_N
#define BATCH_N 8   // Candidates buffered per thread before batch ModInv
#endif

// Maximum recursion depth for the revolving door state machine.
// Bounded by n (num free bits). 66 allows n up to 64 with margin.
#define RD_MAX_DEPTH 66

// Maximum k (Hamming weight). Must be <= n.
#define RD_MAX_K 48

// =====================================================================================
// D-Table: Precomputed EC point differences for single-swap transitions
//
// D[i][j] = G_free[j] - G_free[i]  (as affine point)
//
// For a revolving door step that removes free-bit i and adds free-bit j:
//   next_point = current_point + D[i][j]
//
// Storage: n × n × 64 bytes. For n=59: 59×59×64 = ~218 KB. Fits in L2 cache.
// =====================================================================================

__device__ uint64_t* d_DTableX;   // [n * n * 4]  flattened
__device__ uint64_t* d_DTableY;   // [n * n * 4]  flattened

// Device-side C(n,k) table for revolving-door unranking.
// Layout: row-major, d_rdCombTable[i * d_rdCombK + j] = C(i, j)
// Dimensions: (n+1) × (K_max+1), uploaded by host.
__device__ uint64_t* d_rdCombTable;
__device__ int       d_rdCombStride;  // = K_max + 1 (row stride)

// =====================================================================================
// Revolving Door State Machine — Stack Frame
// =====================================================================================

struct RDFrame {
    uint8_t n;       // universe size at this recursion level
    uint8_t k;       // combination size at this recursion level
    uint8_t phase;   // 0=first sub-call, 1=transition, 2=second sub-call, 3=done/pop
    uint8_t is_neg;  // 0 = GEN (forward), 1 = NEG (reverse)
};

// =====================================================================================
// Device-side C(n,k) lookup via read-only cache
// =====================================================================================
__device__ __forceinline__ uint64_t rd_comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    return __ldg(&d_rdCombTable[n * d_rdCombStride + k]);
}

// =====================================================================================
// revdoor_unrank: O(n) positional unranking for the revolving-door sequence.
//
// Given position `pos` in the revolving-door ordering of C(n, k),
// simultaneously constructs:
//   - c[0..k-1]: the combination at that position (exact array state matching
//                 what the state machine expects)
//   - c[k] = n:  sentinel
//   - stk[0..sp]: the recursion stack (so revdoor_step can continue from here)
//
// The recursion structure:
//   GEN(n, k) = GEN(n-1, k) [C(n-1,k) items] | transition | NEG(n-1, k-1) [C(n-1,k-1) items]
//   NEG(n, k) = GEN(n-1, k-1) [C(n-1,k-1) items] | transition | NEG(n-1, k) [C(n-1,k) items]
//
// At each level we compare `pos` against the boundary to determine which half
// we're in. If second half (phase=2), we apply the transition's effect on c[].
//
// Key insight for c[] reconstruction:
//   first(GEN(n, k)) in c[0..k-1] = {0, 1, ..., k-1}
//   first(NEG(n, k)) in c[0..k-1] = {0, 1, ..., k-2, n-1}
//   last(GEN(n, k))  in c[0..k-1] = {0, 1, ..., k-2, n-1}
//   last(NEG(n, k))  in c[0..k-1] = {0, 1, ..., k-1}
//
// After a GEN(n, k) transition:  c[k-2]=n-2, c[k-1]=n-1 (k>=2) or c[0]=n-1 (k=1)
// After a NEG(n, k) transition:  c[k-2]=k-2, c[k-1]=n-2 (k>=2) or c[0]=n-2 (k=1)
// =====================================================================================

__device__ void revdoor_unrank(
    int N, int K, uint64_t pos,
    int c[],            // output: combination array, c[K] = N sentinel
    RDFrame stk[],      // output: recursion stack
    int *sp)            // output: stack pointer (points to top frame)
{
    // Initialize c[] to the identity combination (first of GEN(N, K))
    for (int i = 0; i < K; i++) c[i] = i;
    c[K] = N;

    int n = N, k = K;
    int is_neg = 0;
    *sp = -1;

    while (k > 0 && k < n) {
        (*sp)++;

        if (!is_neg) {
            // ─── GEN(n, k) ───
            uint64_t boundary = rd_comb(n - 1, k);

            if (pos < boundary) {
                // First half: descend into GEN(n-1, k)
                stk[*sp].n = (uint8_t)n;
                stk[*sp].k = (uint8_t)k;
                stk[*sp].phase = 0;
                stk[*sp].is_neg = 0;
                // c[] is correct: at entry, c[0..k-1] = first(GEN(n,k)) = identity
                // first(GEN(n-1, k)) = same identity → no change needed
                n--;
                // k, is_neg unchanged
            } else {
                // Second half: past transition, descend into NEG(n-1, k-1)
                stk[*sp].n = (uint8_t)n;
                stk[*sp].k = (uint8_t)k;
                stk[*sp].phase = 2;
                stk[*sp].is_neg = 0;
                pos -= boundary;

                // Set c[] to first(NEG(n-1, k-1)) after GEN(n,k) transition:
                if (k == 1) {
                    // GEN k=1 transition: c[0] = n-1
                    c[0] = n - 1;
                    // Descend into NEG(n-1, 0) → base case
                    n--;
                    k = 0;
                    is_neg = 1;
                } else {
                    // GEN k>=2 transition: c[k-2]=n-2, c[k-1]=n-1
                    // first(NEG(n-1, k-1)) in c[0..k-2]: {0,...,k-3, n-2}
                    for (int i = 0; i <= k - 3; i++) c[i] = i;
                    c[k - 2] = n - 2;
                    c[k - 1] = n - 1;
                    n--;
                    k--;
                    is_neg = 1;
                }
            }
        } else {
            // ─── NEG(n, k) ───
            uint64_t boundary = rd_comb(n - 1, k - 1);

            if (pos < boundary) {
                // First half: descend into GEN(n-1, k-1)
                stk[*sp].n = (uint8_t)n;
                stk[*sp].k = (uint8_t)k;
                stk[*sp].phase = 0;
                stk[*sp].is_neg = 1;
                // NEG(n, k) starts with GEN(n-1, k-1) operating on c[0..k-2]
                // first(GEN(n-1, k-1)) = {0,...,k-2} in c[0..k-2]
                // Ensure c[0..k-2] is identity (may have been set by parent)
                for (int i = 0; i <= k - 2; i++) c[i] = i;
                // c[k-1] stays at whatever the parent set (correct)
                n--;
                k--;
                is_neg = 0;
            } else {
                // Second half: past transition, descend into NEG(n-1, k)
                stk[*sp].n = (uint8_t)n;
                stk[*sp].k = (uint8_t)k;
                stk[*sp].phase = 2;
                stk[*sp].is_neg = 1;
                pos -= boundary;

                // Set c[] to first(NEG(n-1, k)) after NEG(n,k) transition:
                if (k == 1) {
                    // NEG k=1 transition: c[0] = n-2
                    c[0] = n - 2;
                    n--;
                    // k stays 1, is_neg stays 1
                } else {
                    // NEG k>=2 transition: c[k-2]=k-2, c[k-1]=n-2
                    // first(NEG(n-1, k)) in c[0..k-1]: {0,...,k-2, n-2}
                    for (int i = 0; i <= k - 2; i++) c[i] = i;
                    c[k - 1] = n - 2;
                    n--;
                    // k stays, is_neg stays 1
                }
            }
        }
    }

    // Base case: k==0 or k==n
    if (k == n) {
        for (int i = 0; i < k; i++) c[i] = i;
    }
    // k==0: nothing to set

    (*sp)++;
    stk[*sp].n = (uint8_t)n;
    stk[*sp].k = (uint8_t)k;
    stk[*sp].phase = 3; // base case → will pop on first step
    stk[*sp].is_neg = (uint8_t)is_neg;
}

// =====================================================================================
// revdoor_step: Advance the revolving door by one step.
//
// Returns true if a new combination was reached.
// On success: c[] is updated, *removed and *added contain the free-bit indices
// of the swapped elements (for D-table lookup and mask maintenance).
//
// Amortized O(1) per step. The while loop handles stack unwinding (cheap integer
// ops only — no 256-bit EC math). Zero warp divergence on the expensive EC path.
// =====================================================================================

__device__ bool revdoor_step(
    int c[],            // combination array (modified in place)
    RDFrame stk[],      // recursion stack (modified in place)
    int *sp,            // stack pointer (modified in place)
    int K,              // global combination size (constant, unused but kept for API compat)
    int *removed,       // output: free-bit index of removed element
    int *added)         // output: free-bit index of added element
{
    while (*sp >= 0) {
        RDFrame *f = &stk[*sp];

        // Base case: single-combination level — pop and advance parent
        if (f->k == 0 || f->k == f->n) {
            (*sp)--;
            if (*sp >= 0) stk[*sp].phase++;
            continue;
        }

        switch (f->phase) {

        case 0: {
            // Push first sub-call
            int child_sp = *sp + 1;
            if (!f->is_neg) {
                // GEN(n,k) → GEN(n-1, k)
                stk[child_sp].n = f->n - 1;
                stk[child_sp].k = f->k;
                stk[child_sp].phase = 0;
                stk[child_sp].is_neg = 0;
            } else {
                // NEG(n,k) → GEN(n-1, k-1)
                stk[child_sp].n = f->n - 1;
                stk[child_sp].k = f->k - 1;
                stk[child_sp].phase = 0;
                stk[child_sp].is_neg = 0;
            }
            *sp = child_sp;
            break;
        }

        case 1: {
            // ─── TRANSITION (the heart of the revolving door) ───
            if (!f->is_neg) {
                // GEN(n, k) transition
                if (f->k == 1) {
                    *removed = c[0];
                    c[0] = (int)(f->n - 1);
                    *added = c[0];
                } else {
                    *removed = c[f->k - 2];
                    c[f->k - 2] = c[f->k - 1];
                    c[f->k - 1] = (int)(f->n - 1);
                    *added = (int)(f->n - 1);
                }
            } else {
                // NEG(n, k) transition
                if (f->k == 1) {
                    *removed = c[0];
                    c[0] = (int)(f->n - 2);
                    *added = c[0];
                } else {
                    *removed = c[f->k - 1];
                    *added = (int)(f->k - 2);
                    c[f->k - 1] = c[f->k - 2];
                    c[f->k - 2] = (int)(f->k - 2);
                }
            }

            f->phase = 2;
            return true; // Combination changed by exactly one swap
        }

        case 2: {
            // Push second sub-call
            int child_sp = *sp + 1;
            if (!f->is_neg) {
                // GEN(n,k) → NEG(n-1, k-1)
                stk[child_sp].n = f->n - 1;
                stk[child_sp].k = f->k - 1;
                stk[child_sp].phase = 0;
                stk[child_sp].is_neg = 1;
            } else {
                // NEG(n,k) → NEG(n-1, k)
                stk[child_sp].n = f->n - 1;
                stk[child_sp].k = f->k;
                stk[child_sp].phase = 0;
                stk[child_sp].is_neg = 1;
            }
            *sp = child_sp;
            break;
        }

        case 3: {
            // Done with this level — pop
            (*sp)--;
            if (*sp >= 0) stk[*sp].phase++;
            break;
        }

        } // switch
    }

    return false; // exhausted all C(n, k) combinations
}

// =====================================================================================
// Thread-local Montgomery batch inversion (identical to SEP7-GW-v2)
// =====================================================================================
__device__ void rd_batch_invert_Z(
    uint64_t Z_buf[][4],
    uint64_t Zinv_buf[][4],
    int count)
{
    if (count == 0) return;
    if (count == 1) {
        uint64_t tmp[5];
        Load256(tmp, Z_buf[0]);
        tmp[4] = 0;
        _ModInv(tmp);
        Load256(Zinv_buf[0], tmp);
        return;
    }

    uint64_t prefix[BATCH_N][4];
    Load256(prefix[0], Z_buf[0]);
    for (int i = 1; i < count; i++) {
        _ModMult(prefix[i], prefix[i-1], Z_buf[i]);
    }

    uint64_t total_inv[5];
    Load256(total_inv, prefix[count - 1]);
    total_inv[4] = 0;
    _ModInv(total_inv);

    for (int i = count - 1; i >= 1; i--) {
        _ModMult(Zinv_buf[i], prefix[i-1], (uint64_t*)total_inv);
        uint64_t t[4];
        _ModMult(t, (uint64_t*)total_inv, Z_buf[i]);
        Load256(total_inv, t);
    }
    Load256(Zinv_buf[0], total_inv);
}

// =====================================================================================
// D-Table lookup: Load D[removed][added] via read-only texture cache
// =====================================================================================
__device__ __forceinline__ void load_Dtable(
    int removed, int added, int n,
    uint64_t dx[4], uint64_t dy[4])
{
    int idx = (removed * n + added) * 4;
    ulonglong2 vx_lo = __ldg((ulonglong2*)&d_DTableX[idx]);
    ulonglong2 vx_hi = __ldg((ulonglong2*)&d_DTableX[idx + 2]);
    ulonglong2 vy_lo = __ldg((ulonglong2*)&d_DTableY[idx]);
    ulonglong2 vy_hi = __ldg((ulonglong2*)&d_DTableY[idx + 2]);
    dx[0] = vx_lo.x; dx[1] = vx_lo.y; dx[2] = vx_hi.x; dx[3] = vx_hi.y;
    dy[0] = vy_lo.x; dy[1] = vy_lo.y; dy[2] = vy_hi.x; dy[3] = vy_hi.y;
}

// =====================================================================================
// Bitmask from combination array (for initial EC point computation)
// =====================================================================================
__device__ __forceinline__ uint64_t combo_to_mask(const int c[], int k) {
    uint64_t mask = 0;
    for (int i = 0; i < k; i++) {
        mask |= (1ULL << c[i]);
    }
    return mask;
}

// =====================================================================================
// THE KERNEL: Revolving Door EC Walker
//
// Each thread = one independent walk through a chunk of C(n, hamming_h).
//
// PARALLELIZATION:
//   The host partitions the revolving-door sequence [0, C(n,h)) into chunks.
//   Thread walk_id covers positions [walk_id * chunk_size, (walk_id+1) * chunk_size).
//   revdoor_unrank jumps each thread to its starting position in O(n) time,
//   then revdoor_step generates subsequent positions in amortized O(1) each.
//
// WARP DIVERGENCE: The EC addition path is perfectly uniform (always exactly 1
//   jacobian_add_affine per step). The revdoor_step function has variable-length
//   stack unwinding, but that's ~20 clocks of integer work, not 256-bit EC math.
// =====================================================================================

template <int MAX_BATCH>
__global__ __launch_bounds__(32, 14)
void comp_keys_revdoor(
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    int hamming_h,
    uint64_t base_pos,          // starting POSITION in revolving-door order (not lex rank)
    uint64_t totalCombs,        // C(n, hamming_h)
    int chunk_size)
{
    int lane_id = threadIdx.x;
    uint32_t walk_id = blockIdx.x * 32 + lane_id;

    // This thread's starting position in the revolving-door sequence
    uint64_t start_pos = base_pos + (uint64_t)walk_id * (uint64_t)chunk_size;
    if (start_pos >= totalCombs) return;

    int n = d_numFreeBits;
    if (n > 64) return;

    // ═══════ REVOLVING DOOR INITIALIZATION via O(n) unranking ═══════

    int c[RD_MAX_K + 1];
    RDFrame stk[RD_MAX_DEPTH];
    int sp;

    revdoor_unrank(n, hamming_h, start_pos, c, stk, &sp);

    // Build bitmask from the unranked combination
    uint64_t mask = combo_to_mask(c, hamming_h);

    // ═══════ COMPUTE INITIAL EC POINT (window method, same as SEP6/SEP7) ═══════

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

    // ═══════ BATCHED REVOLVING DOOR WALK LOOP ═══════

    uint64_t buf_X[MAX_BATCH][4];
    uint64_t buf_Y[MAX_BATCH][4];
    uint64_t buf_Z[MAX_BATCH][4];
    uint64_t buf_masks[MAX_BATCH];
    uint64_t Zinv[MAX_BATCH][4];

    int steps_done = 0;
    int end_step = chunk_size;

    // Clamp to totalCombs boundary
    uint64_t max_steps = totalCombs - start_pos;
    if ((uint64_t)end_step > max_steps) end_step = (int)max_steps;

    // Store initial point as batch entry 0
    int batch_count = 0;
    Load256(buf_X[0], accX);
    Load256(buf_Y[0], accY);
    Load256(buf_Z[0], accZ);
    buf_masks[0] = mask;
    batch_count = 1;
    steps_done = 1; // step 0 = initial combination (from unranking)

    while (steps_done < end_step) {

        // ─── REVOLVING DOOR STEP: exactly one swap ───
        int removed_idx, added_idx;
        if (!revdoor_step(c, stk, &sp, hamming_h, &removed_idx, &added_idx)) {
            break; // exhausted this walk's portion of C(n, h)
        }

        // Update bitmask
        mask = (mask & ~(1ULL << removed_idx)) | (1ULL << added_idx);

        // ─── SINGLE EC ADDITION via D-table ───
        // This is the ONLY EC operation per step. Perfectly uniform across the warp.
        uint64_t dX[4], dY[4];
        load_Dtable(removed_idx, added_idx, n, dX, dY);
        jacobian_add_affine_inplace(accX, accY, accZ, dX, dY);

        // Buffer this step
        Load256(buf_X[batch_count], accX);
        Load256(buf_Y[batch_count], accY);
        Load256(buf_Z[batch_count], accZ);
        buf_masks[batch_count] = mask;
        batch_count++;
        steps_done++;

        // ─── FLUSH BATCH when full ───
        if (batch_count >= MAX_BATCH) {

            rd_batch_invert_Z(buf_Z, Zinv, batch_count);

            for (int b = 0; b < batch_count; b++) {
                // Popcount pre-filter
                uint64_t seed_check = (buf_masks[b] ^ d_targetSeedLo) & d_seedMaskLo;
                int pc_abs = __popcll(seed_check) + d_lockedPopcount;
                if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) continue;

                // Affine conversion
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
                    uint32_t pos = atomicAdd(out, 1);
                    if (pos < 65536) {
                        uint32_t* item = out + 1 + pos * ITEM_SIZE32;
                        item[0] = walk_id;
                        int16_t* ptr = (int16_t*)&item[1];
                        ptr[0] = (int16_t)(step_idx & 0x7FFF);
                        ptr[1] = (int16_t)((step_idx >> 15) & 0x7FFF);
                        memcpy(item + 2, h, 20);
                    }
                }
            }

            // Reset accumulator to affine from the LAST batch entry
            if (steps_done < end_step) {
                int last = batch_count - 1;
                uint64_t Zinv_sq[4], Zinv_cb[4];
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
        rd_batch_invert_Z(buf_Z, Zinv, batch_count);

        for (int b = 0; b < batch_count; b++) {
            uint64_t seed_check = (buf_masks[b] ^ d_targetSeedLo) & d_seedMaskLo;
            int pc_abs = __popcll(seed_check) + d_lockedPopcount;
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
                uint32_t pos = atomicAdd(out, 1);
                if (pos < 65536) {
                    uint32_t* item = out + 1 + pos * ITEM_SIZE32;
                    item[0] = walk_id;
                    int16_t* ptr = (int16_t*)&item[1];
                    ptr[0] = (int16_t)(step_idx & 0x7FFF);
                    ptr[1] = (int16_t)((step_idx >> 15) & 0x7FFF);
                    memcpy(item + 2, h, 20);
                }
            }
        }
    }
}


// =====================================================================================
// Host-side: D-Table Computation and Upload
// =====================================================================================

bool GPUEngine::ComputeDTable(Secp256K1* secp, StringCrackConfig* config) {
    int n = config->numFreeBits;
    if (n <= 0 || n > 64) {
        printf("[SEP7-RD] D-Table: invalid numFreeBits %d (must be 1..64)\n", n);
        return false;
    }

    size_t tableEntries = (size_t)n * n * 4;
    size_t tableBytes   = tableEntries * sizeof(uint64_t);

    printf("[SEP7-RD] Computing D-Table: %d x %d x 64 bytes = %.1f KB\n",
           n, n, (float)tableBytes / 1024.0f);
    fflush(stdout);

    uint64_t* h_DX = (uint64_t*)calloc(tableEntries, sizeof(uint64_t));
    uint64_t* h_DY = (uint64_t*)calloc(tableEntries, sizeof(uint64_t));
    if (!h_DX || !h_DY) {
        printf("[SEP7-RD] D-Table host allocation failed\n");
        free(h_DX); free(h_DY);
        return false;
    }

    // Precompute G_free[i] for each free bit
    Point* Gfree = new Point[n];
    for (int i = 0; i < n; i++) {
        int pos = config->freeBitPositions[i];
        Int key;
        key.SetInt32(0);
        key.bits64[pos >> 6] |= (1ULL << (pos & 63));
        Gfree[i] = secp->ComputePublicKey(&key);
    }

    // D[i][j] = G_free[j] + (-G_free[i])
    #pragma omp parallel for schedule(dynamic) if(n > 16)
    for (int i = 0; i < n; i++) {
        Point negGi = Gfree[i];
        negGi.y.ModNeg();

        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            Point D = secp->AddDirect(Gfree[j], negGi);
            int idx = (i * n + j) * 4;
            memcpy(&h_DX[idx], D.x.bits64, 32);
            memcpy(&h_DY[idx], D.y.bits64, 32);
        }
    }

    delete[] Gfree;

    // Upload to GPU
    uint64_t *dd_DX = nullptr, *dd_DY = nullptr;
    cudaError_t err;

    uint64_t* old_DX = nullptr;
    uint64_t* old_DY = nullptr;
    cudaMemcpyFromSymbol(&old_DX, d_DTableX, sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&old_DY, d_DTableY, sizeof(uint64_t*));
    if (old_DX) cudaFree(old_DX);
    if (old_DY) cudaFree(old_DY);

    err = cudaMalloc((void**)&dd_DX, tableBytes);
    if (err != cudaSuccess) { printf("[SEP7-RD] D-Table GPU alloc DX: %s\n", cudaGetErrorString(err)); goto fail; }

    err = cudaMalloc((void**)&dd_DY, tableBytes);
    if (err != cudaSuccess) { printf("[SEP7-RD] D-Table GPU alloc DY: %s\n", cudaGetErrorString(err)); cudaFree(dd_DX); goto fail; }

    cudaMemcpy(dd_DX, h_DX, tableBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dd_DY, h_DY, tableBytes, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_DTableX, &dd_DX, sizeof(uint64_t*));
    cudaMemcpyToSymbol(d_DTableY, &dd_DY, sizeof(uint64_t*));

    printf("[SEP7-RD] D-Table uploaded: %d x %d = %d entries (%.1f KB)\n",
           n, n, n*n, (float)(tableBytes * 2) / 1024.0f);
    fflush(stdout);

    free(h_DX);
    free(h_DY);

    // ─── Also upload the C(n,k) table for device-side unranking ───
    {
        int maxN = n + 1;   // rows: 0..n
        int maxK = n + 1;   // cols: 0..n (k can be up to n)
        size_t combEntries = (size_t)maxN * maxK;
        size_t combBytes   = combEntries * sizeof(uint64_t);

        uint64_t* h_comb = (uint64_t*)calloc(combEntries, sizeof(uint64_t));
        if (!h_comb) {
            printf("[SEP7-RD] C(n,k) table host alloc failed\n");
            return false;
        }

        // Pascal's triangle
        for (int i = 0; i < maxN; i++) {
            h_comb[i * maxK + 0] = 1;
            for (int j = 1; j <= i && j < maxK; j++) {
                uint64_t a = h_comb[(i - 1) * maxK + (j - 1)];
                uint64_t b = h_comb[(i - 1) * maxK + j];
                h_comb[i * maxK + j] = (a > 0xFFFFFFFFFFFFFFFFULL - b)
                                       ? 0xFFFFFFFFFFFFFFFFULL : a + b;
            }
        }

        uint64_t* dd_comb = nullptr;
        err = cudaMalloc((void**)&dd_comb, combBytes);
        if (err != cudaSuccess) {
            printf("[SEP7-RD] C(n,k) GPU alloc: %s\n", cudaGetErrorString(err));
            free(h_comb);
            return false;
        }

        cudaMemcpy(dd_comb, h_comb, combBytes, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_rdCombTable, &dd_comb, sizeof(uint64_t*));

        int stride = maxK;
        cudaMemcpyToSymbol(d_rdCombStride, &stride, sizeof(int));

        printf("[SEP7-RD] C(n,k) table uploaded: %d x %d (%.1f KB)\n",
               maxN, maxK, (float)combBytes / 1024.0f);
        fflush(stdout);

        free(h_comb);
    }

    return true;

fail:
    printf("[SEP7-RD] D-Table GPU upload failed\n");
    free(h_DX);
    free(h_DY);
    return false;
}


// =====================================================================================
// Host-side: Kernel Dispatcher
//
// NOTE: base_pos is now a POSITION in the revolving-door sequence (0-based),
// NOT a lexicographic rank. The host dispatch loop manages positions sequentially.
// =====================================================================================

void GPUEngine::LaunchRevDoorAsync(int hamming_h, uint64_t base_pos,
                                    uint64_t totalCombs, int chunk_size, int numWalks) {
    int s = currentStep % 2;
    cudaMemsetAsync(d_output[s], 0, 4, streams[s]);

    int threadsPerBlock = 32;
    // THE FIX: Use numWalks instead of nbThread
    int numBlocks = (numWalks + threadsPerBlock - 1) / threadsPerBlock;

    comp_keys_revdoor<BATCH_N><<<numBlocks, threadsPerBlock, 0, streams[s]>>>(
        inputAddress, inputAddressLookUp, d_output[s],
        hamming_h, base_pos, totalCombs, chunk_size);

    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize,
                    cudaMemcpyDeviceToHost, streams[s]);
    currentStep++;
}

uint32_t GPUEngine::SyncRevDoorBatch(int stepToSync, std::vector<ITEM> &addressFound) {
    int s = stepToSync % 2;
    cudaStreamSynchronize(streams[s]);

    uint32_t nbFound = h_outputPinned[s][0];
    if (nbFound > maxFound) nbFound = maxFound;

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
// Host-side: Key Reconstruction on Hit
//
// Replays the revolving door walk on CPU from position
// base_pos + walk_id * chunk_size, advancing step_idx steps.
// =====================================================================================

void VanitySearch::reconstructRevDoorKey(
    uint32_t walk_id, uint32_t step_idx,
    int hamming_h, uint64_t base_pos, int chunk_size,
    uint8_t* hash, StringCrackConfig* config,
    const uint64_t* h_combTable, int tableK)
{
    int n = config->numFreeBits;

    // 1. Compute this walk's starting position (revolving-door order)
    uint64_t start_pos = base_pos + (uint64_t)walk_id * (uint64_t)chunk_size;

    // 2. CPU version of revdoor_unrank
    // We need a host-side C(n,k) lookup. Use h_combTable.
    // h_combTable[i * tableK + j] = C(i, j)

    int c[RD_MAX_K + 1];
    for (int i = 0; i < hamming_h; i++) c[i] = i;
    c[hamming_h] = n;

    struct CPURDFrame {
        int n, k, phase;
        bool is_neg;
    };

    CPURDFrame stk[RD_MAX_DEPTH];
    int sp = -1;

    {
        int ln = n, lk = hamming_h;
        bool l_neg = false;
        uint64_t pos = start_pos;

        while (lk > 0 && lk < ln) {
            sp++;

            if (!l_neg) {
                uint64_t boundary = h_combTable[(ln - 1) * tableK + lk];
                if (pos < boundary) {
                    stk[sp] = {ln, lk, 0, false};
                    ln--;
                } else {
                    stk[sp] = {ln, lk, 2, false};
                    pos -= boundary;
                    if (lk == 1) {
                        c[0] = ln - 1;
                        ln--; lk = 0; l_neg = true;
                    } else {
                        for (int i = 0; i <= lk - 3; i++) c[i] = i;
                        c[lk - 2] = ln - 2;
                        c[lk - 1] = ln - 1;
                        ln--; lk--; l_neg = true;
                    }
                }
            } else {
                uint64_t boundary = h_combTable[(ln - 1) * tableK + (lk - 1)];
                if (pos < boundary) {
                    stk[sp] = {ln, lk, 0, true};
                    for (int i = 0; i <= lk - 2; i++) c[i] = i;
                    ln--; lk--; l_neg = false;
                } else {
                    stk[sp] = {ln, lk, 2, true};
                    pos -= boundary;
                    if (lk == 1) {
                        c[0] = ln - 2;
                        ln--;
                    } else {
                        for (int i = 0; i <= lk - 2; i++) c[i] = i;
                        c[lk - 1] = ln - 2;
                        ln--;
                    }
                }
            }
        }

        if (lk == ln) {
            for (int i = 0; i < lk; i++) c[i] = i;
        }
        sp++;
        stk[sp] = {ln, lk, 3, l_neg};
    }

    // 3. Build mask from c[]
    uint64_t mask = 0;
    for (int i = 0; i < hamming_h; i++) mask |= (1ULL << c[i]);

    // 4. Replay step_idx transitions (CPU revdoor_step, integer ops only)
    for (uint32_t s = 0; s < step_idx; s++) {
        bool found = false;
        while (sp >= 0 && !found) {
            CPURDFrame *f = &stk[sp];
            if (f->k == 0 || f->k == f->n) {
                sp--;
                if (sp >= 0) stk[sp].phase++;
                continue;
            }
            switch (f->phase) {
            case 0: {
                int child_sp = sp + 1;
                if (!f->is_neg) {
                    stk[child_sp] = {f->n - 1, f->k, 0, false};
                } else {
                    stk[child_sp] = {f->n - 1, f->k - 1, 0, false};
                }
                sp = child_sp;
                break;
            }
            case 1: {
                if (!f->is_neg) {
                    if (f->k == 1) {
                        int old = c[0]; c[0] = f->n - 1;
                        mask = (mask & ~(1ULL << old)) | (1ULL << c[0]);
                    } else {
                        int old = c[f->k - 2];
                        c[f->k - 2] = c[f->k - 1];
                        c[f->k - 1] = f->n - 1;
                        mask = (mask & ~(1ULL << old)) | (1ULL << (f->n - 1));
                    }
                } else {
                    if (f->k == 1) {
                        int old = c[0]; c[0] = f->n - 2;
                        mask = (mask & ~(1ULL << old)) | (1ULL << c[0]);
                    } else {
                        int old = c[f->k - 1];
                        int added_val = f->k - 2;
                        c[f->k - 1] = c[f->k - 2];
                        c[f->k - 2] = added_val;
                        mask = (mask & ~(1ULL << old)) | (1ULL << added_val);
                    }
                }
                f->phase = 2;
                found = true;
                break;
            }
            case 2: {
                int child_sp = sp + 1;
                if (!f->is_neg) {
                    stk[child_sp] = {f->n - 1, f->k - 1, 0, true};
                } else {
                    stk[child_sp] = {f->n - 1, f->k, 0, true};
                }
                sp = child_sp;
                break;
            }
            case 3:
                sp--;
                if (sp >= 0) stk[sp].phase++;
                break;
            }
        }
    }

    // 5. XOR with center to get seed
    uint64_t seedMaskLo = (n < 64) ? ((1ULL << n) - 1ULL) : 0xFFFFFFFFFFFFFFFFULL;
    uint64_t seed_lo = (mask ^ config->targetSeedLo) & seedMaskLo;

    // 6. Expand to full 256-bit key
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

    // 7. Verify and output
    Int privkey;
    privkey.SetInt32(0);
    privkey.bits64[0] = keyBits[0];
    privkey.bits64[1] = keyBits[1];
    privkey.bits64[2] = keyBits[2];
    privkey.bits64[3] = keyBits[3];

    checkAddr(*(address_t*)(hash), hash, privkey, 0, 0, true);
}


#endif // SEP7_REVOLVING_DOOR_CU
