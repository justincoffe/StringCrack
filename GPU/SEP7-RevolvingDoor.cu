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

// RevDoor batch size — independent of GosperWalk's BATCH_N
// Must be >= FLUSH_INTERVAL (16) to prevent overflow during cooperative flush
#define RD_BATCH_N 20

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
// Register-Packed Unranking (PURE SILICON - ZERO ARRAY)
// =====================================================================================
// Safe bitmask generator to prevent 1ULL << 64 overflow
__device__ __forceinline__ uint64_t n_bits_mask(int bits) {
    return (bits >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << bits) - 1);
}

__device__ void revdoor_unrank_reg(
    int N, int K, uint64_t pos,
    uint64_t &mask, uint64_t &p0, uint64_t &p1, uint64_t &neg_bits,
    int *sp, int *out_curr_n, int *out_curr_k)
{
    mask = n_bits_mask(K); // Initial mask: first(GEN(N, K))
    int n = N, k = K;
    int is_neg = 0;
    *sp = -1;
    p0 = 0; p1 = 0; neg_bits = 0;

    while (k > 0 && k < n) {
        (*sp)++;
        if (!is_neg) {
            uint64_t boundary = rd_comb(n - 1, k);
            if (pos < boundary) {
                n--;
            } else {
                p1 |= (1ULL << *sp);
                pos -= boundary;
                
                // Transition to NEG(n-1, k-1) + bit (n-1)
                mask &= ~n_bits_mask(n);
                if (k == 1) {
                    mask |= (1ULL << (n - 1));
                    n--; k = 0; is_neg = 1;
                } else {
                    mask |= n_bits_mask(k - 2);
                    mask |= (1ULL << (n - 2));
                    mask |= (1ULL << (n - 1));
                    n--; k--; is_neg = 1;
                }
            }
        } else {
            uint64_t boundary = rd_comb(n - 1, k - 1);
            if (pos < boundary) {
                neg_bits |= (1ULL << *sp);
                
                // Descend to GEN(n-1, k-1) + bit (n-1)
                mask &= ~n_bits_mask(n);
                mask |= n_bits_mask(k - 1);
                mask |= (1ULL << (n - 1));
                n--; k--; is_neg = 0;
            } else {
                p1 |= (1ULL << *sp);
                neg_bits |= (1ULL << *sp);
                pos -= boundary;
                
                // Transition to NEG(n-1, k)
                mask &= ~n_bits_mask(n);
                if (k == 1) {
                    mask |= (1ULL << (n - 2));
                    n--;
                } else {
                    mask |= n_bits_mask(k - 1);
                    mask |= (1ULL << (n - 2));
                    n--;
                }
            }
        }
    }

    // Base cases
    mask &= ~n_bits_mask(n);
    if (k == n) {
        mask |= n_bits_mask(k);
    }

    (*sp)++;
    p0 |= (1ULL << *sp);
    p1 |= (1ULL << *sp);
    if (is_neg) neg_bits |= (1ULL << *sp);

    *out_curr_n = n;
    *out_curr_k = k;
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
// Register-Packed Step Engine (PURE SILICON - ZERO ARRAY)
// =====================================================================================
__device__ bool revdoor_step_reg(
    uint64_t &mask, uint64_t &p0, uint64_t &p1, uint64_t &neg_bits,
    int *sp_ptr, int *curr_n_ptr, int *curr_k_ptr,
    int *removed, int *added)
{
    int sp = *sp_ptr;
    int curr_n = *curr_n_ptr;
    int curr_k = *curr_k_ptr;

    while (sp >= 0) {
        if (curr_k == 0 || curr_k == curr_n) {
            sp--;
            if (sp >= 0) {
                curr_n++;
                int p_parent = ((p0 >> sp) & 1) | (((p1 >> sp) & 1) << 1);
                int neg_parent = (neg_bits >> sp) & 1;
                
                if (p_parent == 0 && neg_parent == 1) curr_k++;
                else if (p_parent == 2 && neg_parent == 0) curr_k++;

                p_parent++;
                p0 = (p0 & ~(1ULL << sp)) | ((uint64_t)(p_parent & 1) << sp);
                p1 = (p1 & ~(1ULL << sp)) | ((uint64_t)((p_parent >> 1) & 1) << sp);
            }
            continue;
        }

        int p = ((p0 >> sp) & 1) | (((p1 >> sp) & 1) << 1);
        int neg = (neg_bits >> sp) & 1;

        if (p == 0) {
            int child_sp = sp + 1;
            if (!neg) curr_n--; else { curr_n--; curr_k--; }
            p0 &= ~(1ULL << child_sp); 
            p1 &= ~(1ULL << child_sp); 
            neg_bits &= ~(1ULL << child_sp);
            sp = child_sp;
        }
        else if (p == 1) {
            // HARDWARE INTRINSIC EXTRACTION (No c[] array needed!)
            uint64_t sub_mask = mask & n_bits_mask(curr_n);
            
            if (!neg) {
                if (curr_k == 1) {
                    *removed = __ffsll(sub_mask) - 1; 
                    *added = curr_n - 1;
                } else {
                    int highest = 63 - __clzll(sub_mask);
                    uint64_t without_highest = sub_mask ^ (1ULL << highest);
                    *removed = 63 - __clzll(without_highest); // Second highest bit
                    *added = curr_n - 1;
                }
            } else {
                if (curr_k == 1) {
                    *removed = __ffsll(sub_mask) - 1; 
                    *added = curr_n - 2;
                } else {
                    *removed = 63 - __clzll(sub_mask); // Highest bit
                    *added = curr_k - 2;
                }
            }
            
            p0 &= ~(1ULL << sp); p1 |= (1ULL << sp);
            *sp_ptr = sp; *curr_n_ptr = curr_n; *curr_k_ptr = curr_k;
            return true;
        }
        else if (p == 2) {
            int child_sp = sp + 1;
            if (!neg) { curr_n--; curr_k--; neg_bits |= (1ULL << child_sp); }
            else      { curr_n--;           neg_bits |= (1ULL << child_sp); }
            p0 &= ~(1ULL << child_sp); p1 &= ~(1ULL << child_sp);
            sp = child_sp;
        }
        else if (p == 3) {
            sp--;
            if (sp >= 0) {
                curr_n++;
                int p_parent = ((p0 >> sp) & 1) | (((p1 >> sp) & 1) << 1);
                int neg_parent = (neg_bits >> sp) & 1;
                
                if (p_parent == 0 && neg_parent == 1) curr_k++;
                else if (p_parent == 2 && neg_parent == 0) curr_k++;

                p_parent++;
                p0 = (p0 & ~(1ULL << sp)) | ((uint64_t)(p_parent & 1) << sp);
                p1 = (p1 & ~(1ULL << sp)) | ((uint64_t)((p_parent >> 1) & 1) << sp);
            }
        }
    }
    
    *sp_ptr = sp; *curr_n_ptr = curr_n; *curr_k_ptr = curr_k;
    return false;
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

    uint64_t prefix[RD_BATCH_N][4];
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
    uint64_t base_pos,          // starting POSITION in revolving-door order
    uint64_t totalCombs,        // C(n, hamming_h)
    int chunk_size)
{
    int lane_id = threadIdx.x;
    uint32_t walk_id = blockIdx.x * 32 + lane_id; // <-- 32 threads per block

    // This thread's starting position in the revolving-door sequence
    uint64_t start_pos = base_pos + (uint64_t)walk_id * (uint64_t)chunk_size;
    if (start_pos >= totalCombs) return;

    int n = d_numFreeBits;
    if (n > 64) return;

    // ═══════ REVOLVING DOOR INITIALIZATION ═══════
    uint64_t mask;
    uint64_t p0, p1, neg_bits;
    int curr_n, curr_k, sp;

    revdoor_unrank_reg(n, hamming_h, start_pos, mask, p0, p1, neg_bits, &sp, &curr_n, &curr_k);

    // ═══════ COMPUTE INITIAL EC POINT using G_free table (SEP-compatible) ═══════

    // 1. XOR with the center string to get the TRUE physical seed
    uint64_t seedMaskLo = (n < 64) ? ((1ULL << n) - 1ULL) : 0xFFFFFFFFFFFFFFFFULL;
    uint64_t seed_lo = (mask ^ d_targetSeedLo) & seedMaskLo;

    // 2. Build initial EC point from seed_lo using standard G_free table
    uint64_t accX[4], accY[4], accZ[4];
    bool pointSet = false;

    if (d_lockedPopcount > 0) {
        Load256(accX, d_basePointX);
        Load256(accY, d_basePointY);
        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
        pointSet = true;
    }

    // Add each free bit that's set in the physical seed using G_free table
    for (int i = 0; i < n; i++) {
        if ((seed_lo >> i) & 1ULL) {
            int idx = i * 4;
            uint64_t gx[4], gy[4];
            gx[0] = __ldg(&d_GfreeX[idx]);
            gx[1] = __ldg(&d_GfreeX[idx + 1]);
            gx[2] = __ldg(&d_GfreeX[idx + 2]);
            gx[3] = __ldg(&d_GfreeX[idx + 3]);
            gy[0] = __ldg(&d_GfreeY[idx]);
            gy[1] = __ldg(&d_GfreeY[idx + 1]);
            gy[2] = __ldg(&d_GfreeY[idx + 2]);
            gy[3] = __ldg(&d_GfreeY[idx + 3]);

            if (!pointSet) {
                Load256(accX, gx); Load256(accY, gy);
                accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                pointSet = true;
            } else {
                jacobian_add_affine_inplace(accX, accY, accZ, gx, gy);
            }
        }
    }

    if (!pointSet) return;

    // ═══════ BATCHED REVOLVING DOOR WALK LOOP ═══════

    uint64_t buf_X[MAX_BATCH][4];
    uint64_t buf_Y[MAX_BATCH][4];
    uint64_t buf_Z[MAX_BATCH][4];
    uint64_t buf_masks[MAX_BATCH];
    uint32_t buf_steps[MAX_BATCH];
    uint64_t Zinv[MAX_BATCH][4];

    int steps_done = 0;
    int end_step = chunk_size;

    // Clamp to totalCombs boundary
    uint64_t max_steps = totalCombs - start_pos;
    if ((uint64_t)end_step > max_steps) end_step = (int)max_steps;

    // Initialize walk tracking
    uint64_t last_ec_mask = mask;
    bool use_pcfilter = (d_popcountMin > 0 || d_popcountMax < 256);
    steps_done = 1;
    int batch_count = 0;
    const int FLUSH_INTERVAL = 16;
    int steps_since_flush = 0;

    // Buffer initial point ONLY if it passes popcount
    if (use_pcfilter) {
        uint64_t seed_check = (mask ^ d_targetSeedLo) & seedMaskLo;
        int pc_abs = __popcll(seed_check) + d_lockedPopcount;
        if (pc_abs >= d_popcountMin && pc_abs <= d_popcountMax) {
            Load256(buf_X[0], accX);
            Load256(buf_Y[0], accY);
            Load256(buf_Z[0], accZ);
            buf_masks[0] = mask;
            buf_steps[0] = 1;
            batch_count = 1;
        }
    } else {
        Load256(buf_X[0], accX);
        Load256(buf_Y[0], accY);
        Load256(buf_Z[0], accZ);
        buf_masks[0] = mask;
        buf_steps[0] = 1;
        batch_count = 1;
    }

    // ═══════ TELEPORTATION-ENABLED WALK LOOP ═══════
    // Track last EC-synced mask. When popcount fails, advance mask only (5 cycles).
    // When popcount passes, teleport EC accumulator via seed-space XOR diff.
    // WARP-COOPERATIVE: Flush synchronized every FLUSH_INTERVAL steps to eliminate divergence.

    while (steps_done < end_step) {

        // ─── REVOLVING DOOR STEP ───
        int removed_idx, added_idx;
        if (!revdoor_step_reg(mask, p0, p1, neg_bits, &sp, &curr_n, &curr_k, &removed_idx, &added_idx)) {
            break;
        }
        mask = (mask & ~(1ULL << removed_idx)) | (1ULL << added_idx);
        steps_done++;
        steps_since_flush++;

        // ─── POPCOUNT PRE-FILTER ───
        bool passes = true;
        if (use_pcfilter) {
            uint64_t seed_check = (mask ^ d_targetSeedLo) & seedMaskLo;
            int pc_abs = __popcll(seed_check) + d_lockedPopcount;
            if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) {
                passes = false;
            }
        }

        // ─── TELEPORT + BUFFER (only if passes) ───
        if (passes) {
            uint64_t old_seed = (last_ec_mask ^ d_targetSeedLo) & seedMaskLo;
            uint64_t new_seed = (mask ^ d_targetSeedLo) & seedMaskLo;

            if (old_seed != new_seed) {
                apply_xor_diff(accX, accY, accZ, old_seed, new_seed);
            }
            last_ec_mask = mask;

            Load256(buf_X[batch_count], accX);
            Load256(buf_Y[batch_count], accY);
            Load256(buf_Z[batch_count], accZ);
            buf_masks[batch_count] = mask;
            buf_steps[batch_count] = steps_done;
            batch_count++;
        }

        // ─── WARP-COOPERATIVE FLUSH ───
        if (steps_since_flush >= FLUSH_INTERVAL || batch_count >= MAX_BATCH) {
            steps_since_flush = 0;

            if (batch_count > 0) {
                rd_batch_invert_Z(buf_Z, Zinv, batch_count);

                for (int b = 0; b < batch_count; b++) {
                    uint64_t Zinv_sq[4], px[4], py[4], Zinv_cb[4];
                    _ModSqr(Zinv_sq, Zinv[b]);
                    _ModMult(px, Zinv_sq, buf_X[b]);
                    _ModMult(Zinv_cb, Zinv_sq, Zinv[b]);
                    _ModMult(py, Zinv_cb, buf_Y[b]);

                    uint8_t odd_py = (uint8_t)(py[0] & 1);
                    uint32_t h[5];
                    _GetHash160Comp(px, odd_py, (uint8_t*)h);

                    if (sAddress[h[0] & 0xFFFF] != 0) {
                        uint32_t step_idx = buf_steps[b];
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

                // Reset accumulator to affine
                if (steps_done < end_step) {
                    if (!use_pcfilter) {
                        int last = batch_count - 1;
                        uint64_t Zinv_sq[4], Zinv_cb[4];
                        _ModSqr(Zinv_sq, Zinv[last]);
                        _ModMult(accX, Zinv_sq, buf_X[last]);
                        _ModMult(Zinv_cb, Zinv_sq, Zinv[last]);
                        _ModMult(accY, Zinv_cb, buf_Y[last]);
                        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                    } else {
                        uint64_t Zinv_reset[5];
                        Zinv_reset[0] = accZ[0]; Zinv_reset[1] = accZ[1];
                        Zinv_reset[2] = accZ[2]; Zinv_reset[3] = accZ[3]; Zinv_reset[4] = 0;
                        _ModInv(Zinv_reset);
                        uint64_t Zsq[4], Zcb[4];
                        _ModSqr(Zsq, Zinv_reset);
                        _ModMult(accX, accX, Zsq);
                        _ModMult(Zcb, Zsq, Zinv_reset);
                        _ModMult(accY, accY, Zcb);
                        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                    }
                }
                batch_count = 0;
            }
        }
    }

    // ─── FLUSH REMAINING BATCH ───
    if (batch_count > 0) {
        rd_batch_invert_Z(buf_Z, Zinv, batch_count);

        for (int b = 0; b < batch_count; b++) {
            uint64_t Zinv_sq[4], px[4], py[4], Zinv_cb[4];
            _ModSqr(Zinv_sq, Zinv[b]);
            _ModMult(px, Zinv_sq, buf_X[b]);
            _ModMult(Zinv_cb, Zinv_sq, Zinv[b]);
            _ModMult(py, Zinv_cb, buf_Y[b]);

            uint8_t odd_py = (uint8_t)(py[0] & 1);
            uint32_t h[5];
            _GetHash160Comp(px, odd_py, (uint8_t*)h);

            if (sAddress[h[0] & 0xFFFF] != 0) {
                uint32_t step_idx = buf_steps[b];
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

    // D[i][j] = P_j + P_i with SEP center-aware sign resolution
    // This ensures the D-table entries correctly compute the EC delta
    // when transitioning between combinations with the SEP center string
    #pragma omp parallel for schedule(dynamic) if(n > 16)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            
            // 1. Check if the bits are 1 in the SEP Center String
            bool T_i = (config->targetSeedLo & (1ULL << i)) != 0;
            bool T_j = (config->targetSeedLo & (1ULL << j)) != 0;
            
            // 2. Resolve Added Bit (j)
            Point P_j = Gfree[j]; 
            if (T_j) P_j.y.ModNeg(); // If center is 1, adding to mask REMOVES from seed
            
            // 3. Resolve Removed Bit (i)
            Point P_i = Gfree[i];
            if (!T_i) P_i.y.ModNeg(); // If center is 0, removing from mask REMOVES from seed
            
            // 4. Bake the exact Delta into the D-Table
            Point D = secp->AddDirect(P_j, P_i); 
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
// DEVICE HELPER: Process a batch of buffered Jacobian points
// Uses template boolean to completely compile out runtime branches!
// =====================================================================================
template <int MAX_BATCH, bool IS_WARP_PACKED>
__device__ __forceinline__ void rd_process_batch(
    uint64_t buf_X[][4], uint64_t buf_Y[][4], uint64_t buf_Z[][4],
    uint64_t buf_masks[], uint32_t buf_steps[], uint64_t Zinv[][4],
    int batch_count, uint32_t walk_id, int lane_id,
    address_t* sAddress, uint32_t* lookup32, uint32_t* out)
{
    rd_batch_invert_Z(buf_Z, Zinv, batch_count);
    for (int b = 0; b < batch_count; b++) {
        uint64_t Zinv_sq[4], px[4], py[4], Zinv_cb[4];
        _ModSqr(Zinv_sq, Zinv[b]);
        _ModMult(px, Zinv_sq, buf_X[b]);
        _ModMult(Zinv_cb, Zinv_sq, Zinv[b]);
        _ModMult(py, Zinv_cb, buf_Y[b]);
        uint8_t odd_py = (uint8_t)(py[0] & 1);
        uint32_t h[5];
        _GetHash160Comp(px, odd_py, (uint8_t*)h);

        if (sAddress[h[0] & 0xFFFF] != 0) {
            uint32_t step_idx = buf_steps[b];
            uint32_t pos = atomicAdd(out, 1);
            if (pos < 65536) {
                if (IS_WARP_PACKED) {
                    uint32_t* item = out + 1 + pos * ITEM_SIZE32_WARP;
                    item[0] = walk_id;
                    item[1] = (uint32_t)lane_id;
                    int16_t* ptr = (int16_t*)&item[2];
                    ptr[0] = (int16_t)(step_idx & 0x7FFF);
                    ptr[1] = (int16_t)((step_idx >> 15) & 0x7FFF);
                    memcpy(item + 3, h, 20);
                } else {
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
// GOD ENGINE: GRID-STRIDED COSET REVDOOR (W >= 64)
// Scales infinitely. 1 Walk = ceil(W/128) Blocks. Pure L2 Cache Broadcast.
// =====================================================================================

template <int MAX_BATCH>
__global__ __launch_bounds__(128, 4) 
void comp_keys_coset_revdoor(
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    int L_bits, int k2, int B_top, int k1,
    uint64_t base_pos, uint64_t L_totalCombs,
    int chunk_size, int qi_chunks, uint64_t W, uint64_t* d_Qi_array)
{
    int lane = threadIdx.x; // 0..127
    
    // Grid-Stride logic: Map blocks to Walk IDs and Q_i chunks
    uint32_t walk_id = blockIdx.x / qi_chunks;
    uint32_t qi_batch_id = blockIdx.x % qi_chunks;

    int qi_idx = (qi_batch_id * 128) + lane;
    if (qi_idx >= W) return; // Kills out-of-bounds threads gracefully

    uint64_t start_pos = base_pos + (uint64_t)walk_id * chunk_size;
    if (start_pos >= L_totalCombs) return;

    // Grab specific offset from the massive array
    uint64_t qi_mask = d_Qi_array[qi_idx] << L_bits;

    uint64_t p_mask, p0, p1, neg_bits;
    int curr_n, curr_k, sp;
    revdoor_unrank_reg(L_bits, k2, start_pos, p_mask, p0, p1, neg_bits, &sp, &curr_n, &curr_k);

    uint64_t full_mask = qi_mask | p_mask;

    // ═══════ INITIAL EC POINT ═══════
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

    // ═══════ BATCHED WALK LOOP ═══════
    uint64_t buf_X[MAX_BATCH][4], buf_Y[MAX_BATCH][4], buf_Z[MAX_BATCH][4];
    uint64_t buf_masks[MAX_BATCH], Zinv[MAX_BATCH][4];
    uint32_t buf_steps[MAX_BATCH];

    int steps_done = 0;
    int end_step = chunk_size;
    uint64_t max_steps = L_totalCombs - start_pos;
    if ((uint64_t)end_step > max_steps) end_step = (int)max_steps;

    int batch_count = 0;
    steps_done = 1;
    const int FLUSH_INTERVAL = 16;
    int steps_since_flush = 0;
    bool use_pcfilter = (d_popcountMin > 0 || d_popcountMax < 256);

    if (use_pcfilter) {
        uint64_t seed_check = (full_mask ^ d_targetSeedLo) & d_seedMaskLo;
        int pc_abs = __popcll(seed_check) + d_lockedPopcount;
        if (pc_abs >= d_popcountMin && pc_abs <= d_popcountMax) {
            Load256(buf_X[0], accX); Load256(buf_Y[0], accY); Load256(buf_Z[0], accZ);
            buf_masks[0] = full_mask;
            buf_steps[0] = 1;
            batch_count = 1;
        }
    } else {
        Load256(buf_X[0], accX); Load256(buf_Y[0], accY); Load256(buf_Z[0], accZ);
        buf_masks[0] = full_mask;
        buf_steps[0] = 1;
        batch_count = 1;
    }

    while (steps_done < end_step) {
        
        // PERFECT WARP SYNCHRONIZATION: Every thread executes identical state machine logic
        int removed_idx, added_idx;
        if (!revdoor_step_reg(p_mask, p0, p1, neg_bits, &sp, &curr_n, &curr_k, &removed_idx, &added_idx)) {
            break; 
        }

        p_mask = (p_mask & ~(1ULL << removed_idx)) | (1ULL << added_idx);
        full_mask = qi_mask | p_mask;

        // PERFECT MEMORY COALESCING: All threads request the exact same D-Table index
        uint64_t dX[4], dY[4];
        load_Dtable(removed_idx, added_idx, d_numFreeBits, dX, dY);
        jacobian_add_affine_inplace(accX, accY, accZ, dX, dY);
        steps_done++;
        steps_since_flush++;

        // ─── POPCOUNT GATE (no continue — preserves warp sync) ───
        bool passes = true;
        if (use_pcfilter) {
            uint64_t seed_check = (full_mask ^ d_targetSeedLo) & d_seedMaskLo;
            int pc_abs = __popcll(seed_check) + d_lockedPopcount;
            if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) {
                passes = false;
            }
        }

        if (passes) {
            Load256(buf_X[batch_count], accX); Load256(buf_Y[batch_count], accY); Load256(buf_Z[batch_count], accZ);
            buf_masks[batch_count] = full_mask;
            buf_steps[batch_count] = steps_done;
            batch_count++;
        }

        // ─── WARP-COOPERATIVE FLUSH ───
        if (steps_since_flush >= FLUSH_INTERVAL || batch_count >= MAX_BATCH) {
            steps_since_flush = 0;

            if (batch_count > 0) {
                uint32_t packed_id = (walk_id << 12) | (uint32_t)qi_idx;
                rd_process_batch<MAX_BATCH, false>(
                    buf_X, buf_Y, buf_Z, buf_masks, buf_steps, Zinv,
                    batch_count, packed_id, 0,
                    sAddress, lookup32, out);

                if (steps_done < end_step) {
                    if (!use_pcfilter) {
                        int last = batch_count - 1;
                        uint64_t Zinv_sq[4], Zinv_cb[4];
                        _ModSqr(Zinv_sq, Zinv[last]); _ModMult(accX, Zinv_sq, buf_X[last]);
                        _ModMult(Zinv_cb, Zinv_sq, Zinv[last]); _ModMult(accY, Zinv_cb, buf_Y[last]);
                        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                    } else {
                        uint64_t Zinv_reset[5];
                        Zinv_reset[0] = accZ[0]; Zinv_reset[1] = accZ[1];
                        Zinv_reset[2] = accZ[2]; Zinv_reset[3] = accZ[3]; Zinv_reset[4] = 0;
                        _ModInv(Zinv_reset);
                        uint64_t Zsq[4], Zcb[4];
                        _ModSqr(Zsq, Zinv_reset);
                        _ModMult(accX, accX, Zsq);
                        _ModMult(Zcb, Zsq, Zinv_reset);
                        _ModMult(accY, accY, Zcb);
                        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                    }
                }
                batch_count = 0;
            }
        }
    }

    if (batch_count > 0) {
        uint32_t packed_id = (walk_id << 12) | (uint32_t)qi_idx;
        rd_process_batch<MAX_BATCH, false>(
            buf_X, buf_Y, buf_Z, buf_masks, buf_steps, Zinv,
            batch_count, packed_id, 0,
            sAddress, lookup32, out);
    }
}


// =====================================================================================
// WARP-PACKED COSET REVDOOR (4 Walks per Block, 1 Walk per Warp)
// Designed for 32 <= W < 128.
// Pure Register State. Zero Shared Memory. Zero __syncwarp.
// Q_i baked into initial accumulator — zero redundant EC adds in loop.
// =====================================================================================
template <int MAX_BATCH>
__global__ __launch_bounds__(128, 4)
void comp_keys_warp_packed_revdoor(
    address_t* sAddress, uint32_t* lookup32, uint32_t* out,
    int L_bits, int k2, int B_top, int k1,
    uint64_t base_pos, uint64_t L_totalCombs,
    int chunk_size, int qi_batches, uint64_t* d_Qi_array)
{
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    uint32_t global_warp_id = (blockIdx.x * 4) + warp_id;

    // Decompose into walk chunk and Q_i batch
    uint32_t walk_chunk_id = global_warp_id / qi_batches;
    uint32_t qi_batch_id   = global_warp_id % qi_batches;

    // Q_i assignment for this lane
    int qi_idx = qi_batch_id * 32 + lane;
    uint64_t W = rd_comb(B_top, k1);
    if (qi_idx >= (int)W) return;

    // Walk position from walk_chunk_id
    uint64_t start_pos = base_pos + (uint64_t)walk_chunk_id * chunk_size;
    if (start_pos >= L_totalCombs) return;

    // Unique Q_i mask for this lane, shifted into upper bits
    uint64_t qi_mask = d_Qi_array[qi_idx] << L_bits;

    // ═══════ WARP-UNIFORM UNRANKING ═══════
    // All 32 active lanes compute the EXACT same revolving-door starting state
    uint64_t p_mask, p0, p1, neg_bits;
    int curr_n, curr_k, sp;
    revdoor_unrank_reg(L_bits, k2, start_pos, p_mask, p0, p1, neg_bits, &sp, &curr_n, &curr_k);

    // Per-lane unique full_mask (Q_i baked in)
    uint64_t full_mask = qi_mask | p_mask;

    // ═══════ PRIVATE INITIAL EC POINT (Q_i baked into accumulator) ═══════
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

    // ═══════ PURE REGISTER REVOLVING DOOR LOOP ═══════
    uint64_t buf_X[MAX_BATCH][4], buf_Y[MAX_BATCH][4], buf_Z[MAX_BATCH][4];
    uint64_t buf_masks[MAX_BATCH], Zinv[MAX_BATCH][4];
    uint32_t buf_steps[MAX_BATCH];

    int steps_done = 0;
    int end_step = chunk_size;
    uint64_t max_steps = L_totalCombs - start_pos;
    if ((uint64_t)end_step > max_steps) end_step = (int)max_steps;

    // Buffer the initial point (step 0)
    int batch_count = 0;
    steps_done = 1;
    const int FLUSH_INTERVAL = 16;
    int steps_since_flush = 0;
    bool use_pcfilter = (d_popcountMin > 0 || d_popcountMax < 256);

    if (use_pcfilter) {
        uint64_t seed_check = (full_mask ^ d_targetSeedLo) & d_seedMaskLo;
        int pc_abs = __popcll(seed_check) + d_lockedPopcount;
        if (pc_abs >= d_popcountMin && pc_abs <= d_popcountMax) {
            Load256(buf_X[0], accX); Load256(buf_Y[0], accY); Load256(buf_Z[0], accZ);
            buf_masks[0] = full_mask;
            buf_steps[0] = 1;
            batch_count = 1;
        }
    } else {
        Load256(buf_X[0], accX); Load256(buf_Y[0], accY); Load256(buf_Z[0], accZ);
        buf_masks[0] = full_mask;
        buf_steps[0] = 1;
        batch_count = 1;
    }

    while (steps_done < end_step) {

        // STEP 1: Nijenhuis-Wilf state machine — identical across all 32 lanes
        int removed_idx, added_idx;
        if (!revdoor_step_reg(p_mask, p0, p1, neg_bits, &sp, &curr_n, &curr_k, &removed_idx, &added_idx)) {
            break;
        }
        p_mask = (p_mask & ~(1ULL << removed_idx)) | (1ULL << added_idx);
        full_mask = qi_mask | p_mask;

        // STEP 2: D-Table delta fetch — all 32 lanes hit same address, L2 broadcast
        uint64_t dX[4], dY[4];
        load_Dtable(removed_idx, added_idx, d_numFreeBits, dX, dY);

        // STEP 3: Pure register accumulator update — no shared memory, no sync
        jacobian_add_affine_inplace(accX, accY, accZ, dX, dY);
        steps_done++;
        steps_since_flush++;

        // ─── POPCOUNT GATE (no continue) ───
        bool passes = true;
        if (use_pcfilter) {
            uint64_t seed_check = (full_mask ^ d_targetSeedLo) & d_seedMaskLo;
            int pc_abs = __popcll(seed_check) + d_lockedPopcount;
            if (pc_abs < d_popcountMin || pc_abs > d_popcountMax) {
                passes = false;
            }
        }

        if (passes) {
            Load256(buf_X[batch_count], accX);
            Load256(buf_Y[batch_count], accY);
            Load256(buf_Z[batch_count], accZ);
            buf_masks[batch_count] = full_mask;
            buf_steps[batch_count] = steps_done;
            batch_count++;
        }

        // ─── WARP-COOPERATIVE FLUSH ───
        if (steps_since_flush >= FLUSH_INTERVAL || batch_count >= MAX_BATCH) {
            steps_since_flush = 0;

            if (batch_count > 0) {
                rd_process_batch<MAX_BATCH, true>(
                    buf_X, buf_Y, buf_Z, buf_masks, buf_steps, Zinv,
                    batch_count, global_warp_id, lane,
                    sAddress, lookup32, out);

                if (steps_done < end_step) {
                    if (!use_pcfilter) {
                        int last = batch_count - 1;
                        uint64_t Zinv_sq[4], Zinv_cb[4];
                        _ModSqr(Zinv_sq, Zinv[last]);
                        _ModMult(accX, Zinv_sq, buf_X[last]);
                        _ModMult(Zinv_cb, Zinv_sq, Zinv[last]);
                        _ModMult(accY, Zinv_cb, buf_Y[last]);
                        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                    } else {
                        uint64_t Zinv_reset[5];
                        Zinv_reset[0] = accZ[0]; Zinv_reset[1] = accZ[1];
                        Zinv_reset[2] = accZ[2]; Zinv_reset[3] = accZ[3]; Zinv_reset[4] = 0;
                        _ModInv(Zinv_reset);
                        uint64_t Zsq[4], Zcb[4];
                        _ModSqr(Zsq, Zinv_reset);
                        _ModMult(accX, accX, Zsq);
                        _ModMult(Zcb, Zsq, Zinv_reset);
                        _ModMult(accY, accY, Zcb);
                        accZ[0] = 1; accZ[1] = 0; accZ[2] = 0; accZ[3] = 0;
                    }
                }
                batch_count = 0;
            }
        }
    }

    // Flush remaining partial batch
    if (batch_count > 0) {
        rd_process_batch<MAX_BATCH, true>(
            buf_X, buf_Y, buf_Z, buf_masks, buf_steps, Zinv,
            batch_count, global_warp_id, lane,
            sAddress, lookup32, out);
    }
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

    int threadsPerBlock = 32; // <-- Restored to 32
    int numBlocks = (numWalks + threadsPerBlock - 1) / threadsPerBlock;

    comp_keys_revdoor<RD_BATCH_N><<<numBlocks, threadsPerBlock, 0, streams[s]>>>(
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

uint32_t GPUEngine::SyncWarpPackedRevDoorBatch(int stepToSync, std::vector<ITEM> &addressFound) {
    int s = stepToSync % 2;
    cudaStreamSynchronize(streams[s]);

    uint32_t nbFound = h_outputPinned[s][0];
    if (nbFound > maxFound) nbFound = maxFound;

    addressFound.clear();
    if (nbFound > 0) {
        int qi_batches = qi_batches_last;
        for (uint32_t i = 0; i < nbFound; i++) {
            uint32_t* itemPtr = h_outputPinned[s] + (i * ITEM_SIZE32_WARP + 1);
            
            uint32_t walk_id       = itemPtr[0];
            uint32_t lane          = itemPtr[1];
            int16_t* sptr          = (int16_t*)&itemPtr[2];

            uint32_t walk_chunk_id = walk_id / qi_batches;
            uint32_t qi_batch_id   = walk_id % qi_batches;
            uint32_t qi_idx        = (qi_batch_id * 32) + lane;

            ITEM it;
            it.thId = ((uint64_t)walk_chunk_id << 32) | (uint64_t)qi_idx;
            it.endo = sptr[0] & 0x7FFF;
            it.incr = sptr[1];
            it.mode = (sptr[0] & 0x8000) != 0;
            it.hash = (uint8_t*)&itemPtr[3];
            addressFound.push_back(it);
        }
    }
    return nbFound;
}


// =====================================================================================
// SEP7: Coset RevDoor Upload & Dispatch
// =====================================================================================

void GPUEngine::UploadQiArray(uint64_t* h_Qi, uint64_t size) {
    int s = currentStep % 2;
    cudaMemcpyAsync(d_Qi_buffers[s], h_Qi, size * sizeof(uint64_t), cudaMemcpyHostToDevice, streams[s]);
}

void GPUEngine::UploadQiArrayBoth(uint64_t* h_Qi, uint64_t size) {
    cudaMemcpy(d_Qi_buffers[0], h_Qi, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Qi_buffers[1], h_Qi, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
}

void GPUEngine::LaunchRevDoorAsync(int L_bits, int k2, int B_top, int k1, uint64_t base_pos, uint64_t totalCombs, int chunk_size, int numBlocks, int qi_chunks, uint64_t W) {
    int s = currentStep % 2;
    cudaMemsetAsync(d_output[s], 0, 4, streams[s]);
    
    comp_keys_coset_revdoor<RD_BATCH_N><<<numBlocks, 128, 0, streams[s]>>>(
        inputAddress, inputAddressLookUp, d_output[s],
        L_bits, k2, B_top, k1, base_pos, totalCombs, chunk_size, qi_chunks, W, d_Qi_buffers[s]);
        
    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize, cudaMemcpyDeviceToHost, streams[s]);
    currentStep++;
}

void GPUEngine::LaunchWarpPackedRevDoorAsync(
    int L_bits, int k2, int B_top, int k1,
    uint64_t base_pos, uint64_t totalCombs,
    int chunk_size, int numBlocks, int qi_batches)
{
    int s = currentStep % 2;
    cudaMemsetAsync(d_output[s], 0, 4, streams[s]);

    qi_batches_last = qi_batches;

    comp_keys_warp_packed_revdoor<RD_BATCH_N><<<numBlocks, 128, 0, streams[s]>>>(
        inputAddress, inputAddressLookUp, d_output[s],
        L_bits, k2, B_top, k1,
        base_pos, totalCombs,
        chunk_size, qi_batches, d_Qi_buffers[s]);

    cudaMemcpyAsync(h_outputPinned[s], d_output[s], outputSize,
                    cudaMemcpyDeviceToHost, streams[s]);
    currentStep++;
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
