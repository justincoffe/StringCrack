// =====================================================================================
// ShekinahMatrix.h — Dyadic Interval Decomposition + MITM Dispatch Pipeline
// 
// The Shekinah Matrix Algorithm: Given an arbitrary integer corridor [A, B) in
// the key space, decompose it into a minimal set of power-of-2-aligned blocks,
// each expressible as (locked_prefix, free_bits). These blocks feed directly into
// the God Engine's MITM kernel for O(sqrt(N)) coverage per block.
//
// Pipeline:
//   1. Dyadic decomposition: O(log2(B-A)) blocks, zero ghosts
//   2. GPU capacity split: blocks with free_bits > MAX_GPU_FREE_BITS get sub-divided
//   3. Popcount pruning: dead blocks eliminated before any GPU work
//   4. Adaptive merging: tiny edge blocks merged to reduce kernel launches
//   5. Block ordering: center-first heuristic for early termination
//   6. Dispatch: emit per-block lock commands with tightened popcount ranges
//
// Copyright (c) 2026 AlleSerOjje. All rights reserved.
// =====================================================================================

#ifndef SHEKINAH_MATRIX_H
#define SHEKINAH_MATRIX_H

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
// ═══════════════════════════════════════════════════════════════════
// 128-bit helpers (portable, no compiler __int128 needed)
// ═══════════════════════════════════════════════════════════════════
struct uint128_t {
    uint64_t lo;
    uint64_t hi;

    uint128_t() : lo(0), hi(0) {}
    uint128_t(uint64_t l) : lo(l), hi(0) {}
    uint128_t(uint64_t h, uint64_t l) : lo(l), hi(h) {}

    bool operator<(const uint128_t& o) const {
        return (hi < o.hi) || (hi == o.hi && lo < o.lo);
    }
    bool operator>(const uint128_t& o) const { return o < *this; }
    bool operator<=(const uint128_t& o) const { return !(o < *this); }
    bool operator>=(const uint128_t& o) const { return !(*this < o); }
    bool operator==(const uint128_t& o) const { return hi == o.hi && lo == o.lo; }
    bool operator!=(const uint128_t& o) const { return !(*this == o); }

    uint128_t operator+(const uint128_t& o) const {
        uint128_t r;
        r.lo = lo + o.lo;
        r.hi = hi + o.hi + (r.lo < lo ? 1 : 0);
        return r;
    }
    uint128_t operator-(const uint128_t& o) const {
        uint128_t r;
        r.lo = lo - o.lo;
        r.hi = hi - o.hi - (lo < o.lo ? 1 : 0);
        return r;
    }
    uint128_t operator<<(int s) const {
        if (s == 0) return *this;
        if (s >= 128) return uint128_t(0, 0);
        if (s >= 64) return uint128_t(lo << (s - 64), 0);
        return uint128_t((hi << s) | (lo >> (64 - s)), lo << s);
    }
    uint128_t operator>>(int s) const {
        if (s == 0) return *this;
        if (s >= 128) return uint128_t(0, 0);
        if (s >= 64) return uint128_t(0, hi >> (s - 64));
        return uint128_t(hi >> s, (lo >> s) | (hi << (64 - s)));
    }
    uint128_t operator&(const uint128_t& o) const {
        return uint128_t(hi & o.hi, lo & o.lo);
    }
    uint128_t operator|(const uint128_t& o) const {
        return uint128_t(hi | o.hi, lo | o.lo);
    }
    uint128_t operator~() const {
        return uint128_t(~hi, ~lo);
    }
    bool is_zero() const { return lo == 0 && hi == 0; }

    // Count trailing zeros
    int ctz() const {
        if (lo != 0) return __builtin_ctzll(lo);
        if (hi != 0) return 64 + __builtin_ctzll(hi);
        return 128;
    }
    // Bit width (position of highest set bit + 1)
    int bit_width() const {
        if (hi != 0) return 128 - __builtin_clzll(hi);
        if (lo != 0) return 64 - __builtin_clzll(lo);
        return 0;
    }
    // Get bit at position
    int bit(int pos) const {
        if (pos < 64) return (lo >> pos) & 1;
        return (hi >> (pos - 64)) & 1;
    }
    // Popcount
    int popcount() const {
        return __builtin_popcountll(lo) + __builtin_popcountll(hi);
    }
};

static inline uint128_t uint128_one() { return uint128_t(0, 1); }

// ═══════════════════════════════════════════════════════════════════
// CORE DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════

struct ShekinahBlock {
    uint128_t base;         // Starting value of this block
    int       free_bits;    // Number of free (lowest) bits in base
    int       locked_bits;  // total_bits - free_bits
    uint128_t block_size;   // = 1 << free_bits

    int       locked_popcount; // popcount of the frozen prefix bits
    int       free_pop_lo;     // min valid popcount for free bits
    int       free_pop_hi;     // max valid popcount for free bits
};

struct ShekinahEngineConfig {
    int max_free_bits;           // = 64, hard cap (GPU register width)
    int total_bits;              // e.g., 71 for puzzle 71
    int global_pop_lo;           // from -poprange, e.g., 30
    int global_pop_hi;           // from -poprange, e.g., 40
};

// Precomputed binomial coefficients C[n][k] for n,k up to 65
// (sufficient for MITM halves up to 64 bits)
static uint64_t shekinah_C[65][65];
static bool shekinah_C_init = false;

static void shekinah_precompute_binomials() {
    if (shekinah_C_init) return;
    memset(shekinah_C, 0, sizeof(shekinah_C));
    for (int n = 0; n <= 64; n++) {
        shekinah_C[n][0] = 1;
        for (int k = 1; k <= n; k++) {
            shekinah_C[n][k] = shekinah_C[n-1][k-1] + shekinah_C[n-1][k];
            if (shekinah_C[n][k] < shekinah_C[n-1][k-1])
                shekinah_C[n][k] = UINT64_MAX; // overflow saturation
        }
    }
    shekinah_C_init = true;
}

// ═══════════════════════════════════════════════════════════════════
// MITM WORK COST MODEL
// ═══════════════════════════════════════════════════════════════════

struct MITMWorkEstimate {
    uint64_t total_t1_entries;
    uint64_t total_t2_entries;
    uint64_t total_ec_ops;
    uint64_t keys_covered;
    int      num_pop_subrounds;
};

static MITMWorkEstimate shekinah_compute_mitm_work(const ShekinahBlock& blk) {
    shekinah_precompute_binomials();
    MITMWorkEstimate est = {};

    int half1 = blk.free_bits / 2;
    int half2 = blk.free_bits - half1;

    for (int k = blk.free_pop_lo; k <= blk.free_pop_hi; k++) {
        for (int p1 = std::max(0, k - half2); p1 <= std::min(half1, k); p1++) {
            int p2 = k - p1;
            if (p2 < 0 || p2 > half2) continue;

            uint64_t t1 = shekinah_C[half1][p1];
            uint64_t t2 = shekinah_C[half2][p2];
            if (t1 == UINT64_MAX || t2 == UINT64_MAX) continue;

            est.total_t1_entries += t1;
            est.total_t2_entries += t2;
            est.total_ec_ops += t1 + t2;

            if (t1 <= UINT64_MAX / t2) {
                est.keys_covered += t1 * t2;
            } else {
                est.keys_covered = UINT64_MAX;
            }
            est.num_pop_subrounds++;
        }
    }
    return est;
}

// ═══════════════════════════════════════════════════════════════════
// STEP 1: DYADIC INTERVAL DECOMPOSITION
// Decomposes [A, B) into minimal set of power-of-2 aligned blocks.
// O(log2(B-A)) iterations. Zero ghost combinations.
// ═══════════════════════════════════════════════════════════════════

static std::vector<ShekinahBlock> shekinah_decompose_range(
    uint128_t start, uint128_t end, int total_bits)
{
    std::vector<ShekinahBlock> blocks;
    uint128_t current = start;

    while (current < end) {
        // Constraint 1: Alignment — largest k where current is divisible by 2^k
        int k_align;
        if (current.is_zero()) {
            k_align = total_bits;
        } else {
            k_align = current.ctz();
        }

        // Constraint 2: Remaining space — block can't exceed (end - current)
        uint128_t remaining = end - current;
        int k_space = remaining.bit_width() - 1;
        // Verify 2^k_space <= remaining
        while (k_space > 0 && (uint128_one() << k_space) > remaining)
            k_space--;

        // Take the minimum — this is the largest valid block
        int k = std::min(k_align, k_space);
        if (k > total_bits) k = total_bits;

        ShekinahBlock blk;
        blk.base = current;
        blk.free_bits = k;
        blk.locked_bits = total_bits - k;
        blk.block_size = uint128_one() << k;
        blk.locked_popcount = 0;
        blk.free_pop_lo = 0;
        blk.free_pop_hi = k;

        blocks.push_back(blk);
        current = current + blk.block_size;
    }

    return blocks;
}

// ═══════════════════════════════════════════════════════════════════
// STEP 2: ANNOTATE POPCOUNT AND PRUNE DEAD BLOCKS
// ═══════════════════════════════════════════════════════════════════

static std::vector<ShekinahBlock> shekinah_annotate_popcount(
    const std::vector<ShekinahBlock>& blocks,
    const ShekinahEngineConfig& cfg)
{
    std::vector<ShekinahBlock> live;

    for (auto blk : blocks) {
        // Compute popcount of the locked (frozen) bits
        int locked_pop = 0;
        for (int bit = cfg.total_bits - 1; bit >= blk.free_bits; bit--) {
            if (blk.base.bit(bit)) locked_pop++;
        }
        blk.locked_popcount = locked_pop;
        blk.free_pop_lo = std::max(0, cfg.global_pop_lo - locked_pop);
        blk.free_pop_hi = std::min(blk.free_bits, cfg.global_pop_hi - locked_pop);

        // Kill dead blocks: no valid popcount range
        if (blk.free_pop_lo <= blk.free_pop_hi && blk.free_pop_hi >= 0) {
            live.push_back(blk);
        }
    }

    return live;
}

// ═══════════════════════════════════════════════════════════════════
// STEP 3: GPU CAPACITY SPLIT
// Blocks with free_bits > max_free_bits are split into sub-blocks
// by enumerating the excess bits and filtering by popcount.
// ═══════════════════════════════════════════════════════════════════

static std::vector<ShekinahBlock> shekinah_split_for_gpu(
    const ShekinahBlock& blk,
    const ShekinahEngineConfig& cfg)
{
    std::vector<ShekinahBlock> result;

    if (blk.free_bits <= cfg.max_free_bits) {
        result.push_back(blk);
        return result;
    }

    int excess = blk.free_bits - cfg.max_free_bits;
    // Safety: excess can be at most ~7 for puzzle 71 (71 - 64 = 7)
    // For larger puzzles, this could be more. Cap at 20 to avoid OOM.
    if (excess > 20) {
        printf("[Shekinah] WARNING: excess bits %d too large, capping split at 20\n", excess);
        excess = 20;
    }

    uint64_t num_patterns = 1ULL << excess;
    uint128_t sub_size = uint128_one() << cfg.max_free_bits;

    for (uint64_t pattern = 0; pattern < num_patterns; pattern++) {
        int pattern_pop = __builtin_popcountll(pattern);

        // Compute per-sub-block popcount constraint
        int sub_locked_pop = blk.locked_popcount + pattern_pop;
        int sub_free_pop_lo = std::max(0, cfg.global_pop_lo - sub_locked_pop);
        int sub_free_pop_hi = std::min(cfg.max_free_bits, cfg.global_pop_hi - sub_locked_pop);

        // Kill dead sub-blocks immediately
        if (sub_free_pop_lo > sub_free_pop_hi) continue;
        if (sub_free_pop_hi < 0) continue;
        if (sub_free_pop_lo > cfg.max_free_bits) continue;

        ShekinahBlock sub;
        // The pattern occupies bits [max_free_bits, max_free_bits+excess-1] of the free space
        uint128_t pattern128(0, pattern);
        sub.base = blk.base + (pattern128 << cfg.max_free_bits);
        sub.free_bits = cfg.max_free_bits;
        sub.locked_bits = cfg.total_bits - cfg.max_free_bits;
        sub.block_size = sub_size;
        sub.locked_popcount = sub_locked_pop;
        sub.free_pop_lo = sub_free_pop_lo;
        sub.free_pop_hi = sub_free_pop_hi;

        result.push_back(sub);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════
// STEP 4: ADAPTIVE BLOCK MERGING (MITM + Compute-Optimized)
// Merge tiny adjacent blocks to reduce kernel launch overhead.
// ═══════════════════════════════════════════════════════════════════

// Kernel launch overhead in EC-op equivalents (~250K ops)
static const uint64_t SHEKINAH_KERNEL_LAUNCH_OVERHEAD = 1ULL << 18;

static std::vector<ShekinahBlock> shekinah_adaptive_merge(
    const std::vector<ShekinahBlock>& blocks,
    const ShekinahEngineConfig& cfg,
    int min_useful_free_bits = 16)
{
    std::vector<ShekinahBlock> result;
    size_t n = blocks.size();
    size_t i = 0;

    while (i < n) {
        // Large enough blocks: keep as-is
        if (blocks[i].free_bits >= min_useful_free_bits) {
            result.push_back(blocks[i]);
            i++;
            continue;
        }

        // Found a small block. Scan forward for a run of small blocks.
        size_t run_start = i;
        while (i < n && blocks[i].free_bits < min_useful_free_bits) {
            i++;
        }
        size_t run_end = i;

        // Greedy merge: try largest window first
        size_t j = run_start;
        while (j < run_end) {
            bool found_merge = false;

            size_t max_window = std::min(run_end - j, (size_t)32);
            for (size_t window = max_window; window >= 2; window--) {
                // Compute merged block bounds
                uint128_t lo = blocks[j].base;
                uint128_t hi = blocks[j + window - 1].base + blocks[j + window - 1].block_size;
                uint128_t range = hi - lo;

                int merge_k = range.bit_width();
                if (merge_k > 0 && (uint128_one() << (merge_k - 1)) == range) {
                    merge_k = merge_k - 1; // exact power of 2
                }

                // Check alignment
                uint128_t align_mask = (uint128_one() << merge_k) - uint128_one();
                uint128_t aligned_lo = lo & ~align_mask;
                if (aligned_lo + (uint128_one() << merge_k) < hi) {
                    merge_k++;
                    align_mask = (uint128_one() << merge_k) - uint128_one();
                    aligned_lo = lo & ~align_mask;
                }

                // Can't exceed GPU max free bits
                if (merge_k > cfg.max_free_bits) continue;

                // Compute popcount for merged block
                int locked_pop = 0;
                for (int bit = cfg.total_bits - 1; bit >= merge_k; bit--) {
                    if (aligned_lo.bit(bit)) locked_pop++;
                }
                int m_free_pop_lo = std::max(0, cfg.global_pop_lo - locked_pop);
                int m_free_pop_hi = std::min(merge_k, cfg.global_pop_hi - locked_pop);

                if (m_free_pop_lo > m_free_pop_hi) continue;

                // Compute separate work
                uint64_t separate_work = 0;
                for (size_t idx = j; idx < j + window; idx++) {
                    MITMWorkEstimate we = shekinah_compute_mitm_work(blocks[idx]);
                    separate_work += we.total_ec_ops + SHEKINAH_KERNEL_LAUNCH_OVERHEAD;
                }

                // Compute merged work
                ShekinahBlock merged;
                merged.base = aligned_lo;
                merged.free_bits = merge_k;
                merged.locked_bits = cfg.total_bits - merge_k;
                merged.block_size = uint128_one() << merge_k;
                merged.locked_popcount = locked_pop;
                merged.free_pop_lo = m_free_pop_lo;
                merged.free_pop_hi = m_free_pop_hi;

                MITMWorkEstimate merged_we = shekinah_compute_mitm_work(merged);
                uint64_t merged_work = merged_we.total_ec_ops + SHEKINAH_KERNEL_LAUNCH_OVERHEAD;

                if (merged_work < separate_work) {
                    result.push_back(merged);
                    j += window;
                    found_merge = true;
                    break;
                }
            }

            if (!found_merge) {
                result.push_back(blocks[j]);
                j++;
            }
        }
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════
// STEP 5: BLOCK ORDERING FOR EARLY TERMINATION
// Center-first heuristic: blocks near the midpoint of the corridor
// are searched first, with larger blocks preferred as tiebreaker.
// ═══════════════════════════════════════════════════════════════════

static void shekinah_sort_blocks(
    std::vector<ShekinahBlock>& blocks,
    uint128_t corridor_start,
    uint128_t corridor_end)
{
    uint128_t mid = corridor_start + ((corridor_end - corridor_start) >> 1);

    std::sort(blocks.begin(), blocks.end(),
        [&mid](const ShekinahBlock& a, const ShekinahBlock& b) {
            // Distance from corridor midpoint
            uint128_t center_a = a.base + (a.block_size >> 1);
            uint128_t center_b = b.base + (b.block_size >> 1);

            uint128_t dist_a = (center_a > mid) ? (center_a - mid) : (mid - center_a);
            uint128_t dist_b = (center_b > mid) ? (center_b - mid) : (mid - center_b);

            if (dist_a != dist_b) return dist_a < dist_b; // closer to center first
            return a.free_bits > b.free_bits;              // larger blocks first
        });
}

// ═══════════════════════════════════════════════════════════════════
// DISPATCH COMMAND: What the GPU engine actually executes
// ═══════════════════════════════════════════════════════════════════

struct ShekinahDispatchCommand {
    // The base key value for this block (the locked prefix shifted into position)
    uint128_t base_key;

    // Number of free bits — determines the MITM split
    int free_bits;

    // The lock string (bit positions and values for the frozen prefix)
    // Stored as arrays for direct upload to GPU constant memory
    int  num_locked_positions;
    int  locked_positions[128];  // bit position in the full key
    int  locked_values[128];     // 0 or 1

    // Free bit positions (the bits that MITM will enumerate)
    int  num_free_positions;
    int  free_positions[128];

    // Per-block tightened popcount range
    int  pop_lo;         // global popcount min for this block
    int  pop_hi;         // global popcount max for this block

    // Diagnostics
    uint64_t ec_ops;
    uint64_t keys_covered;
    int      pop_subrounds;

    // Lock command string for display
    std::string lock_string;
};

// ═══════════════════════════════════════════════════════════════════
// THE COMPLETE SHEKINAH PIPELINE
// ═══════════════════════════════════════════════════════════════════

struct ShekinahPipelineResult {
    std::vector<ShekinahDispatchCommand> commands;
    std::vector<ShekinahBlock>           blocks;
    uint64_t grand_total_ops;
    uint64_t grand_total_keys;
    uint128_t corridor_start;
    uint128_t corridor_end;
};

static ShekinahPipelineResult shekinah_generate_dispatch(
    uint128_t A,           // start of multiplier corridor (inclusive)
    uint128_t B,           // end of multiplier corridor (exclusive)
    int       total_bits,  // e.g., 71 for puzzle 71
    int       global_pop_lo,
    int       global_pop_hi,
    int       max_gpu_free_bits = 64)
{
    shekinah_precompute_binomials();

    ShekinahEngineConfig cfg;
    cfg.max_free_bits = max_gpu_free_bits;
    cfg.total_bits = total_bits;
    cfg.global_pop_lo = global_pop_lo;
    cfg.global_pop_hi = global_pop_hi;

    ShekinahPipelineResult result;
    result.corridor_start = A;
    result.corridor_end = B;
    result.grand_total_ops = 0;
    result.grand_total_keys = 0;

    // ── Step 1: Dyadic decomposition ──
    printf("[Shekinah] Step 1: Dyadic decomposition of [A, B)...\n");
    uint128_t range = B - A;
    printf("[Shekinah]   Range size: ~2^%.2f\n", log2((double)range.hi * 1.8446744073709552e19 + (double)range.lo));
    fflush(stdout);

    std::vector<ShekinahBlock> raw = shekinah_decompose_range(A, B, total_bits);
    printf("[Shekinah]   Dyadic decomposition: %zu raw blocks\n", raw.size());

    // ── Step 2: Annotate popcount & kill dead blocks ──
    printf("[Shekinah] Step 2: Popcount annotation and pruning...\n");
    std::vector<ShekinahBlock> live = shekinah_annotate_popcount(raw, cfg);
    printf("[Shekinah]   After popcount pruning: %zu live blocks (killed %zu dead)\n",
           live.size(), raw.size() - live.size());

    // ── Step 3: Split oversized blocks ──
    printf("[Shekinah] Step 3: GPU capacity split (max %d free bits)...\n", cfg.max_free_bits);
    std::vector<ShekinahBlock> split;
    for (const auto& blk : live) {
        auto sub = shekinah_split_for_gpu(blk, cfg);
        for (auto& s : sub) split.push_back(s);
    }
    printf("[Shekinah]   After GPU splitting: %zu blocks\n", split.size());

    // ── Step 4: Adaptive merge tiny blocks ──
    printf("[Shekinah] Step 4: Adaptive block merging...\n");
    auto final_blocks = shekinah_adaptive_merge(split, cfg);
    printf("[Shekinah]   After adaptive merge: %zu final blocks\n", final_blocks.size());

    // ── Step 5: Sort for early termination ──
    printf("[Shekinah] Step 5: Block ordering (center-first heuristic)...\n");
    shekinah_sort_blocks(final_blocks, A, B);

    // ── Step 6: Generate dispatch commands ──
    printf("[Shekinah] Step 6: Generating dispatch commands...\n");
    result.blocks = final_blocks;

    for (const auto& blk : final_blocks) {
        ShekinahDispatchCommand cmd;
        cmd.base_key = blk.base;
        cmd.free_bits = blk.free_bits;
        cmd.num_locked_positions = 0;
        cmd.num_free_positions = 0;

        // Build lock positions (bits above free_bits)
        cmd.lock_string = "";
        for (int bit = total_bits - 1; bit >= blk.free_bits; bit--) {
            int val = blk.base.bit(bit);
            cmd.locked_positions[cmd.num_locked_positions] = bit;
            cmd.locked_values[cmd.num_locked_positions] = val;
            cmd.num_locked_positions++;

            if (!cmd.lock_string.empty()) cmd.lock_string += ",";
            cmd.lock_string += std::to_string(bit) + ":" + std::to_string(val);
        }

        // Free positions are bits [0, free_bits)
        for (int bit = 0; bit < blk.free_bits; bit++) {
            cmd.free_positions[cmd.num_free_positions++] = bit;
        }

        // Per-block tightened popcount range (global, including locked bits)
        cmd.pop_lo = blk.locked_popcount + blk.free_pop_lo;
        cmd.pop_hi = blk.locked_popcount + blk.free_pop_hi;

        MITMWorkEstimate we = shekinah_compute_mitm_work(blk);
        cmd.ec_ops = we.total_ec_ops;
        cmd.keys_covered = we.keys_covered;
        cmd.pop_subrounds = we.num_pop_subrounds;

        result.grand_total_ops += we.total_ec_ops;
        result.grand_total_keys += we.keys_covered;

        result.commands.push_back(cmd);
    }

    // ── Print summary ──
    printf("\n");
    printf("==========================================================\n");
    printf("  SHEKINAH MATRIX DISPATCH SUMMARY\n");
    printf("==========================================================\n");
    printf("  Total kernel launches:  %zu\n", result.commands.size());
    if (result.grand_total_ops > 0)
        printf("  Total EC operations:    ~2^%.2f\n", log2((double)result.grand_total_ops));
    if (result.grand_total_keys > 0)
        printf("  Total keys covered:     ~2^%.2f\n", log2((double)result.grand_total_keys));
    printf("  Ghost combinations:     ZERO\n");
    printf("==========================================================\n\n");

    // Print first 20 and last 5 blocks for diagnostics
    int print_count = std::min((int)result.commands.size(), 20);
    for (int i = 0; i < print_count; i++) {
        auto& cmd = result.commands[i];
        auto& blk = result.blocks[i];
        printf("# Block %d: %d free bits, %d pop sub-rounds, ",
               i, blk.free_bits, cmd.pop_subrounds);
        if (cmd.ec_ops > 0) printf("~2^%.1f EC ops, ", log2((double)cmd.ec_ops));
        if (cmd.keys_covered > 0) printf("~2^%.1f keys", log2((double)cmd.keys_covered));
        printf("\n");
        printf("  -lock \"%s\" -radiusrange 0:%d -poprange %d:%d\n\n",
               cmd.lock_string.c_str(), cmd.free_bits, cmd.pop_lo, cmd.pop_hi);
    }
    if ((int)result.commands.size() > 25) {
        printf("  ... (%zu more blocks) ...\n\n", result.commands.size() - 25);
        for (int i = (int)result.commands.size() - 5; i < (int)result.commands.size(); i++) {
            auto& cmd = result.commands[i];
            auto& blk = result.blocks[i];
            printf("# Block %d: %d free bits, %d pop sub-rounds\n",
                   i, blk.free_bits, cmd.pop_subrounds);
            printf("  -lock \"%s\" -radiusrange 0:%d -poprange %d:%d\n\n",
                   cmd.lock_string.c_str(), cmd.free_bits, cmd.pop_lo, cmd.pop_hi);
        }
    }

    fflush(stdout);
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// HELPER: Construct uint128_t from (lo, hi) pair
// ═══════════════════════════════════════════════════════════════════
static uint128_t make_u128(uint64_t lo, uint64_t hi) {
    return uint128_t(hi, lo);
}

#endif // SHEKINAH_MATRIX_H
