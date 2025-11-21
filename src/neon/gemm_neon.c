/**
 * @file gemm_embedded_neon_optimized.h
 * @brief Production-Optimized Embedded GEMM for ARM NEON
 * 
 * OPTIMIZATIONS APPLIED:
 * 1. K-loop unrolling (2×) with software pipelining
 * 2. Pointer arithmetic (no index multiplies in hot path)
 * 3. Lane-based FMA for reduced register pressure
 * 4. Interleaved packing for better L1 streaming
 * 5. Integrated beta scaling (no separate pass)
 * 6. Auto-tuned blocking based on cache sizes
 * 7. Early exit for alpha=0
 * 
 * PERFORMANCE TARGETS:
 * - Cortex-A53: 5-7 GFLOPS (85% peak)
 * - Cortex-A72: 12-15 GFLOPS (85% peak)
 * - Cortex-A76: 15-18 GFLOPS (80% peak)
 */

#ifndef GEMM_EMBEDDED_NEON_OPT_H
#define GEMM_EMBEDDED_NEON_OPT_H

#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

//==============================================================================
// CORRECTED: AUTO-TUNED BLOCKING WITH WORKSPACE SAFETY
//==============================================================================

// Micro-kernel dimensions
#define GEMM_MR 4
#define GEMM_NR 4

// Workspace limits (compile-time maximum)
#define GEMM_MAX_KC 512
#define GEMM_MAX_MC 256

// Calculate KC from L1 (unchanged)
#define GEMM_KC_TARGET ((L1D_CACHE_SIZE * 4) / (5 * (GEMM_MR + GEMM_NR) * 4))
#define GEMM_KC ((GEMM_KC_TARGET + 31) & ~31)

// Calculate MC from L1 (with floor to prevent tiny values)
#define GEMM_MC_TARGET ((L1D_CACHE_SIZE * 4) / (5 * GEMM_KC * 4))
#define GEMM_MC_RAW ((GEMM_MC_TARGET + GEMM_MR - 1) & ~(GEMM_MR - 1))

// ✅ FIX: Ensure MC is at least MR (prevents tiny values)
#if GEMM_MC_RAW < GEMM_MR
  #define GEMM_MC GEMM_MR
#else
  #define GEMM_MC GEMM_MC_RAW
#endif

// ✅ FIX: NC must be constrained by B workspace capacity
// B workspace holds: n_panels × KC × NR floats
// Maximum n_panels = GEMM_WORKSPACE_B_SIZE / (GEMM_MAX_KC × GEMM_NR)
#define GEMM_WORKSPACE_B_SIZE (GEMM_MAX_KC * 64) // 64 columns = 16 panels

#define GEMM_NC_FROM_L2 (L2_CACHE_SIZE / (GEMM_KC * 4 + 1024))
#define GEMM_NC_FROM_WORKSPACE ((GEMM_WORKSPACE_B_SIZE / GEMM_MAX_KC) & ~(GEMM_NR - 1))

// Use the smaller of L2-derived or workspace-constrained
#if GEMM_NC_FROM_L2 > GEMM_NC_FROM_WORKSPACE
  #define GEMM_NC GEMM_NC_FROM_WORKSPACE
#else
  #define GEMM_NC ((GEMM_NC_FROM_L2 + GEMM_NR - 1) & ~(GEMM_NR - 1))
#endif

// Workspace A sizing (unchanged)
#define GEMM_WORKSPACE_A_SIZE (GEMM_MAX_MC * GEMM_MAX_KC)

//==============================================================================
// STATIC WORKSPACE (64-byte aligned for Cortex-A72/A76)
//==============================================================================

static float __attribute__((aligned(64))) 
    g_workspace_a[GEMM_WORKSPACE_A_SIZE];

static float __attribute__((aligned(64))) 
    g_workspace_b[GEMM_WORKSPACE_B_SIZE];

//==============================================================================
// PREFETCH MACROS
//==============================================================================

#define PREFETCH_L1(addr) __builtin_prefetch((addr), 0, 3)
#define PREFETCH_L2(addr) __builtin_prefetch((addr), 0, 2)
#define PREFETCH_W(addr)  __builtin_prefetch((addr), 1, 3)

//==============================================================================
// OPTIMIZED PACKING (Interleaved layout)
//==============================================================================

/**
 * @brief Pack A panel with K-major layout: Ap[k*MR + i] = A[i][k] * alpha
 * 
 * CRITICAL: Kernel expects K-major layout for contiguous vector loads:
 *   float32x4_t a = vld1q_f32(Ap + k * GEMM_MR);
 * 
 * Layout:
 *   Ap[0..3]   = A[i0+0..i0+3][k0+0]  (K=0, all 4 rows)
 *   Ap[4..7]   = A[i0+0..i0+3][k0+1]  (K=1, all 4 rows)
 *   Ap[8..11]  = A[i0+0..i0+3][k0+2]  (K=2, all 4 rows)
 *   ...
 * 
 * This is column-major in the K-dimension, allowing kernel to:
 * - Load 4 rows at once with single vld1q_f32()
 * - Increment pointer by MR per K-iteration
 * 
 * OPTIMIZATIONS:
 * - Prefetching every 16 K-iterations
 * - NEON gather for m_panel=4 (full width)
 * - Pointer arithmetic (no index multiplies)
 */
static inline void pack_A_kmajor_neon(
    float * restrict Ap,
    const float * restrict A,
    size_t lda,
    size_t i0, size_t m_panel,
    size_t k0, size_t k_panel,
    float alpha)
{
    const float *src_base = A + i0 * lda + k0;
    
    // Zero-fill if partial panel (outside loop for efficiency)
    if (m_panel < GEMM_MR) {
        memset(Ap, 0, k_panel * GEMM_MR * sizeof(float));
    }

    float *dst_ptr = Ap;  // Pointer-chasing for output
    
    //==========================================================================
    // FAST PATH: alpha = 1.0 (no scaling)
    //==========================================================================
    if (alpha == 1.0f) {
        for (size_t k = 0; k < k_panel; ++k) {
            // Prefetch ahead (every 16 iterations to avoid over-prefetching)
            if ((k & 15) == 0 && k + 16 < k_panel) {
                PREFETCH_L1(src_base + 16);
            }
            
            const float *src_col = src_base + k;  // Points to A[i0][k0+k]
            
            if (m_panel == GEMM_MR) {
                // Full width: NEON gather (4 strided loads)
                // Build vector: {A[i0+0][k], A[i0+1][k], A[i0+2][k], A[i0+3][k]}
                float32x4_t col;
                col = vsetq_lane_f32(src_col[0 * lda], col, 0);
                col = vsetq_lane_f32(src_col[1 * lda], col, 1);
                col = vsetq_lane_f32(src_col[2 * lda], col, 2);
                col = vsetq_lane_f32(src_col[3 * lda], col, 3);
                vst1q_f32(dst_ptr, col);
            } else {
                // Partial width: Scalar gather with zero-fill
                for (size_t i = 0; i < m_panel; ++i) {
                    dst_ptr[i] = src_col[i * lda];
                }
                // Remaining elements already zeroed by memset
            }
            
            dst_ptr += GEMM_MR;  // Move to next K-iteration
        }
    }
    //==========================================================================
    // GENERAL PATH: alpha ≠ 1.0 (with scaling)
    //==========================================================================
    else {
        float32x4_t valpha = vdupq_n_f32(alpha);
        
        for (size_t k = 0; k < k_panel; ++k) {
            if ((k & 15) == 0 && k + 16 < k_panel) {
                PREFETCH_L1(src_base + 16);
            }
            
            const float *src_col = src_base + k;
            
            if (m_panel == GEMM_MR) {
                // Full width: NEON gather + scale
                float32x4_t col;
                col = vsetq_lane_f32(src_col[0 * lda], col, 0);
                col = vsetq_lane_f32(src_col[1 * lda], col, 1);
                col = vsetq_lane_f32(src_col[2 * lda], col, 2);
                col = vsetq_lane_f32(src_col[3 * lda], col, 3);
                col = vmulq_f32(col, valpha);
                vst1q_f32(dst_ptr, col);
            } else {
                // Partial width: Scalar gather + scale
                for (size_t i = 0; i < m_panel; ++i) {
                    dst_ptr[i] = src_col[i * lda] * alpha;
                }
            }
            
            dst_ptr += GEMM_MR;
        }
    }
}

/**
 * @brief Pack B panel with row-major layout: Bp[k*NR + j] = B[k][j]
 * 
 * Layout (unchanged, already correct):
 *   Bp[0..3]   = B[k0+0][j0..j0+3]  (K=0, all 4 cols)
 *   Bp[4..7]   = B[k0+1][j0..j0+3]  (K=1, all 4 cols)
 *   ...
 * 
 * Kernel broadcasts from this: vdupq_n_f32(Bp[k*NR + j])
 * 
 * ✅ WORKSPACE SAFETY: Caller ensures n_panels * k_block * NR ≤ GEMM_WORKSPACE_B_SIZE
 */
static inline void pack_B_neon(
    float * restrict Bp,
    const float * restrict B,
    size_t ldb,
    size_t k0, size_t k_panel,
    size_t j0, size_t n_panel)
{
    // Zero-fill if partial panel
    if (n_panel < GEMM_NR) {
        memset(Bp, 0, k_panel * GEMM_NR * sizeof(float));
    }

    const float *src_base = B + k0 * ldb + j0;
    float *dst_ptr = Bp;
    
    for (size_t k = 0; k < k_panel; ++k) {
        // Prefetch during packing (every 8 iterations)
        if ((k & 7) == 0 && k + 8 < k_panel) {
            PREFETCH_L1(src_base + 8 * ldb);
        }
        
        const float *src_row = src_base + k * ldb;
        
        if (n_panel == GEMM_NR) {
            // Full width: SIMD copy
            float32x4_t vec = vld1q_f32(src_row);
            vst1q_f32(dst_ptr, vec);
        } else {
            // Partial width: Scalar copy
            for (size_t j = 0; j < n_panel; ++j) {
                dst_ptr[j] = src_row[j];
            }
        }
        
        dst_ptr += GEMM_NR;
    }
}


//==============================================================================
// OPTIMIZED 4×4 MICRO-KERNEL (K-unrolled, pointer-chasing)
//==============================================================================

/**
 * @brief 4×4 micro-kernel with K-unroll and lane-based FMA
 * 
 * OPTIMIZATIONS:
 * - K-loop unrolled by 2 (hides 3-cycle FMA latency)
 * - Pointer arithmetic (no index multiplies)
 * - Lane-based FMA using vfmaq_laneq_f32 (reduces moves)
 * - Direct lane extraction for writeback (no temp buffer)
 * 
 * Register usage: 4 accs + 2 A + 1 B = 7 (plenty of headroom)
 */
static inline void gemm_kernel_4x4_add_opt(
    float * restrict C,
    size_t ldc,
    const float * restrict Ap,
    const float * restrict Bp,
    size_t K,
    float beta_eff) // Effective beta for this tile
{
    // Column accumulators
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);
    float32x4_t c3 = vdupq_n_f32(0.0f);

    // Pointer-chasing (feedback 5A)
    const float *a_ptr = Ap;
    const float *b_ptr = Bp;

    //==========================================================================
    // MAIN K-LOOP: Unrolled by 2 (feedback 1C)
    //==========================================================================
    size_t k = 0;
    for (; k + 1 < K; k += 2) {
        // Load A and B for both K-iterations (software pipelining)
        float32x4_t a0 = vld1q_f32(a_ptr);
        float32x4_t a1 = vld1q_f32(a_ptr + GEMM_MR);
        
        float32x4_t b0 = vld1q_f32(b_ptr);
        float32x4_t b1 = vld1q_f32(b_ptr + GEMM_NR);
        
        // Lane-based FMA (feedback 1D) - reduces register moves
        // c0 += a0 * b0[0], then c0 += a1 * b1[0]
        c0 = vfmaq_laneq_f32(c0, a0, b0, 0);
        c0 = vfmaq_laneq_f32(c0, a1, b1, 0);
        
        c1 = vfmaq_laneq_f32(c1, a0, b0, 1);
        c1 = vfmaq_laneq_f32(c1, a1, b1, 1);
        
        c2 = vfmaq_laneq_f32(c2, a0, b0, 2);
        c2 = vfmaq_laneq_f32(c2, a1, b1, 2);
        
        c3 = vfmaq_laneq_f32(c3, a0, b0, 3);
        c3 = vfmaq_laneq_f32(c3, a1, b1, 3);
        
        a_ptr += 2 * GEMM_MR;
        b_ptr += 2 * GEMM_NR;
    }

    //==========================================================================
    // TAIL: Handle odd K
    //==========================================================================
    if (k < K) {
        float32x4_t a = vld1q_f32(a_ptr);
        float32x4_t b = vld1q_f32(b_ptr);
        
        c0 = vfmaq_laneq_f32(c0, a, b, 0);
        c1 = vfmaq_laneq_f32(c1, a, b, 1);
        c2 = vfmaq_laneq_f32(c2, a, b, 2);
        c3 = vfmaq_laneq_f32(c3, a, b, 3);
    }

    //==========================================================================
    // WRITEBACK: Direct lane extraction (feedback 1A)
    //==========================================================================
    // Pointer-chasing for C too (feedback 5A)
    float *c_col0 = C;
    float *c_col1 = C + 1;
    float *c_col2 = C + 2;
    float *c_col3 = C + 3;
    
    if (beta_eff == 0.0f) {
        // STORE mode: Direct write (no load)
        for (size_t i = 0; i < 4; ++i) {
            *c_col0 = vgetq_lane_f32(c0, i);
            *c_col1 = vgetq_lane_f32(c1, i);
            *c_col2 = vgetq_lane_f32(c2, i);
            *c_col3 = vgetq_lane_f32(c3, i);
            
            c_col0 += ldc;
            c_col1 += ldc;
            c_col2 += ldc;
            c_col3 += ldc;
        }
    } else if (beta_eff == 1.0f) {
        // ADD mode: Load-add-store
        for (size_t i = 0; i < 4; ++i) {
            *c_col0 += vgetq_lane_f32(c0, i);
            *c_col1 += vgetq_lane_f32(c1, i);
            *c_col2 += vgetq_lane_f32(c2, i);
            *c_col3 += vgetq_lane_f32(c3, i);
            
            c_col0 += ldc;
            c_col1 += ldc;
            c_col2 += ldc;
            c_col3 += ldc;
        }
    } else {
        // Scaled ADD: Load, scale, add, store
        for (size_t i = 0; i < 4; ++i) {
            *c_col0 = *c_col0 * beta_eff + vgetq_lane_f32(c0, i);
            *c_col1 = *c_col1 * beta_eff + vgetq_lane_f32(c1, i);
            *c_col2 = *c_col2 * beta_eff + vgetq_lane_f32(c2, i);
            *c_col3 = *c_col3 * beta_eff + vgetq_lane_f32(c3, i);
            
            c_col0 += ldc;
            c_col1 += ldc;
            c_col2 += ldc;
            c_col3 += ldc;
        }
    }
}

/**
 * @brief Edge kernel: Partial tiles (scalar fallback)
 */
static inline void gemm_kernel_edge_neon(
    float * restrict C,
    size_t ldc,
    const float * restrict Ap,
    const float * restrict Bp,
    size_t K,
    size_t m, size_t n,
    float beta_eff)
{
    float temp[4][4] = {{0}};
    
    // Accumulate
    for (size_t k = 0; k < K; ++k) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                temp[i][j] += Ap[k * GEMM_MR + i] * Bp[k * GEMM_NR + j];
            }
        }
    }
    
    // Writeback with beta
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (beta_eff == 0.0f) {
                C[i * ldc + j] = temp[i][j];
            } else if (beta_eff == 1.0f) {
                C[i * ldc + j] += temp[i][j];
            } else {
                C[i * ldc + j] = C[i * ldc + j] * beta_eff + temp[i][j];
            }
        }
    }
}

//==============================================================================
// MAIN OPTIMIZED GEMM ENGINE
//==============================================================================

int gemm_embedded_neon_optimized(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc, size_t lda, size_t ldb,
    float alpha, float beta)
{
    //==========================================================================
    // EARLY EXIT: alpha = 0 (feedback 5D)
    //==========================================================================
    if (alpha == 0.0f) {
        // Just scale C by beta
        if (beta == 0.0f) {
            for (size_t i = 0; i < M; ++i) {
                memset(C + i * ldc, 0, N * sizeof(float));
            }
        } else if (beta != 1.0f) {
            float32x4_t vbeta = vdupq_n_f32(beta);
            for (size_t i = 0; i < M; ++i) {
                float *row = C + i * ldc;
                size_t j = 0;
                for (; j + 3 < N; j += 4) {
                    float32x4_t c = vld1q_f32(row + j);
                    vst1q_f32(row + j, vmulq_f32(c, vbeta));
                }
                for (; j < N; ++j) {
                    row[j] *= beta;
                }
            }
        }
        return 0;
    }

    // Validate workspace
    if (K > GEMM_MAX_KC || M > GEMM_MAX_MC) {
        return -1;
    }

    float *Ap = g_workspace_a;
    float *Bp = g_workspace_b;

    //==========================================================================
    // INTEGRATED BETA SCALING (feedback 5C)
    // Instead of separate pre-scaling pass, track per-tile beta state
    //==========================================================================
    
    //==========================================================================
    // NC LOOP
    //==========================================================================
    for (size_t jj = 0; jj < N; jj += GEMM_NC) {
        size_t n_block = (jj + GEMM_NC <= N) ? GEMM_NC : (N - jj);

        //======================================================================
        // KC LOOP
        //======================================================================
        for (size_t kk = 0; kk < K; kk += GEMM_KC) {
            size_t k_block = (kk + GEMM_KC <= K) ? GEMM_KC : (K - kk);
            
            // Integrated beta scaling (feedback 5C)
            // First K-tile: Use beta directly
            // Later K-tiles: Always accumulate (beta_eff = 1.0)
            float beta_eff = (kk == 0) ? beta : 1.0f;

            //==================================================================
            // Pack B panels
            //==================================================================
            size_t n_panels = (n_block + GEMM_NR - 1) / GEMM_NR;
            
            for (size_t p = 0; p < n_panels; ++p) {
                size_t j = jj + p * GEMM_NR;
                size_t n_panel = (j + GEMM_NR <= jj + n_block) 
                                 ? GEMM_NR : (jj + n_block - j);
                
                float *Bp_panel = Bp + p * k_block * GEMM_NR;
                pack_A_kmajor_neon(Bp_panel, B, ldb, kk, k_block, 
                                        j, n_panel);
            }

            //==================================================================
            // MC LOOP
            //==================================================================
            for (size_t ii = 0; ii < M; ii += GEMM_MC) {
                size_t m_block = (ii + GEMM_MC <= M) ? GEMM_MC : (M - ii);

                //==============================================================
                // MR LOOP
                //==============================================================
                for (size_t i = ii; i < ii + m_block; i += GEMM_MR) {
                    size_t m_panel = (i + GEMM_MR <= M) ? GEMM_MR : (M - i);

                    // Pack A panel
                    pack_A_interleaved_neon(Ap, A, lda, i, m_panel, 
                                           kk, k_block, alpha);

                    //==========================================================
                    // NR LOOP: Process N-panels
                    //==========================================================
                    for (size_t p = 0; p < n_panels; ++p) {
                        size_t j = jj + p * GEMM_NR;
                        size_t n_panel = (j + GEMM_NR <= jj + n_block) 
                                         ? GEMM_NR : (jj + n_block - j);

                        float *Bp_panel = Bp + p * k_block * GEMM_NR;
                        float *Cptr = C + i * ldc + j;

                        // Dispatch
                        if (m_panel == GEMM_MR && n_panel == GEMM_NR) {
                            // Full tile: Optimized kernel
                            gemm_kernel_4x4_add_opt(Cptr, ldc, Ap, 
                                                   Bp_panel, k_block, 
                                                   beta_eff);
                        } else {
                            // Edge tile: Scalar kernel
                            gemm_kernel_edge_neon(Cptr, ldc, Ap, Bp_panel, 
                                                 k_block, m_panel, n_panel,
                                                 beta_eff);
                        }
                    }
                }
            }
        }
    }

    return 0;
}

//==============================================================================
// CONVENIENCE API
//==============================================================================

int gemm_neon_optimized(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    return gemm_embedded_neon_optimized(C, A, B, M, K, N, N, K, N, alpha, beta);
}

#endif // GEMM_EMBEDDED_NEON_OPT_H
