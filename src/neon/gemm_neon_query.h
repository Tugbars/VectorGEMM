/**
 * @file gemm_neon_query.c
 * @brief Query functions for GEMM NEON library
 */

#include "gemm_neon.h"
#include <stdio.h>

// Forward declare blocking parameters (calculated in main implementation)
extern const size_t GEMM_MC, GEMM_KC, GEMM_NC;

void gemm_get_blocking_params(size_t *MC, size_t *KC, size_t *NC)
{
    if (MC) *MC = GEMM_MC;
    if (KC) *KC = GEMM_KC;
    if (NC) *NC = GEMM_NC;
}

void gemm_get_workspace_limits(size_t *max_M, size_t *max_K, size_t *max_N)
{
    if (max_M) *max_M = GEMM_MAX_MC;
    if (max_K) *max_K = GEMM_MAX_KC;
    if (max_N) *max_N = 16384; // Recommended, not a hard limit
}

int gemm_can_compute(size_t M, size_t K, size_t N)
{
    (void)N; // N is not limited
    return (K <= GEMM_MAX_KC && M <= GEMM_MAX_MC) ? 1 : 0;
}

const char* gemm_get_platform_info(void)
{
    static char buf[128];
    snprintf(buf, sizeof(buf),
             "ARMv%d - L1: %d KB, L2: %d KB",
#if defined(__ARM_ARCH_8A__) || defined(__aarch64__)
             8,
#else
             7,
#endif
             L1D_CACHE_SIZE / 1024,
             L2_CACHE_SIZE / 1024);
    return buf;
}

const char* gemm_get_config_info(void)
{
    static char buf[128];
    snprintf(buf, sizeof(buf),
             "K-unroll: 2x, Prefetch: %s, Max K: %d, Max M: %d",
             GEMM_PREFETCH_ENABLED ? "ON" : "OFF",
             GEMM_MAX_KC,
             GEMM_MAX_MC);
    return buf;
}

const char* gemm_get_version(void)
{
    return "2.0.0";
}