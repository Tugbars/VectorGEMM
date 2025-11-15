# VectorGEMM: Safety-Hardened Matrix Multiplication

A production-ready, safety-first GEMM (General Matrix Multiply) implementation achieving **171.9 GFLOPS** on Intel i9-14900 (single-core), with a **162× speedup** over naive implementations. Designed for embedded systems, numerical computing, and performance-critical applications where reliability matters as much as speed.

## Performance

**Intel i9-14900 (Single Core)**
```
=== Large: 1024x1024x1024 (3 runs) ===
  Naive:     2036.722 ms  (1.1 GFLOPS)
  Optimized:  12.495 ms  (171.9 GFLOPS)
  Speedup:   163.00x faster
```

**Benchmark Results (512×512×512)**
```
Naive:       52.0 ms  (5.2 GFLOPS)
Optimized:    1.6 ms  (163.8 GFLOPS)
Speedup:      31.74×
```

Achieves consistent performance across irregular sizes and non-square matrices through adaptive blocking and aspect-ratio-aware optimization.

## Features

### Safety-First Design
- **No `alignas` on stack arrays** — eliminates segfaults from misaligned stacks
- **No masked stores** — replaced with safe scalar loops (2-5% overhead on edge cases)
- **Always unaligned operations** for temporary buffers — defensive, portable
- **Debug assertions** for critical invariants (commented out in release builds)
- **Zero undefined behavior** — validated against UBSan/ASan
- **Comprehensive test suite** — 5 test suites + performance benchmark

### Performance Optimizations
- **Multi-tier architecture** with specialized paths for small, medium, and large matrices
- **SIMD-optimized packing** (1.5-2× faster than scalar)
- **K-loop unrolling** with interleaved FMAs to break dependency chains
- **Software pipelining** to hide memory latency
- **Pre-computed tile counts** — no division in hot paths
- **Adaptive blocking** based on matrix aspect ratios (tall/wide/deep)
- **Beta pre-scaling** — eliminates redundant operations

### Production Ready
- **6 test executables** covering all code paths
- Cross-platform (Windows/Linux, MSVC/GCC/Clang)
- AVX2/FMA3 optimized with SSE2 fallback paths
- Static and dynamic memory modes
- AddressSanitizer and Valgrind integration
- Configurable validation levels (0/1/2)

## Building

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/gemm.git
cd gemm

# Build in Release mode (recommended for performance)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run all tests
cd build && ctest --output-on-failure

# Run benchmark
./test_benchmark
```

### Build Requirements

- **CPU**: AVX2 + FMA3 (Intel Haswell/AMD Excavator or newer)
- **Compiler**: GCC 7+, Clang 8+, or MSVC 2019+
- **CMake**: 3.16 or newer
- **OS**: Linux, Windows, macOS

### Build Options

```bash
# Build with validation (debug builds)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGEMM_VALIDATION_LEVEL=2

# Build with AddressSanitizer
cmake -B build -DENABLE_ASAN=ON

# Build with Valgrind support (Linux)
cmake -B build -DENABLE_VALGRIND=ON

# Build without validation (release builds)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGEMM_VALIDATION_LEVEL=0
```

### Validation Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| 0 | No validation | Production/Release builds |
| 1 | Basic assertions | Development builds |
| 2 | Full instrumentation | Debug/Testing |

## Test Suite

The project includes 6 test executables covering all functionality:

### 1. Small Kernels Test (`test_gemm_small`)
Tests Tier 1 fixed-size kernels (4×4, 6×6, 8×8, etc.)
```bash
./test_gemm_small
```

### 2. Planning Module Test (`test_gemm_planning`)
Tests adaptive blocking and kernel selection
```bash
./test_gemm_planning
```

### 3. Individual Kernel Tests (`test_gemm_large`)
Unit tests for all micro-kernels (1×8, 4×8, 8×8, 8×6, etc.)
```bash
./test_gemm_large
```

### 4. Validated Kernels (`test_gemm_validated`)
Full instrumentation with validation level 2
```bash
./test_gemm_validated
```

### 5. Execution Pipeline (`test_gemm_execute`)
Integration tests for gemm_large.c orchestration
```bash
./test_gemm_execute
```

### 6. Performance Benchmark (`test_benchmark`)
Compares optimized vs naive implementation
```bash
./test_benchmark
```

### Unified Test Runner (`test_all`)
Runs all test suites in sequence
```bash
./test_all
```

## CMake Custom Targets

The build system provides convenient targets:

```bash
# Run all tests via CTest
make run_tests

# Run individual test suites
make run_small        # Small kernels (Tier 1)
make run_planning     # Planning module
make run_kernels      # Individual kernels
make run_validated    # Validated tests
make run_execute      # Execution pipeline
make run_unified      # All tests in sequence

# Performance benchmark
make run_benchmark

# Valgrind integration (if enabled)
make valgrind_execute
make valgrind_validated
```

## CTest Integration

```bash
# Run all tests
ctest --output-on-failure

# Run specific test
ctest -R SmallKernels
ctest -R Planning
ctest -R KernelTests
ctest -R ValidatedKernels
ctest -R ExecutePipeline
ctest -R AllTests

# Verbose output
ctest -V
```

## Architecture

### Three-Tier Design

```
┌─────────────────────────────────────────────────────────┐
│ Tier 1: Small Fixed-Size Kernels                        │
│ • 4×4, 6×6, 8×8 (square, fixed K)                      │
│ • 8×4, 4×8, 8×6, 6×8 (rectangular, variable K)         │
│ • K-outer loops with pre-scaled operands               │
│ • Handles M,N ≤ 16, K ≤ 64, FLOPs ≤ 8192              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Tier 2: Blocked Execution (NC→KC→MC)                    │
│ • Maximizes L2 cache reuse for B panels                │
│ • SIMD-optimized packing with alpha pre-scaling        │
│ • Pre-selected kernels for full tiles                  │
│ • Adaptive blocking (64-512 based on aspect ratio)     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Micro-Kernels (AVX2/FMA3)                               │
│ • 1×8, 4×8, 8×8, 8×6, 8×16, 16×8, 16×6 variants       │
│ • K-unroll by 2 with interleaved computation           │
│ • Register pressure: ≤16 YMM (carefully managed)       │
│ • Separate ADD/STORE variants for beta handling        │
└─────────────────────────────────────────────────────────┘
```

### Key Optimizations

**K-Loop Unrolling (2×)**
```c
// Before: Sequential dependency chain
for (k) { acc += A[k] * B[k]; }

// After: Interleaved computation (better ILP)
for (k+=2) {
  acc = fma(A[k+0], B[k+0], acc);
  acc = fma(A[k+1], B[k+1], acc);  // Independent!
}
```

**Software Pipelining**
```c
// Load next iteration while computing current
a0 = load(A + k);
a1 = load(A + k+1);  // Prefetch
b0 = load(B + k);
b1 = load(B + k+1);  // Prefetch

// Compute with both loaded
```

**SIMD Packing for A**
```c
// Gather 8 rows in one operation
__m256 v = _mm256_set_ps(
    src[7*K], src[6*K], ..., src[0*K]
);
```

**Pre-Scaled Alpha**
```c
// Traditional: 16K multiplies for 4×4
for (i,j,k) C[i,j] += alpha * A[i,k] * B[k,j];

// Optimized: 4K multiplies (alpha absorbed into B)
B_scaled = alpha * B;
for (i,j,k) C[i,j] += A[i,k] * B_scaled[k,j];
```

## Usage

### Basic API
```c
#include "gemm.h"

// Allocate aligned matrices
float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
float *C = gemm_aligned_alloc(64, M * N * sizeof(float));

// Compute: C = alpha*A*B + beta*C
gemm_auto(C, A, B, M, K, N, alpha, beta);

// Cleanup
gemm_aligned_free(A);
gemm_aligned_free(B);
gemm_aligned_free(C);
```

### Planned Execution (Amortize Planning Cost)
```c
// Create plan once
gemm_plan_t *plan = gemm_plan_create(M, K, N);

// Execute multiple times
for (int iter = 0; iter < 1000; iter++) {
    gemm_execute_plan(plan, C, A, B, alpha, beta);
}

gemm_plan_destroy(plan);
```

### Memory Modes
```c
// Static workspace (faster, limited size)
gemm_static(C, A, B, M, K, N, alpha, beta);

// Dynamic allocation (handles any size)
gemm_dynamic(C, A, B, M, K, N, alpha, beta);

// Check static workspace limit
if (gemm_fits_static(M, K, N)) {
    // Can use static mode
}
```

## Technical Details

### Register Pressure Management

Each micro-kernel is carefully designed to stay within AVX2's 16 YMM register limit:

| Kernel | Accumulators | Temps | Total | Status |
|--------|-------------|-------|-------|--------|
| 8×8    | 8           | 3     | 11    | ✅ Safe |
| 8×16   | 16          | 4     | 20    | ⚠️ Composite (2× 8×8) |
| 16×8   | 16          | 4     | 20    | ⚠️ Composite (2× 8×8) |

**Composite kernels** automatically split into multiple calls to avoid register spilling.

### Adaptive Blocking

The planner selects block sizes based on matrix shape:

```c
// Tall matrices (M >> N): Small NC, large MC
if (M/N > 3.0) {
    MC = 256; KC = 128; NC = 128;
}
// Wide matrices (N >> M): Small MC, large NC
else if (M/N < 0.33) {
    MC = 64; KC = 128; NC = 512;
}
// Deep matrices (K >> M,N): Large KC
else if (K/N > 4.0) {
    MC = 64; KC = 512; NC = 128;
}
```

### Cache Optimization

**NC→KC→MC Loop Order** maximizes L2 reuse:
1. Pack B once per KC×NC tile
2. Reuse packed B across all MC tiles
3. Minimize memory traffic (pack overhead ~5%)

### Safety Validation

All kernels validated against:
- **Naive reference implementation** (correctness)
- **AddressSanitizer** (heap/stack overflow)
- **UndefinedBehaviorSanitizer** (UB detection)
- **Valgrind** (memory leaks)

## Development Workflow

```bash
# Debug build with full validation
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug \
    -DGEMM_VALIDATION_LEVEL=2 -DENABLE_ASAN=ON
cmake --build build-debug

# Run validated tests with ASan
cd build-debug
./test_gemm_validated

# Release build for benchmarking
cmake -B build-release -DCMAKE_BUILD_TYPE=Release \
    -DGEMM_VALIDATION_LEVEL=0
cmake --build build-release

# Run benchmark
cd build-release
./test_benchmark

# Memory leak check (Linux)
cmake -B build-valgrind -DENABLE_VALGRIND=ON
cmake --build build-valgrind
cd build-valgrind && make valgrind_execute
```

## Directory Structure

```
gemm/
├── src/gemm_2/              # Core library
│   ├── gemm.h               # Main API
│   ├── gemm_kernels_avx2.h  # Micro-kernels
│   ├── gemm_large.c         # Tier 2 execution
│   ├── gemm_small.c         # Tier 1 kernels
│   ├── gemm_planning.c      # Adaptive planner
│   └── gemm_simd_ops.h      # SIMD operations
├── tests/                   # Test suite
│   ├── test_gemm_small.c    # Tier 1 tests
│   ├── test_planning.c      # Planner tests
│   ├── test_gemm_large.c    # Kernel tests
│   ├── test_gemm_validated.c # Validated tests
│   ├── test_gemm_execute.c  # Pipeline tests
│   ├── test_benchmark.c     # Performance
│   └── CMakeLists.txt       # Build system
└── README.md
```

## Known Issues

1. **`pack_A_panel_simd` buffer overflow** — If `ib > actual_mr`, writes out of bounds. Workaround: Planner ensures `ib ≤ MR`. Fix pending in next release.

2. **No AVX-512 support** — Currently limited to AVX2. AVX-512 kernels would achieve ~300 GFLOPS.

## Contributing

Contributions welcome! Please:
1. Run the full test suite (`make run_tests`)
2. Verify with AddressSanitizer (`-DENABLE_ASAN=ON`)
3. Ensure validation level 2 passes
4. Check performance regression with `test_benchmark`

## License

MIT License - See LICENSE file for details.

## Citation

If you use this code in research, please cite:
```bibtex
@software{gemm_safety_hardened,
  title = {Safety-Hardened GEMM Implementation},
  author = {TUGBARS},
  year = {2025},
  note = {Achieving 169.8 GFLOPS on Intel i9-14900}
}
```

## Author

**TUGBARS** - Embedded systems engineer specializing in numerical optimization and safety-critical code.

---

*Benchmarked on Intel i9-14900, single-threaded. Performance may vary based on CPU architecture, memory bandwidth, and compiler optimizations.*



