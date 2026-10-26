# TACO-UnifaSVD

This repository is a standalone bundle for the CUDA implementation of **UnifaSVD**, an adaptive mixed-precision low-rank SVD system, together with cuSOLVER-based baseline implementations for comparison.

Last updated: 2026-10-26.

## Contents

- `UnifaSVD.cu` — Main implementation of the UnifaSVD system (renamed from `AP_HQB-SVD-HGTSQR-sys.cu`).
- `cusolverDnXgesvd.cu` — cuSOLVER `cusolverDnXgesvd` baseline (full SVD).
- `cusolverDnXgesvdr.cu` — cuSOLVER `cusolverDnXgesvdr` baseline (randomized SVD).
- `src/` — Local dependencies required by the above targets.
- `dataset/` — Small LIBSVM-style sparse datasets used by the matrix-file experiments.
- `Makefile` — Builds all three executables.

## Build

```bash
cd /path/to/TACO-UnifaSVD
make
```

This produces:

```bash
./UnifaSVD
./cusolverDnXgesvd
./cusolverDnXgesvdr
```

## Program Modes

### 1. Synthetic low-rank matrix

```bash
./UnifaSVD 16384 16384 64 1024 1e-6 geometric 1 1e-6 float
```

Meaning:

- `m = 16384`
- `n = 16384`
- `k = 64`
- `rank = 1024`
- `eps = 1e-6`
- spectrum = `geometric`
- singular value range = `[1, 1e-6]`
- compute precision = `float`

### 2. User-provided matrix file

```bash
./UnifaSVD auto auto 64 auto 1e-7 float \
  --matrix-file=dataset/svmguide4.txt \
  --matrix-format=libsvm \
  --unknown-rank --debug
```

For matrix files, `spectrum_type`, `s_max`, and `s_min` are not used.

## Useful Options

- `--debug` — Enables verification and diagnostic output.
- `--fast` — Prefers faster float-heavy paths when possible.
- `--accuracy=low|medium|high` — Controls accuracy/performance tradeoff.
- `--vectors=S|U|Vt|US|SVt|All` — Controls which singular vectors are computed.
- `--energy-tol=<tau>` — Adaptive-stop threshold for captured energy in unknown-rank mode.
- `--tail-norm-eps=<eps>` — Adaptive-stop threshold based on row-tail decay.
- `--force-full-rank` — Keeps full panel accumulation.
- `--allow-rank-drop` — Allows adaptive panel truncation.

## Baselines

### Full SVD via cuSOLVER

```bash
./cusolverDnXgesvd 4096 4096 512 geometric 1 1e-6 float
```

### Randomized SVD via cuSOLVER

```bash
./cusolverDnXgesvdr 4096 4096 512 2 geometric 1 1e-6 float
```

Argument order for `cusolverDnXgesvdr` is generally:

```bash
./cusolverDnXgesvdr <m> <n> <rank> <niters> <spectrum> <s_max> <s_min> <precision>
```

## Provenance

This bundle was prepared from:

- `/home/sl/nfs_data/Arbitrary precision SVD/AP_HQB-SVD-HGTSQR-sys.cu`
- `/home/sl/nfs_data/Arbitrary precision SVD/cusolverDnXgesvd.cu`
- `/home/sl/nfs_data/Arbitrary precision SVD/cusolverDnXgesvdr.cu`
- `/home/sl/nfs_data/Arbitrary precision SVD/src/`
- `/home/sl/nfs_data/SVD/verify_results.h`
- `/home/sl/nfs_data/SVD/my_cuda_kernels_float.h`
