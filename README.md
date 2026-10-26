# TACO-UnifaSVD

CUDA research code for **UnifaSVD**, an adaptive mixed-precision low-rank singular
value decomposition (SVD) implementation, with full and randomized cuSOLVER
baselines. The executables generate synthetic matrices or load supported matrix
files and print timing and numerical diagnostics.

## Requirements and build

- Linux, GNU Make, and a C++17-compatible host compiler supported by the CUDA toolkit.
- NVIDIA CUDA toolkit with cuBLAS, cuBLASLt, cuSOLVER, and cuRAND.
  Compilation is checked with CUDA 12.8; other versions need separate validation.
- An NVIDIA GPU and a compatible driver for numerical runs. Default build targets
  are `sm_80`, `sm_89`, and `sm_90`, with forward-compatible `compute_90` PTX.
- Python 3 for the optional host/command-line regression tests.

```bash
git clone https://github.com/owo75/TACO-UnifaSVD.git
cd TACO-UnifaSVD
make -j2 CUDA_HOME=/usr/local/cuda-12.8
```

The build produces `UnifaSVD`, `cusolverDnXgesvd`, and `cusolverDnXgesvdr`.
Use `make clean` before changing toolkit, compiler flags, or target architectures.
Header and included CUDA-source dependencies are tracked automatically.

For a single target architecture, override `ARCH_FLAGS`, for example:

```bash
make -j2 CUDA_HOME=/usr/local/cuda-12.8 \
  ARCH_FLAGS='-gencode arch=compute_80,code=sm_80'
```

If CUDA shared libraries are not on the system loader path, set their location
before running a program or the tests:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## UnifaSVD usage

Run `./UnifaSVD --help` for supported options. `--help` and `-h` exit successfully
without initializing CUDA; invalid arguments return a nonzero exit status.

### Synthetic matrices

```text
./UnifaSVD <m> <n> <k> <rank|auto> <eps> <spectrum> <s_max> <s_min> <precision> [options]
```

```bash
./UnifaSVD 1024 1024 32 128 1e-6 geometric 1 1e-4 float --debug
```

| Argument | Meaning |
| --- | --- |
| `m`, `n` | Matrix rows and columns; positive integers within the 32-bit dimension limit. |
| `k` | QB panel width; this is distinct from the target rank. |
| `rank` | Synthetic matrix rank; in unknown-rank mode, also the search upper bound. Must not exceed `min(m,n)`. |
| `eps` | Positive QB panel threshold. **It is not a guaranteed final SVD reconstruction-error tolerance.** |
| `spectrum` | `geometric`, `uniform`, `cluster0`, `cluster1`, `arithmetic`, or `normal`. |
| `s_max`, `s_min` | Positive spectrum parameters with `s_max >= s_min`; the selected generator determines the distribution. |
| `precision` | `float`, `double`, `half`, or `bfloat16`; internal stages may use higher precision. |

By default, fixed-rank runs keep full panels, so `eps` does not truncate them.
Use `--allow-rank-drop` to enable panel rank truncation. Full-rank inputs and
matrices with `min(m,n) < 512` select the direct SVD path; unknown-rank mode
selects the adaptive QB path. The selected route is printed at startup.

### Matrix files and unknown rank

Named arguments are supported for file input:

```bash
./UnifaSVD --matrix-file=dataset/svmguide4.txt --matrix-format=libsvm \
  --k=8 --rank=auto --eps=1e-6 --precision=float --debug
```

The equivalent positional interface is:

```bash
./UnifaSVD auto auto 8 auto 1e-6 float \
  --matrix-file=dataset/svmguide4.txt --matrix-format=libsvm --debug
```

With file input, omit `spectrum`, `s_max`, and `s_min`. Text and LIBSVM files
support `auto` dimension inference. Binary files require explicit dimensions.
Explicit dimensions supplied alongside `auto` must match the inferred shape.

`rank=auto` uses `min(m,n)` as the upper bound and enables unknown-rank mode.
`--unknown-rank` enables the same adaptive mode with an explicit rank upper bound.
The default energy and row-tail stopping thresholds are derived from `eps`;
these heuristics do not certify a final reconstruction-error bound.

### Main options

| Option | Behavior |
| --- | --- |
| `--debug` | Enable additional verification and diagnostics; this adds work. |
| `--fast` | Prefer FP32 internal operations for float runs when supported by the route. |
| `--accuracy=low\|medium\|high` | Select SVD refinement checks; default `medium`. |
| `--vectors=S\|U\|Vt\|US\|SVt\|All` | Select the requested singular values/vectors; default `All`. |
| `--force-full-rank` | Keep full QB panels; this is the fixed-rank default. |
| `--allow-rank-drop` | Allow panel truncation; enabled by default in unknown-rank mode. |
| `--energy-tol=<tau>` | Adaptive captured-energy threshold, `0 < tau < 1`. |
| `--tail-norm-eps=<eps>` | Positive adaptive row-tail threshold. |
| `--padding=on\|off` | Enable/disable row padding required by some kernels; default `on`. |

## cuSOLVER baselines

Both baselines generate synthetic matrices. Their `rank` argument sets the
generated matrix rank; the randomized baseline also uses it as the requested
approximation rank. The full-SVD baseline computes the full decomposition and
requires `m >= n`.

```text
./cusolverDnXgesvd <m> <n> <rank> <spectrum> <s_max> <s_min> [--precision=<type>]
./cusolverDnXgesvdr <m> <n> <rank|auto> <spectrum> <s_max> <s_min> [--precision=<type>] [--niters=<N>]
```

```bash
./cusolverDnXgesvd 1024 1024 128 geometric 1 1e-4 --precision=float
./cusolverDnXgesvdr 1024 1024 128 geometric 1 1e-4 --precision=float --niters=2
```

The randomized baseline supports matrix-file input as well:

```bash
./cusolverDnXgesvdr auto auto 8 \
  --matrix-file=dataset/svmguide4.txt --matrix-format=libsvm \
  --precision=float --niters=2
```

`--niters` is a named option, defaults to 10, and must be a positive integer.
Oversampling is `min(2 * rank, min(m,n) - rank)`. Precision is selected with
`--precision=float|half|bfloat16`, rather than a trailing positional token.

**Baseline precision semantics:** all cuSOLVER factorizations in these programs
use FP32. `half` and `bfloat16` quantize the input and convert it back to FP32
before factorization; output conversion is timed separately. Verification uses
the FP32 factors. These modes measure quantization effects and conversion costs,
and do not run a native FP16/BF16 cuSOLVER SVD.

For API details, see the [NVIDIA cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/).

## Matrix-file formats

| Format | File contents | Shape and layout |
| --- | --- | --- |
| `text` (default) | Whitespace-separated numeric values, without a header. | For `auto`, each nonempty line must have equal length. Default `row` layout means one matrix row per line; `col` means one column per line. |
| `binary` | Headerless native floating-point values. | Explicit `m`, `n`, and matching input element type are required; `--matrix-layout=row\|col` selects the ordering. |
| `libsvm` | One sample per line: a label followed by 1-based `index:value` pairs. | Labels are ignored; omitted features are zero. Rows are samples and columns are feature indices. Blank lines are skipped. |

All file inputs are expanded into dense column-major matrices before processing.
Sparse input therefore does not imply sparse GPU memory use. For UnifaSVD, the
file value type is FP64 in `double` mode and FP32 in the other modes before any
storage conversion; the randomized baseline reads FP32 values.

See [dataset/README.md](dataset/README.md) for the checked-in input shapes.

## Validation and benchmarking

```bash
make check CUDA_HOME=/usr/local/cuda-12.8
```

This compiles the executables and runs host matrix-reader checks and command-line
regressions. These tests need the CUDA toolkit and shared libraries, but do not
need a GPU. They cover help/exit status, invalid options and numbers, malformed
LIBSVM input, and file-shape mismatches. They do not validate GPU numerical
accuracy or performance.

For numerical comparisons, run the examples on a GPU and record the commit,
GPU model, driver/toolkit versions, build flags, dimensions, rank, spectrum,
precision, vector mode, stopping criteria, and verification settings. Report
factorization time separately from matrix generation, conversion, and verification.
The full-SVD baseline allocates full singular-vector matrices, so its memory and
work differ from a truncated low-rank run.

## Repository layout

| Path | Purpose |
| --- | --- |
| `UnifaSVD.cu` | UnifaSVD driver, adaptive QB reduction, and SVD routes. |
| `cusolverDnXgesvd.cu` | Full-SVD comparison driver. |
| `cusolverDnXgesvdr.cu` | Randomized-SVD comparison driver. |
| `src/` | QR/QB kernels, CUDA helpers, validation, and matrix readers. |
| `dataset/` | Bundled LIBSVM-format inputs. |
| `tests/` | Host input and command-line regression checks. |
| `Makefile` | Build, dependency tracking, `check`, and `clean` targets. |

This repository preserves the research implementation and its bundled local
dependencies. Use Git history for revision dates and source changes.
