# CUDA root can be overridden from the environment.
CUDA_HOME ?= /usr/local/cuda
CUDA_LIBDIR ?= $(CUDA_HOME)/lib64

.DELETE_ON_ERROR:

NVCC ?= $(CUDA_HOME)/bin/nvcc

CPP_FLAGS = -std=c++17
COMMON_FLAGS = -O2
INCLUDE_PATHS = -I$(CUDA_HOME)/include -Isrc
ARCH_FLAGS = \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90
WARN_FLAGS = -Wno-deprecated-gpu-targets
LAMBDA_FLAG = --extended-lambda

NVCC_FLAGS = $(CPP_FLAGS) $(COMMON_FLAGS) $(INCLUDE_PATHS) $(ARCH_FLAGS) $(WARN_FLAGS) $(LAMBDA_FLAG)
LDFLAGS = -L$(CUDA_LIBDIR) -lcublas -lcublasLt -lcusolver -lcurand -lcudart -lstdc++

TARGETS = UnifaSVD cusolverDnXgesvd cusolverDnXgesvdr

all: $(TARGETS)

UnifaSVD: UnifaSVD.o
	@echo "--- Linking $@ ---"
	$(NVCC) $^ -o $@ $(LDFLAGS)

cusolverDnXgesvd: cusolverDnXgesvd.o
	@echo "--- Linking $@ ---"
	$(NVCC) $^ -o $@ $(LDFLAGS)

cusolverDnXgesvdr: cusolverDnXgesvdr.o
	@echo "--- Linking $@ ---"
	$(NVCC) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	@echo "--- Compiling $< ---"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o
