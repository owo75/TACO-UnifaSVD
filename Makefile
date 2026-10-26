# CUDA root can be overridden from the environment.
CUDA_HOME ?= /usr/local/cuda
CUDA_LIBDIR ?= $(CUDA_HOME)/lib64

.DELETE_ON_ERROR:

NVCC ?= $(CUDA_HOME)/bin/nvcc
PYTHON ?= python3
.PHONY: all clean check

CPP_FLAGS ?= -std=c++17
COMMON_FLAGS ?= -O2
CPPFLAGS += -I$(CUDA_HOME)/include -Isrc
ARCH_FLAGS ?= \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90
WARN_FLAGS ?= -Wno-deprecated-gpu-targets
LAMBDA_FLAG ?= --extended-lambda

NVCC_FLAGS = $(CPP_FLAGS) $(COMMON_FLAGS) $(CPPFLAGS) $(ARCH_FLAGS) $(WARN_FLAGS) $(LAMBDA_FLAG) $(NVCCFLAGS)
LDFLAGS += -L$(CUDA_LIBDIR)
LDLIBS += -lcublas -lcublasLt -lcusolver -lcurand -lcudart

TARGETS = UnifaSVD cusolverDnXgesvd cusolverDnXgesvdr
OBJECTS = $(addsuffix .o,$(TARGETS))
DEPS = $(OBJECTS:.o=.d)

all: $(TARGETS)

$(TARGETS): %: %.o
	@echo "--- Linking $@ ---"
	$(NVCC) $(ARCH_FLAGS) $(WARN_FLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cu
	@echo "--- Compiling $< ---"
	$(NVCC) $(NVCC_FLAGS) -MMD -MP -c $< -o $@

# These tests exercise help and rejected inputs without initializing CUDA.
tests/test_matrix_io: tests/test_matrix_io.o
	$(NVCC) $(ARCH_FLAGS) $(WARN_FLAGS) $^ -o $@ $(LDFLAGS)

check: all tests/test_matrix_io
	$(PYTHON) -m unittest discover -s tests -v

clean:
	$(RM) $(TARGETS) $(OBJECTS) $(DEPS) tests/test_matrix_io tests/test_matrix_io.o tests/test_matrix_io.d

-include $(DEPS)
-include tests/test_matrix_io.d
