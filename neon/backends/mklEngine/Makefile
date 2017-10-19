# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

SRCS			:=$(wildcard  src/*.h)
MKL_ENGINE_SO   := mklEngine.so
MATH_CPU_SO		:= cmath.so
UNAME_S         := $(shell uname -s)
MKL_PATH        := $(MKLROOT)
COMMON_FLAGS    := -std=c99 -O3 -I$(MKL_PATH)/include -L$(MKL_PATH)/lib -Wl,-rpath=$(MKL_PATH)/lib

ifeq   ($(CC), icc)
	CFLAGS  := -qopenmp -lmklml_intel -fPIC -march=native
	#CFLAGS := -qopenmp -lmklml_intel -fPIC -g -liomp5 -xMIC-AVX512
else
	CFLAGS := -fopenmp -lmklml_gnu -fPIC -march=native -g -liomp5
endif

ifeq   ($(WIN_BUILD), 1)
	CFLAGS := -fopenmp -lmklml -fPIC -march=native -g -liomp5md
endif

.PHONY: clean

all:$(MKL_ENGINE_SO) $(MATH_CPU_SO)

$(MKL_ENGINE_SO): src/conv.c src/pooling.c src/relu.c src/batchNorm.c src/concat.c src/softmax.c $(SRCS)
	@echo "Building $@..."
	$(CC) $^ -shared -o $@ $(COMMON_FLAGS) $(CFLAGS)

$(MATH_CPU_SO): src/math_cpu.c
	@echo "Building $@..."
	$(CC) $^ -shared -o $@ $(COMMON_FLAGS) $(CFLAGS)

clean:
	@rm -vf *.o *.so *.dll

