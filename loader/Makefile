# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
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

SRCS            := $(wildcard src/*.hpp)
CFLAGS          := -Wall -Wno-deprecated-declarations -O3 -std=c++11
CC              := g++
THREAD_TEST     := bin/thread_test
DECODE_TEST     := bin/decode_test
LOADER_SO       := bin/loader.so
UNAME_S         := $(shell uname -s)
ifeq ($(UNAME_S), FreeBSD)
	CC          := clang++
endif

PKGCONFIG := $(shell command -v pkg-config 2> /dev/null)
ifndef PKGCONFIG
    $(error "pkg-config must be installed. Consider `sudo apt-get install pkg-config` or `brew install pkg-config`.")
endif

ifeq ($(shell pkg-config --exists opencv; echo $$?), 0)
	IMGFLAG     := -DHAS_IMGLIB
	CFLAGS      := $(shell pkg-config --cflags opencv) $(CFLAGS)
	LDFLAGS     := $(shell pkg-config --libs opencv)
ifeq ($(shell pkg-config --exists libavutil libavformat libavcodec libswscale; echo $$?), 0)
	VIDFLAG     := -DHAS_VIDLIB
	AUDFLAG     := -DHAS_AUDLIB
	CFLAGS      := $(shell pkg-config --cflags libavutil libavformat libavcodec libswscale) $(CFLAGS)
	LDFLAGS     := $(shell pkg-config --libs libavutil libavformat libavcodec libswscale) $(LDFLAGS)
endif
endif

MEDIAFLAGS      := $(IMGFLAG) $(VIDFLAG) $(AUDFLAG)

ifeq ($(HAS_GPU), true)
ifeq ($(CUDA_ROOT),)
	CUDA_ROOT   := $(abspath $(shell which nvcc)/../..)
endif

	GPUFLAG     := -DHAS_GPU
	CFLAGS      := -I$(CUDA_ROOT)/include $(CFLAGS)
	ifeq ($(shell uname -s),Darwin)
		LDFLAGS := -L$(CUDA_ROOT)/lib $(LDFLAGS)
	else
		LDFLAGS := -L$(CUDA_ROOT)/lib64 $(LDFLAGS)
	endif
	LDFLAGS     := -lcuda -lcudart $(LDFLAGS)
endif

$(shell mkdir -p bin)

.PHONY: clean

all: $(THREAD_TEST) $(DECODE_TEST) $(LOADER_SO)

$(THREAD_TEST): test/thread_test.cpp src/loader.cpp $(SRCS)
	@echo "Building $@..."
	$(CC) -o $@ $(CFLAGS) -g -O0 $(GPUFLAG) $(MEDIAFLAGS) -Isrc $< $(LDFLAGS)

$(DECODE_TEST): test/decode_test.cpp $(SRCS)
	@echo "Building $@..."
	$(CC) -o $@ $(CFLAGS) -g -O0 $(MEDIAFLAGS) -Isrc $< $(LDFLAGS)

$(LOADER_SO): src/loader.cpp $(SRCS)
	@echo "Building $@..."
	$(CC) -shared -o $@ -fPIC $(CFLAGS) $(GPUFLAG) $(MEDIAFLAGS) $< $(LDFLAGS)

clean:
	@rm -vf *.o $(THREAD_TEST) $(DECODE_TEST) $(LOADER_SO)
