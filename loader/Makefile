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
CFLAGSDBG       := -g -Wall -O0 -std=c++11
CC              := g++
THREAD_TEST     := bin/thread_test
DECODE_TEST     := bin/decode_test
LOADER_SO       := bin/loader.so
UNAME_S         := $(shell uname -s)
ifeq ($(UNAME_S), FreeBSD)
	CC          := clang++
endif

ifeq ($(shell pkg-config --exists opencv; echo $$?), 0)
	IMGFLAG     := -DHAS_IMGLIB
	INC         := $(shell pkg-config --cflags opencv)
	IMGLDIR     := $(shell pkg-config --libs-only-L opencv)
	IMGLIBS     := $(shell pkg-config --libs-only-l opencv)
ifeq ($(shell pkg-config --exists libavutil libavformat libavcodec libswscale; echo $$?), 0)
	VIDFLAG     := -DHAS_VIDLIB
	AUDFLAG     := -DHAS_AUDLIB
	VIDLDIR     := $(shell pkg-config --libs-only-L libavutil libavformat libavcodec libswscale)
	VIDLIBS     := -lavutil -lavformat -lavcodec -lswscale
endif
endif

MEDIAFLAGS      := $(IMGFLAG) $(VIDFLAG) $(AUDFLAG)
LDIR            := $(IMGLDIR) $(VIDLDIR)
LIBS            := $(IMGLIBS) $(VIDLIBS)

ifeq ($(HAS_GPU), true)
ifeq ($(CUDA_ROOT),)
	CUDA_ROOT   := $(patsubst %/bin/nvcc,%, $(shell which nvcc))
endif

	GPUFLAG     := -DHAS_GPU
	INC         := -I$(CUDA_ROOT)/include $(INC)
	ifeq ($(shell uname -s),Darwin)
		LDIR    := -L$(CUDA_ROOT)/lib $(LDIR)
	else
		LDIR    := -L$(CUDA_ROOT)/lib64 $(LDIR)
	endif
	LIBS        := -lcuda -lcudart $(LIBS)
endif

$(shell mkdir -p bin)

.PHONY: clean

all: $(THREAD_TEST) $(DECODE_TEST) $(LOADER_SO)

$(THREAD_TEST): test/thread_test.cpp src/loader.cpp $(SRCS)
	@echo "Building $@..."
	$(CC) -o $@ $(CFLAGSDBG) $(GPUFLAG) $(MEDIAFLAGS) $< $(INC) -Isrc $(LDIR) $(LIBS)

$(DECODE_TEST): test/decode_test.cpp $(SRCS)
	@echo "Building $@..."
	$(CC) -o $@ $(CFLAGSDBG) $(MEDIAFLAGS) $< $(INC) -Isrc $(LDIR) $(LIBS)

$(LOADER_SO): src/loader.cpp $(SRCS)
	@echo "Building $@..."
	$(CC) -shared -o $@ -fPIC $(CFLAGS) $(GPUFLAG) $(MEDIAFLAGS) $< $(INC) $(LDIR) $(LIBS)

clean:
	@rm -vf *.o $(THREAD_TEST) $(DECODE_TEST) $(LOADER_SO)
