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

SRCS            := $(wildcard *.hpp)
CFLAGS          := -Wall -O3 -std=c++11
CFLAGSDBG       := -g -Wall -O0 -std=c++11
INC             := $(shell pkg-config --cflags opencv)

ifeq ($(shell pkg-config --exists opencv; echo $$?), 0)
    IMGLDIR     := $(shell pkg-config --libs-only-L opencv)
    IMGLIBS     := $(shell pkg-config --libs-only-l opencv)
endif

ifeq ($(IMGLDIR),)
	IMGLDIR     := -L$(shell dirname `pkg-config opencv --libs-only-other | cut -d' ' -f1`)
endif

ifeq ($(shell pkg-config --exists libavutil libavformat libavcodec libswscale; echo $$?), 0)
    VIDFLAG     := -DHAS_VIDLIB
    VIDLDIR     := $(shell pkg-config --libs-only-L libavutil libavformat libavcodec libswscale)
    VIDLIBS     := -lavutil -lavformat -lavcodec -lswscale
endif

MEDIAFLAGS      := $(VIDFLAG) $(AUDFLAG)

LDIR            := $(IMGLDIR) $(VIDLDIR)
LIBS            := $(IMGLIBS) $(VIDLIBS)

CC              := g++
LOADER          := bin/loader
LOADERSO        := bin/loader.so
UNAME_S         := $(shell uname -s)
ifeq ($(UNAME_S), FreeBSD)
	CC          := clang++
endif
OPENCVPKG       := $(shell pkg-config --list-all | grep -e '^opencv' | cut -d' ' -f1)

ifeq ($(CUDA_ROOT),)
    CUDA_ROOT   := $(patsubst %/bin/nvcc,%, $(realpath $(shell which ncc)))
endif

ifeq ($(HAS_GPU), true)
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

all: $(LOADER) $(LOADERSO)

$(LOADER): src/loader.cpp $(SRCS)
ifeq (${OPENCVPKG},)
	$(error opencv not installed.  Unable to build $@)
endif
	@echo "Building $@..."
	$(CC) -o $@ -DSTANDALONE $(CFLAGSDBG) $(GPUFLAG) $(MEDIAFLAGS) $< $(INC) $(LDIR) $(LIBS)

test/decode_test: test/decode_test.cpp $(SRCS)
ifeq (${OPENCVPKG},)
	$(error opencv not installed.  Unable to build $@)
endif
	@echo "Building $@..."
	$(CC) -o $@ -DSTANDALONE $(CFLAGSDBG) $(MEDIAFLAGS) $< $(INC) -Isrc $(LDIR) $(LIBS)

$(LOADERSO): src/loader.cpp $(SRCS)
ifeq (${OPENCVPKG},)
	$(error opencv not installed.  Unable to build $@)
endif
	@echo "Building $@..."
	$(CC) -shared -o $@ -fPIC $(CFLAGS) $(GPUFLAG) $(MEDIAFLAGS) $< $(INC) $(LDIR) $(LIBS)

clean:
	@rm -vf *.o $(LOADER) $(LOADERSO)
