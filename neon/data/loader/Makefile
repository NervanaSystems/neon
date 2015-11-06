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

SRCS = $(wildcard *.hpp)

.PHONY: clean

all: loader loader.so
INC = -I/usr/local/cuda/include
ifeq ($(CC), nvcc)
LIB = -L/usr/local/cuda/lib64
endif
CC = nvcc

loader: loader.cpp $(SRCS)
	@echo "Building $@..."
ifneq ($(shell pkg-config --modversion opencv >/dev/null 2>&1; echo $$?), 0)
	$(error pkg-config or opencv not installed.  Unable to build $@)
endif
ifeq ($(CC), nvcc)
	$(CC) -o $@ -DSTANDALONE -DHASGPU $(INC) $(LIB) -O0 -g -std=c++11 -lcudart -lcuda $< $$(pkg-config opencv --cflags --libs)
else
	$(CC) -o $@ -DSTANDALONE $(INC) $(LIB) -Wall -O0 -g -std=c++11 -pthread $< $$(pkg-config opencv --cflags --libs)
endif

loader.so: loader.cpp $(SRCS)
	@echo "Building $@..."
ifneq ($(shell pkg-config --modversion opencv >/dev/null 2>&1; echo $$?), 0)
	$(error pkg-config or opencv not installed.  Unable to build $@)
endif
ifeq ($(CC), nvcc)
	$(CC) -shared -o $@ -Xcompiler -fpic -DHASGPU $(INC) $(LIB) -O3 -std=c++11 -lcuda -lcudart $< $$(pkg-config opencv --cflags --libs)
else
	$(CC) -shared -o $@ -fPIC $(INC) $(LIB) -Wall -Werror -O0 -g -std=c++11 -pthread $< $$(pkg-config opencv --cflags --libs)
endif

clean:
	@rm -vf *.o loader loader.so
