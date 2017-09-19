# ----------------------------------------------------------------------------
# Copyright 2014-2017 Nervana Systems Inc.  All rights reserved.
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
# Top-level control of the building/installation/cleaning of various targets
#
# set empty to prevent any implicit rules from firing.
.SUFFIXES:

# Choose default Python version; overrideable with "make python2" or "make python3".

PY := $(shell python --version 2>&1  | cut -c8)
VIRTUALENV_DIR_BASE := .venv
STYLEVIRTUALENV_DIR_BASE := .styleenv

# get release version info
RELEASE := $(strip $(shell grep '^VERSION *=' setup.py | cut -f 2 -d '=' \
	                         | tr -d "\'"))

# basic check to see if any CUDA compatible GPU is installed
# set this to false to turn off GPU related functionality
HAS_GPU := $(shell nvcc neon/backends/util/check_gpu.c > /dev/null 2>&1 && ./a.out && rm a.out && echo true)

ifdef HAS_GPU
# Get CUDA_ROOT for LD_RUN_PATH
export CUDA_ROOT:=$(abspath $(shell which nvcc)/../..)
else
# Try to find CUDA.  Kernels will still need nvcc in path
export CUDA_ROOT:=$(firstword $(wildcard $(addprefix /usr/local/, cuda-8.0 cuda-7.5 cuda-7.0 cuda)))

ifdef CUDA_ROOT
export PATH:=$(CUDA_ROOT)/bin:$(PATH)
HAS_GPU := $(shell $(CUDA_ROOT)/bin/nvcc neon/backends/util/check_gpu.c > /dev/null 2>&1 && ./a.out && rm a.out && echo true)
endif
endif
ifdef CUDA_ROOT
# Compiling with LD_RUN_PATH eliminates the need for LD_LIBRARY_PATH
# when running
export LD_RUN_PATH:=$(CUDA_ROOT)/lib64
endif

# set this to true to install visualization dependencies and functionality
# (off by default)
VIS :=

# style checking related
STYLE_CHECK_OPTS :=
STYLE_CHECK_DIRS := neon bin/* tests examples benchmark

# pytest options
TEST_OPTS :=
TEST_DIRS := tests/
# turn off GPU tests if no GPU present
# TODO: refactor neon/backends/tests to run under CPU
ifneq ($(HAS_GPU), true)
	TEST_DIRS := -k "cpu or mkl" -m "not hasgpu" $(TEST_DIRS)
else
	TEST_DIRS := -m "hasgpu or not hasgpu and not mkl_only" $(TEST_DIRS)
endif

# this variable controls where we publish Sphinx docs to
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

#neon mklEngine object
MKL_ENGINE  := neon/backends/mklEngine

VIRTUALENV := $(shell command -v virtualenv 2> /dev/null)
ifndef VIRTUALENV
    $(error "virtualenv must be installed. Consider `pip install virtualenv`.")
endif

ifeq ($(PY), 2)
	VIRTUALENV_EXE := virtualenv -p python2.7
	PYLINT3K_ARGS := --disable=no-absolute-import
	VIRTUALENV_DIR = $(VIRTUALENV_DIR_BASE)$(PY)
	ACTIVATE = $(VIRTUALENV_DIR)/bin/activate
else
	VIRTUALENV_EXE := python3 -m venv
	PYLINT3K_ARGS :=
	VIRTUALENV_DIR = $(VIRTUALENV_DIR_BASE)3
	ACTIVATE = $(VIRTUALENV_DIR)/bin/activate
endif

.PHONY: default all env sysinstall sysinstall_nodeps neon_install python2 python3 \
	    sysdeps sysuninstall clean_py clean_so \
	    clean test coverage style lint lint3k check doc html release examples \
	    serialize_check $(MKL_ENGINE)

default: env

all:
	$(MAKE) PY=3 TEST_OPTS='$(TEST_OPTS)' test
	$(MAKE) PY=2 TEST_OPTS='$(TEST_OPTS)' test

env: $(MKL_ENGINE) $(ACTIVATE)

python2: VIRTUALENV_EXE := virtualenv -p python2.7
python2: VIRTUALENV_DIR := $(VIRTUALENV_DIR_BASE)2
python2: ACTIVATE := $(VIRTUALENV_DIR)/bin/activate
python2: PYLINT3K_ARGS := --disable=no-absolute-import
python2: env

python3: VIRTUALENV_EXE := python3 -m venv
python3: VIRTUALENV_DIR := $(VIRTUALENV_DIR_BASE)3
python3: ACTIVATE := $(VIRTUALENV_DIR)/bin/activate
python3: PYLINT3K_ARGS :=
python3: env

$(ACTIVATE): requirements.txt gpu_requirements.txt vis_requirements.txt
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip install -U pip
	@# cython added separately due to h5py dependency ordering bug.  See:
	@# https://github.com/h5py/h5py/issues/535
	@. $(ACTIVATE); pip install cython==0.23.1
	@. $(ACTIVATE); pip install -r requirements.txt
	@. $(ACTIVATE); $(MAKE) aeon_install
ifeq ($(VIS), true)
	@echo "Updating visualization related dependecies in $(VIRTUALENV_DIR)..."
	@. $(ACTIVATE); pip install -r vis_requirements.txt
endif
	@echo
ifeq ($(HAS_GPU), true)
	@echo "Updating GPU dependencies in $(VIRTUALENV_DIR)..."
	@. $(ACTIVATE); pip install -r gpu_requirements.txt
	@echo
endif
	@echo "Installing neon in development mode..."
	@. $(ACTIVATE); python setup.py develop
	@rm -f $(VIRTUALENV_DIR_BASE); ln -s $(VIRTUALENV_DIR) $(VIRTUALENV_DIR_BASE)
	@echo "###########################################################"
	@echo "Setup complete.  Type:"
	@echo "    . '$(ACTIVATE)'"
	@echo "to work interactively ($(VIRTUALENV_DIR) also symlinked to $(VIRTUALENV_DIR_BASE))"
	@echo "###########################################################"
	@touch $(ACTIVATE)
	@echo
$(MKL_ENGINE):
	@echo "Building MKL Engine..."
	@./install_mkl.sh intel

# TODO: handle kernel/.so compilation via setup.py directly
sysinstall_nodeps: neon_install
sysinstall: $(MKL_ENGINE) sysdeps neon_install
neon_install:
	@echo "Installing neon system wide..."
	@python setup.py install
	@echo

aeon_install:
	@echo "Attempting to install optional aeon dataloader..."
	-@git clone https://github.com/NervanaSystems/aeon.git aeon ; cd aeon && git fetch && git checkout tags/v1.0.0 && mkdir -p build && cd build && rm -rf * && cmake .. && pip install .

sysdeps:
	@echo "Installing neon dependencies system wide..."
	@# cython added separately due to h5py dependency ordering bug.  See:
	@# https://github.com/h5py/h5py/issues/535
	@pip install cython==0.23.1
	@pip install -r requirements.txt
	$(MAKE) aeon_install
ifeq ($(VIS), true)
	@pip install -r vis_requirements.txt
endif
ifeq ($(HAS_GPU), true)
	@pip install -r gpu_requirements.txt
endif

sysuninstall:
	@echo "Uninstalling neon system wide..."
	@pip uninstall neon
	@echo

clean_py:
	@echo "Cleaning compiled python object files..."
	@find . -name "*.py[co]" -type f -delete
	@echo

clean_so:
	@echo "Cleaning compiled shared object files..."
	@cd $(MKL_ENGINE) && $(MAKE) clean
	@echo

clean: clean_py clean_so
	@echo "Removing virtual environment files..."
	@rm -rf aeon
	@rm -rf $(VIRTUALENV_DIR_BASE) $(VIRTUALENV_DIR_BASE)2 $(VIRTUALENV_DIR_BASE)3 $(STYLEVIRTUALENV_DIR_BASE)
	@echo

test: env
	@echo "Running unit tests..."
	@. $(ACTIVATE); py.test --cov=neon $(TEST_OPTS) $(TEST_DIRS); coverage xml
	@echo

systest:
	@echo "Running unit tests..."
	py.test $(TEST_OPTS) $(TEST_DIRS)
	@echo

benchmarks: env
	@echo "Running all benchmarks..."
	@. $(ACTIVATE); tests/run_benchmarks.py
	@echo

serialize_check: env
	@echo "Running CPU backend test of model serialization"
	@. $(ACTIVATE); python tests/serialization_check.py -e 10 -b cpu
	@echo

coverage: env
	@. $(ACTIVATE); py.test --cov=neon tests/
	@echo

style: env
	@. $(ACTIVATE); flake8 $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)
	@. $(ACTIVATE); pylint --reports=n --py3k $(PYLINT3K_ARGS) --ignore=.venv *
	@echo

# doesn't install everything, just runs flake8 and pylint on the code
# using a style environment
fast_style:
	virtualenv .styleenv
	. .styleenv/bin/activate; pip install `grep flake8 requirements.txt`
	. .styleenv/bin/activate; pip install `grep pylint requirements.txt`
	. .styleenv/bin/activate; flake8 $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)
	. .styleenv/bin/activate; pylint --reports=n --py3k $(PYLINT3K_ARGS) --ignore=.styleenv *
	@echo

lint: env
	@. $(ACTIVATE); pylint --output-format=colorized neon
	@echo

lint3k: env
	@. $(ACTIVATE); pylint --py3k $(PYLINT3K_ARGS) --ignore=.venv *
	@echo

check: env
	@echo "Running style checks.  Number of style errors is... "
	-@. $(ACTIVATE); flake8 --count $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS) \
	                 > /dev/null
	@echo
	@echo "Number of missing docstrings is..."
	-@. $(ACTIVATE); pylint --disable=all --enable=missing-docstring -r n \
	                 neon | grep "^C" | wc -l
	@echo
	@echo "Running unit tests..."
	-@. $(ACTIVATE); py.test $(TEST_DIRS) | tail -1 | cut -f 2,3 -d ' '
	@echo

doc: env
	@. $(ACTIVATE); neon --help > doc/source/neon_help_output.txt
	$(MAKE) -C $(DOC_DIR) clean
	@. $(ACTIVATE); $(MAKE) -C $(DOC_DIR) html
	@echo "Documentation built in $(DOC_DIR)/build/html"
	@echo

html: doc
	@echo "To view documents open your browser to: http://localhost:8000"
	@cd $(DOC_DIR)/build/html; python -m SimpleHTTPServer
	@echo

publish_doc: doc
ifneq (, $(DOC_PUB_HOST))
	@echo "relpath: $(DOC_PUB_RELEASE_PATH)"
	@-cd $(DOC_DIR)/build/html && \
		rsync -avz -essh --perms --chmod=ugo+rX . \
		$(DOC_PUB_USER)@$(DOC_PUB_HOST):$(DOC_PUB_RELEASE_PATH)
	@-ssh $(DOC_PUB_USER)@$(DOC_PUB_HOST) \
		'rm -f $(DOC_PUB_PATH)/latest && \
		 ln -sf $(DOC_PUB_RELEASE_PATH) $(DOC_PUB_PATH)/latest'
else
	@echo "Can't publish.  Ensure DOC_PUB_HOST, DOC_PUB_USER, DOC_PUB_PATH set"
endif

dist: env
	@echo "Prepping distribution..."
	@python setup.py sdist

release: check dist
	@echo "Bump version number in setup.py"
	@vi setup.py
	@echo "Update ChangeLog"
	@vi ChangeLog
	@echo "TODO: commit changes"
	@echo "TODO: publish release to PYPI"
	@echo "TODO (manual script): publish documentation"
	@echo
