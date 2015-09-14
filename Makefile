# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
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

.SUFFIXES:  # set empty to prevent any implicit rules from firing.

# where our installed python packages will live
VIRTUALENV_DIR := .venv
VIRTUALENV_EXE := virtualenv  # use pyvenv for python3 install
ACTIVATE := $(VIRTUALENV_DIR)/bin/activate

# get release version info
RELEASE := $(strip $(shell grep '^VERSION *=' setup.py | cut -f 2 -d '=' \
	                         | tr -d "\'"))

# basic check to see if any CUDA compatible GPU is installed
HAS_GPU := $(shell nvcc --version > /dev/null 2>&1 && echo true)

# lazily evaluated determination of CUDA GPU capabilities
CUDA_CAPABILITY_CHECKER := neon/backends/util/cuda_capability
CUDA_COMPUTE_CAPABILITY = $(shell $(CUDA_CAPABILITY_CHECKER) > /dev/null 2>&1 \
													        || echo 0)
COMPUTE_MAJOR = $(shell echo $(CUDA_COMPUTE_CAPABILITY) | cut -f1 -d.)
MAXWELL_MAJOR := 5
HAS_MAXWELL_GPU = $(shell [ $(COMPUTE_MAJOR) -ge $(MAXWELL_MAJOR) ] && echo true)

# style checking related
STYLE_CHECK_OPTS :=
STYLE_CHECK_DIRS := neon bin tests

# pytest options
TEST_OPTS :=

# arguments to running examples
EXAMPLE_ARGS := -e1

# this variable controls where we publish Sphinx docs to
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

# Maxwell assembler project related
MAXAS_PLIB := PERL5LIB=$(VIRTUALENV_DIR)/share/perl/5.18.2
MAXAS_SRC_URL := https://github.com/NervanaSystems/maxas.git
MAXAS_DL_DIR := $(VIRTUALENV_DIR)/maxas
MAXAS := $(VIRTUALENV_DIR)/bin/maxas.pl
MAXAS_VER_FILE := $(VIRTUALENV_DIR)/maxas/lib/MaxAs/MaxAs.pm
MAXAS_INSTALLED_VERSION = $(shell test -f $(ACTIVATE) && . $(ACTIVATE) && $(MAXAS_PLIB) $(MAXAS) --version)
MAXAS_AVAIL_VERSION = $(shell test -f $(MAXAS_VER_FILE) && grep VERSION $(MAXAS_VER_FILE) | cut -f2 -d= | tr -d "'; ")

# GPU Kernel compilation related
KERNEL_BUILDER := neon/backends/make_kernels.py
KERNEL_BUILDER_BUILD_OPTS := --kernels
KERNEL_BUILDER_CLEAN_OPTS := --clean

# neon compiled objects
IMAGESET_DECODER := neon/data/imageset_decoder.so

.PHONY: default env maxas kernels sysinstall sysuninstall clean_py clean_maxas \
	      clean_util clean_so clean_kernels clean test coverage style lint check \
	      doc html release examples serialize_check

default: env

env: $(ACTIVATE) $(CUDA_CAPABILITY_CHECKER) kernels $(IMAGESET_DECODER)

$(ACTIVATE): requirements.txt gpu_requirements.txt vis_requirements.txt
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip install -U pip
	@. $(ACTIVATE); pip install -r requirements.txt
	@. $(ACTIVATE); pip install -r vis_requirements.txt
	@echo
ifeq ($(HAS_GPU), true)
	@echo "Updating GPU dependencies in $(VIRTUALENV_DIR)..."
	@. $(ACTIVATE); pip install -r gpu_requirements.txt
	@echo
endif
	@echo "Installing neon in development mode..."
	@. $(ACTIVATE); python setup.py develop
	@echo "######################"
	@echo "Setup complete.  Type:"
	@echo "    . '$(ACTIVATE)'"
	@echo "to work interactively"
	@echo "######################"
	@touch $(ACTIVATE)
	@echo

$(CUDA_CAPABILITY_CHECKER): $(CUDA_CAPABILITY_CHECKER).c
ifeq ($(HAS_GPU), true)
	@echo "Building $(CUDA_CAPABILITY_CHECKER) ..."
	nvcc -l cuda -o $(CUDA_CAPABILITY_CHECKER) $(CUDA_CAPABILITY_CHECKER).c
	@echo
endif

maxas: $(ACTIVATE) $(MAXAS_DL_DIR)
ifeq ($(HAS_MAXWELL_GPU), true)
	@cd $(MAXAS_DL_DIR) && git pull >/dev/null 2>&1
	@test -f $(MAXAS) ||\
		{ echo "Installing maxas..." &&\
		  cd $(MAXAS_DL_DIR) &&\
		  perl Makefile.PL PREFIX=.. &&\
		  make install ;\
		  if [ $$? != 0 ] ; then \
			  echo "Problems installing maxas"; exit 1 ;\
		  fi }
  ifneq ($(MAXAS_INSTALLED_VERSION),$(MAXAS_AVAIL_VERSION))
		@echo "Updating maxas installation from $(MAXAS_INSTALLED_VERSION) to $(MAXAS_AVAIL_VERSION) ..."
		cd $(MAXAS_DL_DIR) &&\
		perl Makefile.PL PREFIX=.. &&\
		make install ;\
		  if [ $$? != 0 ] ; then \
			  echo "Problems updating maxas"; exit 1 ;\
		  fi
  endif
endif

$(MAXAS_DL_DIR):
ifeq ($(HAS_MAXWELL_GPU), true)
	@test -d $(MAXAS_DL_DIR) ||\
		{ echo "Cloning maxas repo..." ;\
		  git clone $(MAXAS_SRC_URL) $(MAXAS_DL_DIR) ;\
		  echo "";\
		}
endif

kernels: $(ACTIVATE) maxas
ifeq ($(HAS_MAXWELL_GPU), true)
	@. $(ACTIVATE); $(MAXAS_PLIB) $(KERNEL_BUILDER) $(KERNEL_BUILDER_BUILD_OPTS)
	@echo
endif

$(IMAGESET_DECODER): $(subst so,cpp,$(IMAGESET_DECODER))
ifeq ($(shell pkg-config --modversion opencv >/dev/null 2>&1; echo $$?), 0)
	@echo "Compiling $(IMAGESET_DECODER) ..."
  ifeq ($(shell uname -s), Darwin)
		-g++ -w -O3 -stdlib=libc++ -shared -o $(IMAGESET_DECODER) -std=c++11 -fPIC $< $$(pkg-config opencv --cflags --libs)
  else
		-g++ -w -O3 -shared -o $(IMAGESET_DECODER) -std=c++11 -fPIC $< $$(pkg-config opencv --cflags --libs)
  endif
else
	@echo "pkg-config or opencv not installed.  Unable to build imageset_decoder"
	@echo
endif

# TODO: remove env dep and handle kernel/.so compilation via setup.py directly
sysinstall: env
	@echo "Installing neon system wide..."
	@pip install -U pip
	@pip install -r requirements.txt
	@pip install -r vis_requirements.txt
ifeq ($(HAS_GPU), true)
	@pip install -r gpu_requirements.txt
endif
	@pip install .
	@echo

sysuninstall:
	@echo "Uninstalling neon system wide..."
	@pip uninstall neon
	@echo

clean_py:
	@echo "Cleaning compiled python object files..."
	@find . -name "*.py[co]" -type f -delete
	@echo

clean_util:
	@echo "Cleaning compiled utilities..."
	@rm -f $(CUDA_CAPABILITY_CHECKER)
	@echo

clean_so:
	@echo "Cleaning compiled shared object files..."
	@rm -f $(IMAGESET_DECODER)
	@echo

clean_maxas:
ifeq ($(HAS_MAXWELL_GPU), true)
	@echo "Cleaning maxas installation and repo files..."
	@rm -rf $(MAXAS_DL_DIR)
	@echo
endif

clean_kernels:
ifeq ($(HAS_MAXWELL_GPU), true)
	@echo "Cleaning compiled gpu kernel files..."
	@test -f $(ACTIVATE) && . $(ACTIVATE); $(KERNEL_BUILDER) $(KERNEL_BUILDER_CLEAN_OPTS)
	@echo
endif

clean: clean_py clean_util clean_so clean_maxas clean_kernels
	@echo "Removing virtual environment files..."
	@rm -rf $(VIRTUALENV_DIR)
	@echo

test: env
	@echo "Running unit tests..."
	@. $(ACTIVATE); py.test $(TEST_OPTS) tests/ neon/backends/tests/
	@echo

examples: env
	@echo "Running all examples..."
	@. $(ACTIVATE); \
		for fn in `ls -1 examples/*.py`; \
		do \
		    echo "Running $$fn $(EXAMPLE_ARGS)"; \
		    python $$fn $(EXAMPLE_ARGS); \
			if [ $$? -ne 0 ]; \
	        then \
	            exit 1; \
			fi; \
		done;
	@echo

serialize_check: env
	@echo "Running CPU backend test of model serialization"
	@. $(ACTIVATE); python tests/serialization_check.py -e 10 -b cpu
	@echo

coverage: env
	@. $(ACTIVATE); py.test --cov=neon tests/ neon/backends/tests/
	@echo

style: env
	@. $(ACTIVATE); flake8 $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)
	@echo

lint: env
	@. $(ACTIVATE); pylint --output-format=colorized neon
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
	-@. $(ACTIVATE); py.test tests/ | tail -1 | cut -f 2,3 -d ' '
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
