#!/usr/bin/env python
# Copyright 2014 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Top-level control of the building/installation/cleaning of various targets

import optparse
import re
import os
import os.path
import subprocess
import pycuda.driver as drv
import pycuda.autoinit
from kernel_specs import get_ptx_file, kernels, sass_dir, ptx_dir, pre_dir, cubin_dir, dump_dir

p = optparse.OptionParser()
p.add_option("-k", "--kernels", action="store_true", dest="kernels", default=True,
             help="build or update all kernels (default)")
p.add_option("-c", "--clean", action="store_true", dest="clean",
             help="delete all generated files")
p.add_option("-p", "--preprocess", action="store_true", dest="preprocess",
             help="preprocess sass files only (for devel and debug)")
p.add_option("-d", "--dump", action="store_true", dest="dump",
             help="disassemble cubin files only (for devel and debug)")
p.add_option("-w", "--warn", action="store_true", dest="warn",
             help="enable warnings (for devel and debug)")
p.add_option("-j", "--max_concurrent", type="int", default=10,
             help="Concurrently launch a maximum of this many processes.")
p.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
             help="print output of nvcc calls.")
opts, args = p.parse_args()

if opts.preprocess or opts.dump:
    opts.kernels = False

attributes = drv.Context.get_device().get_attributes()
major = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MAJOR]
minor = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MINOR]

if major == 5 and minor == 3:
    arch = "sm_53"
else:
    arch = "sm_50"

ptxas_opts   = ["ptxas", "-arch " + arch ]
maxas_opts_i = ["maxas.pl", "-i"]
maxas_opts_p = ["maxas.pl", "-p"]
dump_opts    = ["nvdisasm", "-raw"]

if not opts.warn:
    maxas_opts_i.append("-w")

include_re = re.compile(r'^<INCLUDE\s+file="([^"]+)"\s*/>')
kernel_re = re.compile(r'\nKernel: (\S+),')


def extract_includes(name, includes=None):
    if not includes:
        includes = list()
    sass_file = os.path.join(sass_dir, name)
    includes.append(sass_file)
    for line in open(sass_file, "r"):
        match = include_re.search(line)
        if match:
            extract_includes(match.group(1), includes)
    return includes

for d in (ptx_dir, cubin_dir, pre_dir, dump_dir):
    if not os.path.exists(d):
        os.mkdir(d)

compile_cubins = []
build_cubins   = []
build_pre      = []
dump_cubins    = []

for kernel_name, kernel_spec in kernels.items():

    sass_name  = kernel_spec["sass"] + ".sass"
    cubin_name = kernel_name + ".cubin"
    pre_name   = kernel_name + "_pre.sass"
    dump_name  = kernel_name + "_dump.sass"

    ptx_file   = get_ptx_file(kernel_name, arch)
    sass_file  = os.path.join(sass_dir, sass_name)
    pre_file   = os.path.join(pre_dir, pre_name)
    cubin_file = os.path.join(cubin_dir, cubin_name)
    dump_file  = os.path.join(dump_dir, dump_name)

    maxas_i    = maxas_opts_i + ["-k " + kernel_name]
    maxas_p    = maxas_opts_p + []

    if "args" in kernel_spec:
        for pair in kernel_spec["args"].items():
            maxas_i.append("-D%s %s" % pair)
            maxas_p.append("-D%s %s" % pair)

    if opts.clean:
        for f in (ptx_file, cubin_file, pre_file, dump_file):
            if os.path.exists(f):
                os.remove(f)
        continue

    if not os.path.exists(sass_file):
        print "Missing sass file: %s for kernel: %s" % (sass_file, kernel_name)
        continue

    ptx_age   = os.path.getmtime(ptx_file)
    pre_age   = os.path.getmtime(pre_file) if os.path.exists(pre_file) else 0
    cubin_age = os.path.getmtime(cubin_file) if os.path.exists(cubin_file) else 0
    dump_age  = os.path.getmtime(dump_file) if os.path.exists(dump_file) else 0

    if opts.kernels and ptx_age > cubin_age:
        compile_cubins.append(ptxas_opts + ["-o %s" % cubin_file, ptx_file])
        cubin_age = 0

    if opts.dump and cubin_age > dump_age:
        dump_cubins.append(dump_opts + [cubin_file, ">", dump_file])

    if opts.kernels or opts.preprocess:
        for include in extract_includes(sass_name):
            include_age = os.path.getmtime(include)
            if opts.preprocess:
                if include_age > pre_age:
                    build_pre.append(maxas_p + [sass_file, pre_file])
                    break
            elif opts.kernels:
                if include_age > cubin_age:
                    build_cubins.append(maxas_i + [sass_file, cubin_file])
                    break


def run_commands(commands):
    kernels_made = []
    while len(commands) > 0:
        procs = []
        for cmdlist in commands[0:opts.max_concurrent]:
            cmdline = " ".join(cmdlist)
            proc = subprocess.Popen(cmdline,
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            procs.append((proc, cmdline))

        commands[0:opts.max_concurrent] = ()
        for proc, cmdline in procs:
            code = proc.wait()
            if opts.verbose:
                print cmdline
            if code:
                print proc.stderr.read()
            output = proc.stdout.read()
            match = kernel_re.search(output)
            if match:
                kernels_made.append(match.group(1))
            if output and opts.verbose:
                print output

    if len(kernels_made) > 0 and not opts.verbose:
        print "%d kernels compiled." % len(kernels_made)

run_commands(compile_cubins)
run_commands(build_cubins)
run_commands(build_pre)
run_commands(dump_cubins)
