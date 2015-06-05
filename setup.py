#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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

import os
from setuptools import setup, find_packages, Command
import subprocess

# Define version information
VERSION = '0.8.1'
FULLVERSION = VERSION
write_version = True

try:
    pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                            stdout=subprocess.PIPE)
    (so, serr) = pipe.communicate()
    if pipe.returncode == 0:
        FULLVERSION += "+%s" % so.strip().decode("utf-8")
except:
    pass

if write_version:
    txt = "# " + ("-" * 77) + "\n"
    txt += "# Copyright 2014 Nervana Systems Inc.\n"
    txt += "# Licensed under the Apache License, Version 2.0 "
    txt += "(the \"License\");\n"
    txt += "# you may not use this file except in compliance with the "
    txt += "License.\n"
    txt += "# You may obtain a copy of the License at\n"
    txt += "#\n"
    txt += "#      http://www.apache.org/licenses/LICENSE-2.0\n"
    txt += "#\n"
    txt += "# Unless required by applicable law or agreed to in writing, "
    txt += "software\n"
    txt += "# distributed under the License is distributed on an \"AS IS\" "
    txt += "BASIS,\n"
    txt += "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or "
    txt += "implied.\n"
    txt += "# See the License for the specific language governing permissions "
    txt += "and\n"
    txt += "# limitations under the License.\n"
    txt += "# " + ("-" * 77) + "\n"
    txt += "\"\"\"\n%s\n\"\"\"\nVERSION = '%s'\nSHORT_VERSION = '%s'\n"
    fname = os.path.join(os.path.dirname(__file__), 'neon', 'version.py')
    a = open(fname, 'w')
    try:
        a.write(txt % ("Project version information.", FULLVERSION, VERSION))
    finally:
        a.close()

# Define dependencies
dependency_links = []
required_packages = ['numpy>=1.8.1', 'PyYAML>=3.11']


class NeonCommand(Command):
    description = "Passes additional build type options to subsequent commands"
    user_options = [('cpu=', None, 'Add CPU backend related dependencies'),
                    ('gpu=', None, 'Add GPU backend related dependencies'),
                    ('dist=', None, 'Add distributed related dependencies'),
                    ('dev=', None, 'Add development related dependencies')]

    def initialize_options(self):
        self.cpu = "0"
        self.gpu = "0"
        self.dist = "0"
        self.dev = "0"

    def run(self):
        if self.dev == "1":
            self.distribution.install_requires += ['nose>=1.3.0',
                                                   'flake8>=2.2.2',
                                                   'pep8-naming>=0.2.2',
                                                   'Pillow>=2.5.0',
                                                   'sphinx>=1.2.2',
                                                   'sphinxcontrib-napoleon' +
                                                   '>=0.2.8',
                                                   'scikit-learn>=0.15.2',
                                                   'matplotlib>=1.4.0',
                                                   'imgworker>=0.2.5']
            self.distribution.dependency_links += ['git+https://github.com/'
                                                   'NervanaSystems/'
                                                   'imgworker.git#'
                                                   'egg=imgworker']
        if self.gpu == "1" or self.gpu == "cudanet":
            self.distribution.install_requires += ['cudanet>=0.2.7',
                                                   'pycuda>=2014.1']
            self.distribution.dependency_links += ['git+https://github.com/'
                                                   'NervanaSystems/'
                                                   'cuda-convnet2.git#'
                                                   'egg=cudanet']
        if self.gpu == "nervanagpu":
            self.distribution.install_requires += ['nervanagpu>=0.3.2']
            self.distribution.dependency_links += ['git+https://github.com/'
                                                   'NervanaSystems/'
                                                   'nervanagpu.git#'
                                                   'egg=nervanagpu']
        if self.dist == "1":
            self.distribution.install_requires += ['mpi4py>=1.3.1']

    def finalize_options(self):
        pass

setup(name='neon',
      version=VERSION,
      description='Deep learning framework with configurable backends',
      long_description=open('README.md').read(),
      author='Nervana Systems',
      author_email='info@nervanasys.com',
      url='http://www.nervanasys.com',
      license='License :: OSI Approved :: Apache Software License',
      scripts=['bin/neon', 'bin/hyperopt'],
      packages=find_packages(),
      install_requires=required_packages,
      cmdclass={'neon': NeonCommand},
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Environment :: Console :: Curses',
                   'Environment :: Web Environment',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: Apache Software License',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: ' +
                   'Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: System :: Distributed Computing'])
