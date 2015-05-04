.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Frequently Asked Questions
--------------------------

Installation
============

* Does neon run on Microsoft Windows?

  * At this time we are only supporting Linux and Mac OSX based installations.

* During install on a Mac I get "yaml.h" file not found error when the PyYAML
  package is being built.  Is that bad?

  * It can safely be ignored in this situation.  The problem is that you don't
    have libyaml installed so PyYAML will resort to its own (slightly slower)
    implementation. Without it, neon will still be able to successfully parse
    and read your Experiment's YAML files.

* When trying to install with GPU=cudanet or GPU=nervanagpu I get an
  "nvcc: Command not found" error.  Why?

  * First ensure that you actually have available a CUDA compliant graphics
    card and that the CUDA drivers and SDK are installed correctly.  It can be
    downloaded from: https://developer.nvidia.com/cuda-downloads
  * Second, ensure that your PATH and/or LD_LIBRARY_PATH are updated to find
    the CUDA installation, as described in:
    http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#post-installation-actions
  * if you are running on the above command as the super user via ``sudo`` on
    Ubuntu Linux, you may need to pass PATH and LD_LIBRARY_PATH from your
    environment like so:
    ``sudo env “PATH=$PATH” env “LD_LIBRARY_PATH=$LD_LIBRARY_PATH” make install``


Running
=======

* The console output is too verbose, how do I reduce the amount of logging?

  * In the YAML file for your experiment, you can reduce the amount of logging
    by increasing the numeric value of ``logging``'s ``level`` parameter.  A
    value of 40 implies that only messages of type ERROR and CRITICAL will be
    displayed for instance.

* The console output is too sparse, how do I increase the amount of logging?

  * In the YAML file for your experiment, you can increase the amount of logging
    by decreasing the numeric value of ``logging``'s ``level`` parameter.  A
    value of 10 implies that messages of type DEBUG, INFO, WARNING, ERROR, and
    CRITICAL will all be displayed for instance.

Contributing
============

* I think I found a bug, what do I do?

  * Please search our
    `Github issues <https://github.com/NervanaSystems/neon/issues>`_ list and 
    if it hasn't already been addressed, file a new issue so we can take a
    look.
