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

neon Contribution Process
-------------------------

1. File an issue:

   * Create an issue on github:
     https://github.com/NervanaSystems/neon/issues

2. Clone and/or update your checked out copy of neon to ensure you have the
   most recent commits from the master branch:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    git fetch origin
    git checkout master
    git pull

3. Create a new feature branch for your work and switch to it.  Give it a
   meaningful name related to the task(s) at hand:

.. code-block:: bash

    # to do both steps at once:
    git checkout -b my_new_feature_branch
    # or separately:
    git branch my_new_feature_branch
    git checkout my_new_feature_branch

4. Locally build neon, with your build type configured (eg. with GPU):

.. code-block:: bash

    # to setup your build type defaults for all future commands, edit setup.cfg
    vi setup.cfg
    make develop
    # or
    make build
    # or override for a specific command
    make -e DEV=1 DIST=1 GPU=cudanet develop

5. Ideally you'd start by creating one or more unit tests with the
   functionality you expect your new feature to perform.  These should reside
   under the appropriate tests subdirectory of whatever you are changing.
   Then hack away at the code until you feel your feature is complete.  Once
   satisfied, run the code through the tests and various style checking:

.. code-block:: bash

    make test   # ensure all are OK for each of your build types
    make sanity # again ensure all pass OK
    make style  # ensure there are no style related issues
    make speed  # ensure there are no performance regressions
    make grad   # ensure sample gradient checks all pass OK
    make lint   # (optional).  We still have a fair bit to clean up currently!

6. If necessary you may want to update and/or rebuild the documentation.
   This all exists under doc/source and is in 
   `Sphinx Restructed Text format <http://sphinx-doc.org/rest.html>`_:

.. code-block:: bash

    make doc         # builds documentation locally

7. Commit your changes and push your feature branch to ypur github fork.  Be
   sure to add a descriptive message and reference the github issue associated
   with your task (ex. #1).  You can create a sequence of separate commits in
   this manner if your task is better broken down into separate components:

.. code-block:: bash

    git add my_updated_file.txt
    git commit -m "Added new awesome functionality.  Closes issue #1"
    git push origin my_new_feature_branch

8. Create a new pull request to get your feature branch merged into master for
   others to use.  You'll first need to ensure your feature branch contains the
   latest changes from master.  Furthermore, internal devs will need to assign
   the request to someone else for a code review.  You should also ensure all
   your tests pass when run through the items defined in step 5.

.. code-block:: bash

    # (external contribs): make a new pull request:
    https://github.com/NervanaSystems/neon/pulls

    # merge latest master changes into your feature branch
    git fetch origin
    git checkout master
    git pull origin master
    git checkout my_new_feature_branch
    git merge master  # you may need to manually resolve any merge conflicts

9. If there are issues you can continue to push commits to your feature branch
   by following step 7.  They will automatically be added to this same merge
   request.

8. Once your change has been successfully merged, you can remove the source
   branch and ensure your local copy is up to date:

.. code-block:: bash

    git fetch origin
    git checkout master
    git pull
    git branch -d my_new_feature_branch
    git branch -d -r origin/my_new_feature_branch

9. Give yourself a high five for a job well done!
