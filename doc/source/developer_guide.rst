.. ---------------------------------------------------------------------------
.. Copyright 2015 Nervana Systems Inc.
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

Developer Guide
===============

.. include:: ../../CONTRIBUTING.rst

Coding Conventions
------------------

To ensure consistency across developers, we enforce the following conventions
on all of the code in neon (which is primarily python plus some
reStructuredText for documentation).

* By and large we conform to PEP8_ with the following exceptions:

  * Maximum line length is 99 characters

* All public class and function names must have docstrings, and these
  docstrings must conform to `Google Style Docstrings`_ as our API
  documentation is auto-generated from these.  To do this we utilize
  the Napoleon_ Sphinx extensions.

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _Google Style Docstrings: http://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Comments
.. _Napoleon: http://sphinxcontrib-napoleon.readthedocs.org/en/latest/index.html

Documentation Conventions
-------------------------

* Documents are created using `reStructuredText`_.

* Limit your docs to 2-3 levels of headings. 

* New .rst files will show up in the sidebar, and any first and second level headings will also show up in the menu. Keep the sidebar short and only add essentials items there. Otherwise, add your documentation to the pre-existing files. You can add to the toctree manually, but please don't add or create pages unless absolutely necessary!

* If you created a new class, add it to the API by creating a new section in api.rst and create an autosummary_. Anytime you add an autosummary, please remember to add :nosignatures: to keep things consistent with the rest of our docs. 

* Every time you make a significant contribution, add a short description of it in the relevant document. 

.. _reStructuredText: http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html
.. _autosummary: http://sphinx-doc.org/latest/ext/autosummary.html

