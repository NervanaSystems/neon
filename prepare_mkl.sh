#!/bin/sh
# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
# set -ex
FindLibrary()
{
  case "$1" in
    intel|1)
      LOCALMKL=`find $DST -name libmklml_intel.so`   # name of MKL SDL lib
      ;;
    *)
      LOCALMKL=`find $DST -name libmklml_gnu.so`   # name of MKL SDL lib
      ;;
  esac

}

GetVersionName()
{
VERSION_LINE=0
if [ $1 ]; then
  VERSION_LINE=`grep __INTEL_MKL_BUILD_DATE $1/include/mkl_version.h 2>/dev/null | sed -e 's/.* //'`
fi
if [ -z $VERSION_LINE ]; then
  VERSION_LINE=0
fi
echo $VERSION_LINE  # Return Version Line
}

# MKL
DST=`dirname $0`
OMP=0
VERSION_MATCH=20170908
ARCHIVE_BASE=mklml_lnx_2018.0.$VERSION_MATCH
ARCHIVE_BASENAME=$ARCHIVE_BASE.tgz
GITHUB_RELEASE_TAG=v0.10
MKLURL="https://github.com/01org/mkl-dnn/releases/download/$GITHUB_RELEASE_TAG/$ARCHIVE_BASENAME"
# there are diffrent MKL lib to be used for GCC and for ICC
reg='^[0-9]+$'
echo Checking MKLML dependencies...
VERSION_LINE=`GetVersionName $MKLROOT`
# Check if MKLROOT is set if positive then set one will be used..
if [ -z $MKLROOT ] || [ $VERSION_LINE -lt $VERSION_MATCH ]; then
	# ..if MKLROOT is not set then check if we have MKL downloaded in proper version
    VERSION_LINE=`GetVersionName $DST/$ARCHIVE_BASE`
    if [ $VERSION_LINE -lt $VERSION_MATCH ] ; then
      #...If it is not then downloaded and unpacked
      echo Downloading required MKLML version ${ARCHIVE_BASE} ...
      wget --no-check-certificate -P $DST $MKLURL -O $DST/$ARCHIVE_BASENAME > /dev/null 2>&1
      tar -xzf $DST/$ARCHIVE_BASENAME -C $DST > /dev/null 2>&1
    fi
  FindLibrary $1
  MKLROOT=$PWD/`echo $LOCALMKL | sed -e 's/lib.*$//'`
  echo MKLML dependencies installed: MKLROOT=${MKLROOT}
fi

# Check what MKL lib we have in MKLROOT
if [ -z `find $MKLROOT -name libmkl_rt.so -print -quit` ]; then
  LIBRARIES=`basename $LOCALMKL | sed -e 's/^.*lib//' | sed -e 's/\.so.*$//'`
  OMP=1
else
  LIBRARIES="mkl_rt"
fi


# return value to calling script (Makefile,cmake)
echo $MKLROOT $LIBRARIES $OMP
