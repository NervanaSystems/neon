#!/bin/bash
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
# Run all subsets of examples sequentially, while collecting timing and
# performance information.  Stats against prior runs are compared as well as
# appended to the named file.
# usage: run_integration_tests.sh [out_file [log_file]]
THIS_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
OUT_FILE="${HOME}/.nervana/integration_test_results.tsv"
LOG_FILE="${HOME}/.nervana/integration_test_results.log"
if [ "$#" -ge 1 ]; then
  OUT_FILE="$1"
  if [ "$#" -ge 2 ]; then
    LOG_FILE="$2"
  fi
fi
NEON_EXE="${THIS_DIR}/../bin/neon"
NEON_OPTS="-r 0 --integration -o ${OUT_FILE}"  # non-distributed
CMP_EXE="${THIS_DIR}/../bin/compare_metrics"
CMP_OPTS=""
CMP_FIRST_OPTS=""

mkdir -p "$(dirname $OUT_FILE)"
mkdir -p "$(dirname $LOG_FILE)"

cpu_yaml=("${THIS_DIR}/recurrent/mobydick-lstm-small.yaml" \
          "${THIS_DIR}/recurrent/mobydick-rnn-small.yaml")
hpu_yaml=("${THIS_DIR}/convnet/i1k-alexnet-fp16.yaml")
gpu_yaml=("${THIS_DIR}/convnet/i1k-alexnet-fp32.yaml")
all_yaml=("${THIS_DIR}/convnet/mnist-small.yaml" \
          "${THIS_DIR}/mlp/mnist-small.yaml" \
          "${THIS_DIR}/convnet/cifar10-small.yaml" \
          "${THIS_DIR}/mlp/cifar10-small.yaml")

cpu_back=("cpu")
hpu_back=("nervanagpu")
gpu_back=("cudanet" "nervanagpu")
all_back=("cpu" "cudanet" "nervanagpu")


run_yamls()
{
  declare -a yamls=("${!1}")
  declare -a backends=("${!2}")
  for yaml in ${yamls[@]}
  do
    for backend in ${backends[@]}
    do
      echo "$(date) - Running: $yaml $backend..." | tee -a "$LOG_FILE"
      PYTHONPATH="${THIS_DIR}/..:${PYTHONPATH}" $NEON_EXE $NEON_OPTS \
      --gpu $backend "$yaml" >> "$LOG_FILE" 2>&1
      if [ $? -eq 0 ]
      then
        PYTHONPATH="${THIS_DIR}/..:${PYTHONPATH}" \
          $CMP_EXE $CMP_OPTS $CMP_FIRST_OPTS "$OUT_FILE" "$yaml"
        if [ "$CMP_FIRST_OPTS" == "" ]
        then
          CMP_FIRST_OPTS="--no_header"
        fi
      else
        echo -e "problems running ${yaml}\t${backend}"
      fi
      echo "" >> "$LOG_FILE"
    done
  done
}

run_yamls hpu_yaml[@] hpu_back[@]
run_yamls gpu_yaml[@] gpu_back[@]
run_yamls cpu_yaml[@] cpu_back[@]
run_yamls all_yaml[@] all_back[@]
