# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
# ------------------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see fast-rcnn/LICENSE for details]
# Written by Ross Girshick
# ------------------------------------------------------------------
from neon.backends.util.source_module import SourceModule
from pycuda.tools import context_dependent_memoize

"""
CUDA kernels for ROI pooling layers.
There is a fprop function, a bprop function.
The fprop and bprop CUDA-C code are adapted from Fast R-CNN model.
Each of the kernels uses templating to perform %(type)
conversion so it works for all data types (currently fp32 and fp16 are supported).
"""


def map_string2func(funcname, clss):
    """
    Helper function that converts string function names to function calls
    """
    if funcname == "fprop_roipooling":
        return _get_fprop_roipooling(clss)
    if funcname == "bprop_roipooling":
        return _get_bprop_roipooling(clss)
    raise AttributeError("kernel type '" + funcname + "' not understood")


# This section of the code contains templated CUDA-C code for the kernels.
@context_dependent_memoize
def _get_fprop_roipooling(clss):

    code = r"""
#define FLT_MAX 3.402823466E+38F

__global__ void fprop_roipooling(const int nthreads,
    const int num_rois, const int img_count,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const float* bottom_data, const float* bottom_rois, float* top_data,
    int* argmax_data, const float spatial_scale) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
        index < (nthreads); index += blockDim.x * gridDim.x){
        // (c, ph, pw, n) is an element in the pooled output
        int n = index % num_rois;
        int pw = (index / num_rois) % pooled_width;
        int ph = (index / num_rois / pooled_width) % pooled_height;
        int c = index / num_rois / pooled_width / pooled_height;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        int roi_start_w = round(bottom_rois[1] * spatial_scale);
        int roi_start_h = round(bottom_rois[2] * spatial_scale);
        int roi_end_w = round(bottom_rois[3] * spatial_scale);
        int roi_end_h = round(bottom_rois[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = static_cast<float>(roi_height)
                           / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width)
                           / static_cast<float>(pooled_width);

        int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                            * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                            * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                         * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                         * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;

        bottom_data += c * height * width * img_count;

        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width * img_count + w * img_count + roi_batch_ind;
            if (bottom_data[bottom_index] > maxval) {
              maxval = bottom_data[bottom_index];
              maxidx = bottom_index;
            }
          }
        }
        top_data[index] = maxval;
        argmax_data[index] = maxidx;
        // Notice the maxidx (from bottom_index) is relative to the dimension
        // (h, w, img_count) of the feature map, so max value is HWN
    }
}

"""

    module = SourceModule(code)
    kernel = module.get_function("fprop_roipooling")
    sig = "8I 4P 1f"
    kernel.prepare(sig)
    return kernel


# This section of the code contains templated CUDA-C code for the kernels.
@context_dependent_memoize
def _get_bprop_roipooling(clss):

    code = r"""
__global__ void bprop_roipooling(const int nthreads,
    const int num_rois, const int img_count,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const float* top_diff, const float* bottom_rois, float* bottom_diff,
    const int* argmax_data, const float spatial_scale) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; \
        index < (nthreads); index += blockDim.x * gridDim.x){
        // (c, h, w, n) coords in bottom data on feature map
        int n = index % img_count;
        int w = (index / img_count) % width;
        int h = (index / img_count / width) % height;
        int c = index / img_count/ width / height;

        float gradient = 0;
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
          const float* offset_bottom_rois = bottom_rois + roi_n * 5;
          int roi_batch_ind = offset_bottom_rois[0];
          // Skip if ROI's batch index doesn't match n
          if (n != roi_batch_ind) {
            continue;
          }

          int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
          int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
          int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
          int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

          // Skip if ROI doesn't include (h, w)
          const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                               h >= roi_start_h && h <= roi_end_h);
          if (!in_roi) {
            continue;
          }

          int offset = c * pooled_height * pooled_width * num_rois;
          const float* offset_top_diff = top_diff + offset;
          const int* offset_argmax_data = argmax_data + offset;

          // Compute feasible set of pooled units that could have pooled
          // this bottom unit

          // Force malformed ROIs to be 1x1
          int roi_width = max(roi_end_w - roi_start_w + 1, 1);
          int roi_height = max(roi_end_h - roi_start_h + 1, 1);

          float bin_size_h = static_cast<float>(roi_height)
                             / static_cast<float>(pooled_height);
          float bin_size_w = static_cast<float>(roi_width)
                             / static_cast<float>(pooled_width);

          int phstart = floor(static_cast<float>(h - roi_start_h) / bin_size_h);
          int phend = ceil(static_cast<float>(h - roi_start_h + 1) / bin_size_h);
          int pwstart = floor(static_cast<float>(w - roi_start_w) / bin_size_w);
          int pwend = ceil(static_cast<float>(w - roi_start_w + 1) / bin_size_w);

          phstart = min(max(phstart, 0), pooled_height);
          phend = min(max(phend, 0), pooled_height);
          pwstart = min(max(pwstart, 0), pooled_width);
          pwend = min(max(pwend, 0), pooled_width);

          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              int top_index = ph * pooled_width * num_rois + pw * num_rois + roi_n;
              int bottom_index = h * width * img_count + w * img_count + roi_batch_ind;
              if (offset_argmax_data[top_index] == bottom_index) {
                gradient += offset_top_diff[top_index];
              }
            }
          }
        }
        bottom_diff[index] = gradient;
    }
}

"""

    module = SourceModule(code)
    kernel = module.get_function("bprop_roipooling")
    sig = "8I 4P 1f"
    kernel.prepare(sig)
    return kernel
