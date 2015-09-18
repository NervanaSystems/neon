# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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

# Imports should not be a requirement for building documentation
try:
    from bokeh.plotting import figure
    from bokeh.palettes import brewer
    from bokeh.io import vplot, hplot
except ImportError:
    pass

import numpy as np


def x_label(epoch_axis):
    """
    Get the x axis label depending on the boolean epoch_axis.
    """
    return "Epoch" if epoch_axis else "Minibatch"


def cost_fig(cost_data, plot_height, plot_width, epoch_axis=True):
    """
    Generate a figure with lines for each element in cost_data.
    """

    fig = figure(plot_height=plot_height,
                 plot_width=plot_width,
                 title="Cost",
                 x_axis_label=x_label(epoch_axis),
                 y_axis_label="Cross Entropy Error (%)")

    # Spectral palette supports 3 - 11 distinct colors
    num_colors_required = len(cost_data)
    assert num_colors_required <= 11, "Insufficient colors in predefined palette."
    colors = brewer["Spectral"][max(3, len(cost_data))]
    if num_colors_required < 3:
        # manually adjust pallette for better contrast
        colors[0] = brewer["Spectral"][6][0]
        if num_colors_required == 2:
            colors[1] = brewer["Spectral"][6][-1]

    for name, x, y in cost_data:
        fig.line(x, y, legend=name, color=colors.pop(0), line_width=2)
    return fig


def hist_fig(hist_data, plot_height, plot_width, x_range=None, epoch_axis=True):
    """
    Generate a figure with an image plot for hist_data, bins on the Y axis and
    time on the X axis.
    """
    name, hdata, dh, dw, bins, offset = hist_data
    if x_range is None:
        x_range = (0, dw)
    fig = figure(plot_height=plot_height,
                 plot_width=plot_width,
                 title=name,
                 x_axis_label=x_label(epoch_axis),
                 x_range=x_range,
                 y_range=(offset, offset + bins))
    fig.image(image=[hdata], x=[0],  y=[offset], dw=[dw], dh=[dh], palette="Spectral11")
    return fig


def scale_to_rgb(img):
    """
    Convert float data to valid RGB values in the range [0, 255]

    Arguments:
        img (ndarray): the image data

    Returns:
        img (ndarray): image array with valid RGB values
    """
    absMax = np.max((abs(img)))
    minVal = - absMax
    img -= minVal
    maxImg = np.max(img)
    maxVal = max(absMax - minVal, maxImg)
    if maxVal == 0:
        maxVal = 1
    img = img / maxVal * 255
    return img


def convert_rgb_to_bokehrgba(img_data, dh, dw):
    """
    Convert RGB image to two-dimensional array of RGBA values (encoded as 32-bit integers)
    (required by Bokeh). The functionality is currently not available in Bokeh.
    An issue was raised here: https://github.com/bokeh/bokeh/issues/1699 and this function is a
    modified version of the suggested solution.

    Arguments:
        img_data: img (ndarray, shape: [N, M, 3], dtype: uint8): image data
        dh: height of image
        dw: width of image

    Returns:
        img (ndarray): 2D image array of RGBA values
    """
    if img_data.dtype != np.uint8:
        raise NotImplementedError

    if img_data.ndim != 3:
        raise NotImplementedError

    bokeh_img = np.dstack([img_data, 255 * np.ones(img_data.shape[:2], np.uint8)])
    # This step is necessary, because we transposed the data before passing it into this function
    # and somehow, that messed with some shape attribute.
    final_image = bokeh_img.reshape(dh * dw * 4).view(np.uint32)
    final_image = final_image.reshape((dh, dw))

    return final_image


def deconv_fig(layer_data, plot_size, figs_per_row=10):
    """
    Generate a figure for each projection of feature map activations back to pixel space.

    Arguments:
        img_data (tuple): feature map name, array with image rgb values
        plot_size (int): height and width of plot
        figs_per_row (int, optional): the number of images to plot in a row
    """
    rows = list()
    rowfigs = list()
    img_h, img_w = layer_data[0][1].shape[1], layer_data[0][1].shape[2]

    for fm_num, (fm_name, deconv_data, img_data) in enumerate(layer_data):
        if fm_num > 30:
            break
        img_data = img_data.reshape((3, img_h, img_w))
        img_data = np.transpose(img_data, (1, 2, 0))
        deconv_data = np.transpose(deconv_data, (1, 2, 0))

        deconv_rgb = scale_to_rgb(deconv_data)
        deconv_rgb = deconv_rgb.astype(np.uint8)
        img_rgb = scale_to_rgb(img_data)
        img_rgb = img_rgb.astype(np.uint8)

        deconv_final = convert_rgb_to_bokehrgba(deconv_rgb, img_h, img_w)
        deconv_final = deconv_final.flatten()
        deconv_final = deconv_final[::-1]
        deconv_final = deconv_final.reshape((img_h, img_w))
        img_final = convert_rgb_to_bokehrgba(img_rgb, img_h, img_w)
        img_final = img_final.flatten()
        img_final = img_final[::-1]
        img_final = img_final.reshape((img_h, img_w))

        deconv_fig = figure(title=str(fm_num+1), title_text_font_size='6pt',
                            x_range=[0, img_w], y_range=[0, img_h],
                            plot_width=plot_size-15, plot_height=plot_size,
                            toolbar_location=None)
        deconv_fig.axis.visible = None
        deconv_fig.image_rgba([deconv_final][::-1], x=[0], y=[0], dw=[img_w], dh=[img_h])
        deconv_fig.min_border = 0

        img_fig = figure(title=str(fm_num+1), title_text_font_size='6pt',
                         x_range=[0, img_w], y_range=[0, img_h],
                         plot_width=plot_size-15, plot_height=plot_size,
                         toolbar_location=None)
        img_fig.axis.visible = None
        img_fig.image_rgba([img_final][::-1], x=[0], y=[0], dw=[img_w], dh=[img_h])
        img_fig.min_border = 0

        if len(rowfigs) < figs_per_row:
            rowfigs.append(deconv_fig)
            rowfigs.append(img_fig)
        else:
            rows.append(hplot(*rowfigs))
            rowfigs = list()

    if len(rowfigs):
        rows.append(hplot(*rowfigs))

    allfigs = vplot(*rows)

    return allfigs
