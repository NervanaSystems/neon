# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
    from bokeh.models import Range1d
    from bokeh.embed import components
    from jinja2 import Template
except ImportError:
    pass


def x_label(epoch_axis):
    """
    Get the x axis label depending on the boolean epoch_axis.

    Arguments:
        epoch_axis (bool): If true, use Epoch, if false use Minibatch

    Returns:
        str: "Epoch" or "Minibatch"
    """
    return "Epoch" if epoch_axis else "Minibatch"


def cost_fig(cost_data, plot_height, plot_width, epoch_axis=True):
    """
    Generate a figure with lines for each element in cost_data.

    Arguments:
        cost_data (list): Cost data to plot
        plot_height (int): Plot height
        plot_width (int): Plot width
        epoch_axis (bool, optional): If true, use Epoch, if false use Minibatch

    Returns:
        bokeh.plotting.figure: cost_data figure
    """

    fig = figure(plot_height=plot_height,
                 plot_width=plot_width,
                 title="Cost",
                 x_axis_label=x_label(epoch_axis),
                 y_axis_label="Cross Entropy Error (%)")

    # Spectral palette supports 3 - 11 distinct colors
    num_colors_required = len(cost_data)
    assert num_colors_required <= 11, "Insufficient colors in predefined palette."
    colors = list(brewer["Spectral"][max(3, len(cost_data))])
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

    Arguments:
        hist_data (tuple): Hist data to plot
        plot_height (int): Plot height
        plot_width (int): Plot width
        x_range (tuple, optional): (start, end) range for x
        epoch_axis (boolm optional): If true, use Epoch, if false use Minibatch

    Returns:
        bokeh.plotting.figure: hist_data figure
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
    fig.image(image=[hdata], x=[0], y=[offset], dw=[dw], dh=[dh], palette="Spectral11")
    return fig


def image_fig(data, h, w, x_range, y_range, plot_size):
    """
    Helper function to generate a figure

    Arguments:
        data (int): data to plot
        h (int): height
        w (int): width
        x_range (tuple, optional): (start, end) range for x
        y_range (tuple, optional): (start, end) range for y
        plot_size (int): plot size

    Returns:
        bokeh.plotting.figure: Generated figure
    """
    fig = figure(x_range=x_range, y_range=y_range,
                 plot_width=plot_size, plot_height=plot_size,
                 toolbar_location=None)
    fig.image_rgba([data], x=[0], y=[0], dw=[w], dh=[h])
    fig.axis.visible = None
    fig.min_border = 0
    return fig


def deconv_figs(layer_name, layer_data, fm_max=8, plot_size=120):
    """
    Helper function to generate deconv visualization figures

    Arguments:
        layer_name (str): Layer name
        layer_data (list): Layer data to plot
        fm_max (int): Max layers to process
        plot_size (int, optional): Plot size

    Returns:
        tuple if vis_keys, img_keys, fig_dict
    """
    vis_keys = dict()
    img_keys = dict()
    fig_dict = dict()

    for fm_num, (fm_name, deconv_data, img_data) in enumerate(layer_data):

        if fm_num >= fm_max:
            break

        img_h, img_w = img_data.shape
        x_range = Range1d(start=0, end=img_w)
        y_range = Range1d(start=0, end=img_h)
        img_fig = image_fig(img_data, img_h, img_w, x_range, y_range, plot_size)
        deconv_fig = image_fig(deconv_data, img_h, img_w, x_range, y_range, plot_size)

        title = "{}_fmap_{:04d}".format(layer_name, fm_num)
        vis_keys[fm_num] = "vis_" + title
        img_keys[fm_num] = "img_" + title

        fig_dict[vis_keys[fm_num]] = deconv_fig
        fig_dict[img_keys[fm_num]] = img_fig

    return vis_keys, img_keys, fig_dict


def deconv_summary_page(filename, cost_data, deconv_data):
    """
    Generate an HTML page with a Deconv visualization

    Arguments:
        filename: Output filename
        cost_data (list): Cost data to plot
        deconv_data (tuple): deconv data to plot
    """
    fig_dict = dict()

    cost_key = "cost_plot"
    fig_dict[cost_key] = cost_fig(cost_data, 300, 533, epoch_axis=True)

    vis_keys = dict()
    img_keys = dict()
    for layer, layer_data in deconv_data:
        lyr_vis_keys, lyr_img_keys, lyr_fig_dict = deconv_figs(layer, layer_data, fm_max=4)
        vis_keys[layer] = lyr_vis_keys
        img_keys[layer] = lyr_img_keys
        fig_dict.update(lyr_fig_dict)

    script, div = components(fig_dict)

    template = Template('''
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>{{page_title}}</title>
        <style> div{float: left;} </style>
        <link rel="stylesheet"
              href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
              type="text/css" />
        <script type="text/javascript"
                src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>
        {{ script }}
    </head>
    <body>
    <div id=cost_plot style="width:100%; padding:10px">
      {{ div[cost_key]}}
    </div>

    {% for layer in sorted_layers %}
        <div id=Outer{{layer}} style="padding:20px">
        <div id={{layer}} style="background-color: #C6FFF1; padding:10px">
        Layer {{layer}}<br>
        {% for fm in vis_keys[layer].keys() %}
            <div id={{fm}} style="padding:10px">
            Feature Map {{fm}}<br>
            {{ div[vis_keys[layer][fm]] }}
            {{ div[img_keys[layer][fm]] }}
            </div>
        {% endfor %}
        </div>
        </div>

        <br><br>
    {% endfor %}
    </body>
</html>
''')

    with open(filename, 'w') as htmlfile:
        htmlfile.write(template.render(page_title="Deconv Visualization", script=script,
                                       div=div, cost_key=cost_key, vis_keys=vis_keys,
                                       img_keys=img_keys,
                                       sorted_layers=sorted(vis_keys)))
