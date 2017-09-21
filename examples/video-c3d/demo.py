#!/usr/bin/env python
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
import os
import numpy as np
import subprocess
import atexit
import shutil
from tempfile import mkdtemp
from neon.models import Model
from neon.util.argparser import NeonArgparser
from data import make_inference_loader


def segment_video(infile, tmpdir):
    segments_out = os.path.join(tmpdir, 'segment.csv')
    cmd = '''ffmpeg  -i {0} -an -vf scale=171:128 -framerate 25 -c:v mjpeg -q:v 3 \
    -f segment -segment_time 0.64 -reset_timestamps 1 -segment_list {1} \
    -segment_list_entry_prefix {2}/ -y {2}/clip_%02d.avi'''
    proc = subprocess.Popen(cmd.format(infile, segments_out, tmpdir), shell=True)
    proc.communicate()

    manifest_file = os.path.join(tmpdir, 'manifest.csv')
    all_clips = np.genfromtxt(segments_out, dtype=None, delimiter=',')
    all_clips = np.atleast_1d(all_clips)
    valid_clips = [l[0].decode() for l in all_clips if l[2] - l[1] > 0.63]
    np.savetxt(manifest_file, valid_clips, fmt='%s', header='@FILE', comments='')
    return manifest_file


def caption_video(infile, caption, outfile):
    cmd = '''ffmpeg  -i {0} -an \
    -vf drawtext="textfile={1}: fontcolor=white: fontsize=16: box=1: boxcolor=black@0.5" \
    -y {2}'''
    proc = subprocess.Popen(cmd.format(infile, caption, outfile), shell=True)
    proc.communicate()


# parse the command line arguments
demo_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test.cfg')
config_files = [demo_config] if os.path.exists(demo_config) else []

parser = NeonArgparser(__doc__, default_config_files=config_files)
parser.add_argument('--input_video', help='video file')
parser.add_argument('--output_video', help='Video file with overlayed inference hypotheses')
args = parser.parse_args()

assert args.model_file is not None, "need a model file for testing"
model = Model(args.model_file)

assert 'categories' in args.manifest, "Missing categories file"
category_map = {t[0].decode(): t[1] for t in np.genfromtxt(args.manifest['categories'],
                                                           dtype=None, delimiter=',')}

# Make a temporary directory and clean up afterwards
outdir = mkdtemp()
atexit.register(shutil.rmtree, outdir)
caption_file = os.path.join(outdir, 'caption.txt')

manifest = segment_video(args.input_video, outdir)

test = make_inference_loader(manifest, model.be)
clip_pred = model.get_outputs(test)
tot_prob = clip_pred[:test.ndata, :].mean(axis=0)
top_5 = np.argsort(tot_prob)[-5:]
category_vals = list(category_map.values())
category_keys = list(category_map.keys())
hyps = ["{:0.5f} {}".format(tot_prob[i],
                            category_keys[category_vals.index(i)]) for i in reversed(top_5)]
np.savetxt(caption_file, hyps, fmt='%s')

caption_video(args.input_video, caption_file, args.output_video)
