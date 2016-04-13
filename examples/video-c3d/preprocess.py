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
"""
Preprocess videos by resizing and splitting videos into clips.
"""

import cv2
import logging
import math
import os
import configargparse
import numpy as np

logger = logging.getLogger(__name__)


def check_opencv_version(major):
    return cv2.__version__.startswith(major)


class VideoPreprocessor():
    """
    Utility class for preprocessing videos.


    Arguments:
        num_frames_per_clip (int): Split videos into clips. Each clip
                                   will have this number of frames.
        clip_resize_dim ((int, int)): Height and width to resize every frame.
        video_dir (str): Root directory of raw video data.
        data_split_file (str): Path of two column file indicating
                               examples and their corresponding labels
                               in a given data split.
        class_ind_file (str): Path of two column file mapping integer
                              class labels to their canonical names.
        preprocessed_dir (str): Output directory of preprocessed clips.
    """

    def __init__(self, num_frames_per_clip, clip_resize_dim, video_dir,
                 data_split_file, class_ind_file, preprocessed_dir):
        self.num_frames_per_clip = num_frames_per_clip
        self.clip_resize_dim = clip_resize_dim
        self.preprocessed_dir = preprocessed_dir
        self.video_dir = video_dir
        self.data_split_file = data_split_file
        self.class_ind_file = class_ind_file
        self.num_frames_per_clip = num_frames_per_clip
        self.clip_resize_dim = clip_resize_dim

    def get_clip_path(self, video_path, clip_idx):
        video_dir = os.path.dirname(video_path)
        if self.video_dir is not None:
            reldir = os.path.relpath(video_dir, self.video_dir)
            preprocessed_clip_path = os.path.join(self.preprocessed_dir, reldir)
            if not os.path.exists(preprocessed_clip_path):
                os.makedirs(preprocessed_clip_path)
        else:
            preprocessed_clip_path = self.preprocessed_dir

        basename = os.path.basename(video_path)
        filename, extension = os.path.splitext(basename)
        clip_file = os.path.join(preprocessed_clip_path, filename)
        return clip_file + "_" + str(clip_idx) + extension

    def preprocess_video(self, video_path, label):
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)

        if check_opencv_version('3.'):
            codec_func = cv2.VideoWriter_fourcc
            cap_prop_fps = cv2.CAP_PROP_FPS
            cap_prop_frame_count = cv2.CAP_PROP_FRAME_COUNT
        else:
            codec_func = cv2.cv.CV_FOURCC
            cap_prop_fps = cv2.cv.CV_CAP_PROP_FPS
            cap_prop_frame_count = cv2.cv.CV_CAP_PROP_FRAME_COUNT

        codec = codec_func(*'MJPG')

        video = cv2.VideoCapture(video_path)
        fps = video.get(cap_prop_fps)
        if np.isnan(fps):
            fps = 25
        frame_count = video.get(cap_prop_frame_count)
        num_clips = math.floor(frame_count/self.num_frames_per_clip)

        clip_idx = 0
        frame_idx = 0
        while clip_idx < num_clips:
            if frame_idx == 0:
                clip_path = self.get_clip_path(video_path, clip_idx)
                clip = cv2.VideoWriter(clip_path, codec, fps, self.clip_resize_dim)

            ret, frame = video.read()

            if not ret:
                break

            frame = cv2.resize(frame, self.clip_resize_dim)
            clip.write(frame)

            frame_idx += 1

            if frame_idx == self.num_frames_per_clip:
                clip_idx += 1
                frame_idx = 0

    def preprocess_list(self, video_list):
        """
        Preprocess a list of videos.

        Arguments:
            video_list (list of 2-tuples): List of 2-tuples. The first element
                                           is the video path and the second is
                                           the label.
        """

        for video_path, label in video_list:
            if self.video_dir is not None:
                video_path = os.path.join(self.video_dir, video_path)

            self.preprocess_video(video_path, label)

    def run(self):
        label_index = {}
        with open(self.class_ind_file) as label_index_file:
            for line in label_index_file:
                index, label = line.split()
                label_index[label] = int(index) - 1

        with open(self.data_split_file) as data_split:
            first_line = next(data_split)
            num_columns = len(first_line.split())

        video_list = []
        with open(self.data_split_file) as data_split:
            for line in data_split.read().strip().split("\r\n"):
                if num_columns == 2:
                    name, label = line.split()
                    video_list.append((name, int(label) - 1))
                elif num_columns == 1:
                    label_string, basename = line.split("/")
                    video_list.append((line, label_index[label_string]))
                else:
                    raise ValueError("Invalid data split file.")

        self.preprocess_list(video_list)

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add_argument('--video_dir', required=True,
                        help='Root directory of raw video data.')
    parser.add_argument('--data_split_file', required=True,
                        help='Path of two column file indicating'
                             'examples and their corresponding labels'
                             'in a given data split.')
    parser.add_argument('--class_ind_file', required=True,
                        help='Path of two column file mapping integer'
                             'class labels to their canonical names.')
    parser.add_argument('--preprocessed_dir', required=True,
                        help='Output directory of preprocessed clips.')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    num_frames_per_clip = 16
    clip_resize_dim = (128, 171)
    vpp = VideoPreprocessor(
        num_frames_per_clip=num_frames_per_clip,
        clip_resize_dim=clip_resize_dim,
        video_dir=args.video_dir,
        data_split_file=args.data_split_file,
        class_ind_file=args.class_ind_file,
        preprocessed_dir=args.preprocessed_dir
    )
    vpp.run()
