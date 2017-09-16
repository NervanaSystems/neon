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
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
"""
The mAP evaluation script and various util functions are adapted from:
https://github.com/rbgirshick/py-faster-rcnn/commit/45e0da9a246fab5fd86e8c96dc351be7f145499f
"""
from __future__ import print_function
import numpy as np


GT_CLASS_INDEX = 4
GT_DIFFICULT_INDEX = 5
GT_DETECTED_INDEX = 6


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(all_boxes, all_gt_boxes, classes, ovthresh=0.5, use_07_metric=False, verbose=True):
    # num_classes = len(classes)

    # take the bounding box detections and flatten in one list
    det_bbox = np.array([b[:5] for idx in all_boxes for b in idx])
    det_cls_idx = np.array([b[-1] for idx in all_boxes for b in idx])
    det_img_idx = np.array([img_idx for (img_idx, idx) in enumerate(all_boxes) for b in idx])

    # MAP = np.zeros((num_classes, 1))
    MAP = {}

    # replicate rounding to match old approach
    det_bbox[:, :4] = np.round(det_bbox[:, :4], decimals=1)
    det_bbox[:, -1] = np.round(det_bbox[:, -1], decimals=3)

    for cls_idx, cls in enumerate(classes):

        if cls == '__background__':
            continue

        # calculate total number of gtboxes of this class across the image set
        npos = 0
        for gt_boxes in all_gt_boxes:
            npos = npos + len(np.where(np.logical_and(gt_boxes[:, GT_CLASS_INDEX] == cls_idx,
                                                      gt_boxes[:, GT_DIFFICULT_INDEX] == 0))[0])

        # grab all the class detections and their associate image id
        idx = np.where(det_cls_idx == cls_idx)[0]
        cls_bb = det_bbox[idx]  # bounding boxes with [xmin, ymin, xmax, ymax, score]
        cls_img_idx = det_img_idx[idx]  # associated image for each bbox detection

        # sort by scores
        sorted_ind = np.argsort(-cls_bb[:, -1])
        cls_bb = cls_bb[sorted_ind, :]
        cls_img_idx = cls_img_idx[sorted_ind]

        # initialize TP and FP
        nd = len(sorted_ind)
        tp = np.zeros(len(sorted_ind))
        fp = np.zeros(len(sorted_ind))

        # go through each of the detections and mark TPs and FPs
        for d in range(nd):
            bb = cls_bb[d, :]
            img_idx = cls_img_idx[d]

            # only use gt boxes from the same class
            bbgt = all_gt_boxes[img_idx]
            box_idx = np.where(bbgt[:, GT_CLASS_INDEX] == cls_idx)[0]
            BBGT = bbgt[box_idx, :]

            ovmax = -np.inf

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not bbgt[box_idx[jmax], GT_DIFFICULT_INDEX]:
                    if not bbgt[box_idx[jmax], GT_DETECTED_INDEX]:
                        tp[d] = 1.
                        bbgt[box_idx[jmax], GT_DETECTED_INDEX] = True
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / (tp + fp + 1e-10)
        ap = voc_ap(rec, prec, True)
        # MAP[cls_idx] = ap
        MAP[cls] = ap
        if verbose:
            print("AP for {} = {:.4f}".format(cls, ap))

    if verbose:
        print("Mean AP = {:.4f}".format(np.mean([MAP[ky] for ky in MAP])))
    return MAP
