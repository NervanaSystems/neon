import numpy as np
from neon.callbacks.callbacks import Callback
from neon import logger
from inference import get_boxes
from util.voc_eval import voc_eval as voc_eval
from util.util import plot_image as plot_image
import os
"""
We include two callbacks that can be activated during training.

1. The MAP Callback computes the Mean Average Precision at every epoch
2. The ssd_image_callback saves sample inference results at every epoch

"""


class MAP_Callback(Callback):

    def __init__(self, eval_set, epoch_freq=1):
        super(MAP_Callback, self).__init__(epoch_freq=epoch_freq)
        self.eval_set = eval_set

    def on_epoch_end(self, callback_data, model, epoch):
        all_boxes = []
        all_gt_boxes = []
        logger.info('Calculating Mean AP on eval set')
        self.eval_set.reset()
        (all_boxes, all_gt_boxes) = get_boxes(model, self.eval_set)

        MAP = voc_eval(
            all_boxes,
            all_gt_boxes,
            self.eval_set.CLASSES,
            use_07_metric=True,
            verbose=False)
        logger.info('AP scores: %s' % ' '.join([ky+':' + '%.2f' % val for ky, val in MAP.items()]))
        logger.info('Mean AP: %.2f' % np.mean([MAP[ky] for ky in MAP]))


class ssd_image_callback(Callback):

    def __init__(self, eval_set, image_dir, classes,
                 plot_labels=True, num_images=5, epoch_freq=1,
                 score_threshold=0.6):
        super(ssd_image_callback, self).__init__(epoch_freq=epoch_freq)
        self.eval_set = eval_set
        self.image_dir = image_dir
        self.num_images = num_images
        self.classes = classes
        self.score_threshold = score_threshold

        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
            logger.info('Creating folder for sample images: {}'.format(image_dir))

    def on_epoch_end(self, callback_data, model, epoch):

        self.eval_set.reset()
        eval_set = self.eval_set.__iter__()

        n = 0
        while n < self.num_images:
            (x, t) = eval_set.next()
            (gt_boxes, gt_classes, num_gt_boxes, difficult, im_shape) = t

            gt_boxes = gt_boxes.reshape((-1, 4, self.be.bsz))

            outputs = model.fprop(x, inference=True)

            images = x.get()
            for k, output in enumerate(outputs):
                ngt = num_gt_boxes[0, k]

                # creates a PIL image object with the gt_boxes and predicted boxes
                img = plot_image(img=images[:, k], im_shape=im_shape[:, k],
                                 gt_boxes=gt_boxes[:ngt, :, k], boxes=output,
                                 score_threshold=self.score_threshold)

                file_name = os.path.join(self.image_dir, 'image_e{}_{}.jpg'.format(epoch, n))
                img.save(file_name)
                # logger.info('Saved sample images to: {}'.format(file_name))

                n = n + 1
                if n >= self.num_images:
                    break
