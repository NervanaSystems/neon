from neon import NervanaObject
import numpy as np
from util import util
from neon.transforms import Softmax

EPS = np.finfo(np.float32).tiny


class MBoxLoss(NervanaObject):
    def __init__(self, num_classes=21, share_location=True,
                 clip=False, background_label_id=0,
                 use_difficult=True, mining_type='MAX_NEGATIVE', loc_weight=1.0,
                 neg_overlap=0.5, neg_pos_ratio=3.0, overlap_threshold=0.5,
                 loc_loss_type='SMOOTH_L1', conf_loss_type='SOFTMAX', use_prior_for_matching=True,
                 match_type='PER_PREDICTION', convolution=False, use_prior_for_nms=False,
                 name=None):

        self.costfunc = self  # this is needed for the progress bar callback
        self.name = name
        # encode_variance_in_target is False

        # only implemented these types so far
        assert mining_type == 'MAX_NEGATIVE', 'other methods not supported'
        assert share_location, 'Non sharing not supported yet'
        assert loc_loss_type == 'SMOOTH_L1'
        assert conf_loss_type == 'SOFTMAX'
        assert use_prior_for_matching
        assert match_type == 'PER_PREDICTION'
        assert not convolution
        assert not use_prior_for_nms

        self.overlap_threshold = overlap_threshold
        self.neg_overlap = neg_overlap
        self.neg_pos_ratio = neg_pos_ratio

        self.loc_weight = loc_weight

        self.use_prior_for_nms = use_prior_for_nms
        self.convolution = convolution
        self.use_prior_for_matching = use_prior_for_matching
        self.match_type = match_type
        self.loc_loss_type = loc_loss_type
        self.conf_loss_type = conf_loss_type
        self.mining_type = mining_type

        self.background_label_id = background_label_id
        self.use_difficult = use_difficult
        self.mining_type = mining_type

        self.share_location = share_location

        self.num_classes = num_classes
        self.num_ = self.be.bsz
        self.loc_classes = 1 if share_location else self.num_classes

        self.clip = clip
        self.loc_deltas_dev = None
        self.conf_deltas_dev = None

        self.loss = None
        self.y_conf_dev = None
        self.softmax_dev = Softmax(axis=1)

    def initialize(self, *args, **kwargs):
        # just needed to interface with neon.models.model properly
        pass

    def get_cost(self, y, t):
        y_loc = y[0]  # (batch, num_preds/class, 4)
        y_conf = y[1]  # (batch, num_preds/class, num_classes)
        priors = y[2]  # Nx4 array  (xmin, ymin, xmax, ymax)
        t_loc = t[0]  # (max_boxes*4, batch) max_boxes x 4 [xmin, ymin, xmax, ymax] in pixels
        t_conf = t[1]  # (max_boxes, batch)
        n_gt = t[2]  # (1, barch)

        self.num_priors = priors.shape[0]

        # for now, transpose the input to the shape expected by this layer
        y_loc = y_loc.get().reshape(-1, 4, self.be.bsz).transpose((2, 0, 1))
        y_conf = y_conf.reshape(-1, self.num_classes, self.be.bsz)

        # allocate buffers needed to store intermediate results
        if self.loss is None:
            self.loss = self.be.empty((self.be.bsz * self.num_priors, self.num_classes))

        if self.y_conf_dev is None:
            self.y_conf_dev = self.be.empty(self.loss.shape).reshape((self.be.bsz, -1,
                                                                      self.num_classes))

        # transpose the results to y_conf_dev (needed for softmax)
        self.be.copy_transpose(y_conf, self.y_conf_dev, axes=(2, 0, 1))

        y_conf = self.y_conf_dev.get()

        assert y_conf.shape[0] == self.be.bsz
        assert y_conf.shape[1] == priors.shape[0]
        assert y_conf.shape[2] == self.num_classes

        assert y_loc.shape[0] == self.be.bsz
        assert y_loc.shape[1] == priors.shape[0]

        # layer shape is the output shape of the convolution
        # layer generating the input to the priorbox layer

        # get the GT boxes
        gt_boxes = self.gen_gt_boxes(t_loc, t_conf, n_gt)

        # get the model box predictions
        loc_data = self.gen_locs(y_loc, priors.shape[0])

        # find GT prior boxes matches
        matches = self.find_matches(priors, gt_boxes)

        # the conf loss is added to matches
        # sparse take of loss from matching ground truth
        self.get_conf_loss(self.y_conf_dev, matches, gt_boxes)

        # negative sampling
        # maybe here we need to move to host?
        (all_neg_inds, num_matches) = self.mine_examples(matches)

        self.num_matches = num_matches
        if num_matches > 0:
            (loc_gt_data, loc_pred_data) = self.encodeLocPrediction(num_matches, matches, loc_data,
                                                                    gt_boxes, priors)
            # get a smooth L1 loss for these two vectors
            # may also need to stor values to bprop
            self.loc_diff_data = (loc_pred_data - loc_gt_data)
            loc_loss = np.sum(self.smoothL1loss(self.loc_diff_data))
        else:
            self.loc_diff_data = 0.0
            loc_loss = 0.0

        self.all_neg_inds = all_neg_inds
        num_conf = num_matches + sum([len(all_neg_inds[bt]) for bt in all_neg_inds])
        if num_conf >= 1:
            # conf_gt is num_conf x 1
            # conf_pred is num_conf x num_classes
            (conf_gt, conf_pred) = self.EncodeConfPrediction(num_conf, y_conf, matches,
                                                             all_neg_inds, gt_boxes)
            # conf_gt has the GT labels
            # conf_pred has the probs
            # softmax loss
            self.probs = self.softmax(conf_pred, axis=1)
            self.labels = conf_gt

            conf_loss = np.sum(-1 * np.log(np.maximum(EPS, self.probs[range(self.probs.shape[0]),
                                                      self.labels])))
        else:
            conf_loss = 0.0

        self.matches = matches
        self.norm_ = self.be.bsz*self.num_priors if num_matches == -1 else num_matches
        loss = (self.loc_weight*loc_loss + conf_loss) / self.norm_

        self.cost = loss
        return loss

    def get_errors(self, x, t):
        # back propogate the loc and conf losses
        if getattr(self, 'loc_deltas', None) is None:
            self.loc_deltas = np.zeros((self.be.bsz, self.num_priors, 4))
        else:
            self.loc_deltas[:] = 0.0

        if getattr(self, 'conf_deltas', None) is None:
            self.conf_deltas = np.zeros((self.be.bsz, self.num_priors, self.num_classes))
        else:
            self.conf_deltas[:] = 0.0

        loc_deltas = self.loc_deltas
        conf_deltas = self.conf_deltas

        num_matches = self.num_matches
        matches = self.matches

        if num_matches > 0:
            # backprop the smooth L1 loss first
            loc_diff = self.loc_diff_data  # this is the errors calculated above
            loc_diff[self.inds_more] = np.sign(loc_diff[self.inds_more])

            # backprop the loc loss
            # sclae the loc_predictions by num_matches
            loc_diff /= float(num_matches)

            count = 0
            for batch, match in enumerate(matches):
                match_inds = match['inds']
                # load up the errors to the proper prior box location
                # in the loc network outputs matches the diffs up with
                # the correct location in loc_deltas which is the full
                # length of all the location predictions concated together
                inds_ = np.where(match_inds > -1)[0]
                loc_deltas[batch, inds_, :] = loc_diff[range(count, count+len(inds_)), :]
                count += len(inds_)
        else:
            loc_deltas[:] = 0.0

        # first bprop the softmax
        conf_diff = self.probs.copy()  # may not need copy, could do in place?
        for ind in range(len(self.labels)):
            conf_diff[ind, self.labels[ind]] -= 1.0

        conf_diff /= float(self.num_matches)

        count = 0
        for batch, match in enumerate(matches):
            match_inds = match['inds']
            inds_ = np.where(match_inds > -1)[0]
            self.conf_deltas[batch, :, :] = 0.0
            self.conf_deltas[batch, inds_, :] = conf_diff[range(count, count+len(inds_)), :]
            count += len(inds_)

            inds_ = self.all_neg_inds[batch]
            assert np.all(inds_ < self.num_priors)
            self.conf_deltas[batch, inds_, :] = conf_diff[range(count, count+len(inds_)), :]
            count += len(inds_)

        # for now, to match rest of the network, transpose from (N, K, 4) to (K4, N)
        # and load onto device
        loc_deltas = np.ascontiguousarray(
                      self.loc_deltas.reshape((self.be.bsz, -1)).transpose((1, 0)))
        conf_deltas = np.ascontiguousarray(
                      self.conf_deltas.reshape((self.be.bsz, -1)).transpose((1, 0)))
        if self.conf_deltas_dev is None:
            self.conf_deltas_dev = self.be.array(conf_deltas)
        else:
            self.conf_deltas_dev.set(conf_deltas)

        if self.loc_deltas_dev is None:
            self.loc_deltas_dev = self.be.array(loc_deltas)
        else:
            self.loc_deltas_dev.set(loc_deltas)

        return (self.loc_deltas_dev, self.conf_deltas_dev)

    def EncodeConfPrediction(self, num_conf, y_conf, all_matches, all_neg_inds, all_gt_boxes):

        conf_gt = np.zeros((num_conf, ), dtype=np.int) + self.background_label_id
        conf_pred = np.zeros((num_conf, self.num_classes))

        num_priors = y_conf.shape[1]

        count = 0
        for batch in range(self.be.bsz):
            gt_boxes = all_gt_boxes[batch]
            if len(gt_boxes) == 0:
                # no GT labels for this image
                continue

            matches = all_matches[batch]
            match_inds = matches['inds']
            assert len(match_inds) == num_priors

            positive_inds = np.where(match_inds > -1)[0]
            negative_inds = all_neg_inds[batch]

            n_pos = len(positive_inds)
            n_neg = len(negative_inds)

            # store positive matches
            conf_pred[count:(count+n_pos), :] = y_conf[batch, positive_inds, :]
            conf_gt[count:(count+n_pos)] = gt_boxes[match_inds[positive_inds], 4]
            count += n_pos

            # pad with negative matches
            conf_pred[count:(count+n_neg), :] = y_conf[batch, negative_inds, :]
            conf_gt[count:(count+n_neg)] = self.background_label_id
            count += n_neg

        return (conf_gt, conf_pred)

    def encodeBBox(self, prior, bbox):
        self.variances = [0.1, 0.1, 0.2, 0.2]

        xmin = 0.5*(bbox[2] + bbox[0] - prior[2] - prior[0]) / (
               prior[2] - prior[0]) / self.variances[0]
        ymin = 0.5*(bbox[3] + bbox[1] - prior[3] - prior[1]) / (
               prior[3] - prior[1]) / self.variances[1]
        xmax = np.log((bbox[2] - bbox[0]) / (prior[2] - prior[0])) / self.variances[2]
        ymax = np.log((bbox[3] - bbox[1]) / (prior[3] - prior[1])) / self.variances[3]
        return [xmin, ymin, xmax,  ymax]

    # find matching gt_boxes and prior boxes and compute the regression target
    def encodeLocPrediction(self, num_matches, matches, loc_data, gt_boxes, priors):
        # output regression gt data
        loc_gt_data = np.zeros((num_matches, 4))
        loc_pred_data = np.zeros((num_matches, 4))
        cnt = 0

        for batch in range(self.be.bsz):
            match_inds = matches[batch]['inds']

            inds = np.where(match_inds > -1)[0]
            for ind in inds:

                gt_ind = match_inds[ind]
                gt_box = gt_boxes[batch][gt_ind, :]

                loc_gt_data[cnt, :] = self.encodeBBox(priors[ind], gt_box)
                loc_pred_data[cnt, :] = loc_data[batch][ind, :]
                cnt += 1

        return (loc_gt_data, loc_pred_data)

    def count_matches(self, matches):
        num_matches = 0
        for batch in range(self.be.bsz):
            inds = matches[batch]['inds']
            num_matches += len(np.where(inds > -1)[0])

        return num_matches

    def mine_examples(self, matches):
        num_matches = self.count_matches(matches)

        # go through each image in the batch
        all_neg_inds = {}
        for batch in range(self.be.bsz):
            # loss  = conf_loss + loc_loss but loc_loss for max_negative is 0

            # loop over the matches
            # we have only implemented share_location = True so the matches
            # are not done by location, BBs are shared across all classes
            match_inds = matches[batch]['inds']
            match_ovlp = matches[batch]['overlap']
            sel_inds = set()
            num_sel = 0
            loss_inds = []
            loss = matches[batch]['conf_loss']
            num_pos = np.count_nonzero(match_inds > -1)

            linds = np.where((match_inds == -1) & (match_ovlp < self.neg_overlap))[0]
            loss_inds = np.vstack([loss[linds], linds]).T
            num_sel = min([int(num_pos*self.neg_pos_ratio), linds.shape[0]])

            # sort the losses
            sort_inds = np.argsort(loss_inds[:, 0])

            sel_inds = np.unique(loss_inds[sort_inds[-num_sel:], 1].astype(np.int))
            cnt = num_sel + 1
            # the loop below should seldom run i think
            while (len(sel_inds) < num_sel):
                if cnt > sort_inds.shape[0]:
                    break
                sel_inds = np.union1d(sel_inds, [np.int(loss_inds[sort_inds[-cnt], 1])])
                cnt += 1

            assert np.all(match_inds[sel_inds] == -1)
            all_neg_inds[batch] = sel_inds
        return (all_neg_inds, num_matches)

    def isEligibleMining(self, mind, movlp):
        return mind == -1 and movlp < self.neg_overlap

    def softmax(self, x, axis=0):
        exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def smoothL1loss(self, x):

        loc_loss_vec = np.abs(x)
        self.inds_less = np.where(loc_loss_vec < 1.0)
        self.inds_more = np.where(loc_loss_vec >= 1.0)
        loc_loss_vec[self.inds_less] = 0.5*loc_loss_vec[self.inds_less]**2
        loc_loss_vec[self.inds_more] -= 0.5

        return loc_loss_vec

    def get_conf_loss(self, y_conf, matches, gt_boxes):
        # y_conf (batch, num_preds/class, num_classes)
        # match the caffe version where things are (HWC, N)
        # y_conf should be (batch_size, num_preds_per_class, num_classes)

        # for each of the confidences we need to the GT label
        # which comes from the matches

        # calculate the softmax loss
        prob = self.softmax_dev(y_conf.reshape(-1, self.num_classes))
        self.loss[:] = -1 * self.be.log(prob)

        loss = self.loss.reshape(y_conf.shape).get()

        for batch in range(self.be.bsz):
            match_inds = matches[batch]['inds']
            gt_boxes_ = gt_boxes[batch]
            if len(gt_boxes_) == 0:
                continue

            label = match_inds.copy().astype(np.int)
            label[match_inds == -1] = self.background_label_id
            inds = np.where(match_inds > -1)[0]
            label[inds] = gt_boxes_[match_inds[inds], 4]

            matches[batch]['conf_loss'] = loss[batch, range(loss.shape[1]), label]

    def find_matches(self, priors, gt):
        assert self.use_prior_for_matching
        assert self.share_location

        matches = len(gt) * [None]

        for batch, gt_boxes in enumerate(gt):

            # check to see if there are any GT boxes for this image
            if len(gt_boxes) == 0:
                # if not then all the predictions are negative samples
                # may need to return all -1 for inds and 0 for overlap with same shape as locs?
                print('WARNING - found image with no ground truth box')
                continue

            # 1. Compute the overlap of each prior box with each ground truth box
            overlaps = util.calculate_bb_overlap(priors, gt_boxes[:, :4])
            # overlaps[overlaps <= 1.0e-6] = 0.0

            # 2. Use overlaps to compute the gt box each prior box is 'closest' to
            gt_assignment = overlaps.argmax(axis=1)  # max_inds
            match_overlaps = overlaps.max(axis=1)  # match_overlaps

            # 3. Set up the match_inds
            match_inds = np.empty((overlaps.shape[0], ), dtype=np.int)
            match_inds.fill(-1)

            # 4. loop through each gt box, assigning the best match
            for inds in range(len(gt_boxes)):
                max_ind = np.argmax(overlaps)
                (ind_priorbox, ind_gt) = np.unravel_index(max_ind, overlaps.shape)

                match_inds[ind_priorbox] = ind_gt
                match_overlaps[ind_priorbox] = overlaps[ind_priorbox, ind_gt]

                # remove that match
                overlaps[:, ind_gt] = -1.0
                overlaps[ind_priorbox, :] = -1.0

            # 5. any prior box above the overlap threshold with any gt box
            inds_ = np.where((match_inds == -1) & (match_overlaps > self.overlap_threshold))[0]
            match_inds[inds_] = gt_assignment[inds_]

            # match_inds maps from prior box ind to GT box
            matches[batch] = {'inds': match_inds,
                              'overlap': match_overlaps}
        return matches

    def gen_locs(self, y_loc, num_preds_per_class):
        # y_loc should be (batch_size, num_preds_per_class, 4)
        assert self.share_location, 'non share location is not supported yet'
        assert y_loc.shape[1] == num_preds_per_class
        bboxes = {}
        for batch in range(self.be.bsz):
            bboxes[batch] = y_loc[batch, :, :].copy()
        return bboxes

    # convert gt_boxes to boxes per batch, removing
    # zero-padded entries
    def gen_gt_boxes(self, gt_boxes, gt_classes, num_gt):
        # gt_boxes is (max_gt_boxes, batch_size)
        # gt_classes is (max_gt_boxes, batch_size)
        # num_gt is (1, batch_size)  number of GT box counts
        bboxes = [np.zeros((ngt, 5)) for ngt in num_gt[0]]

        gt_boxes = gt_boxes.reshape((-1, 4, self.be.bsz))
        for batch in range(self.be.bsz):
            ngt = num_gt[0][batch]
            bboxes[batch][:, :4] = gt_boxes[:ngt, :, batch]
            bboxes[batch][:, 4] = gt_classes[:ngt, batch]

        return bboxes
