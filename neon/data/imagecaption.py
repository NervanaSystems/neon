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

from collections import Counter
import numpy as np
import os

from neon import logger as neon_logger
from neon.data.dataiterator import NervanaDataIterator
from neon.data.datasets import Dataset
from neon.util.compat import xrange


class ImageCaption(NervanaDataIterator):
    """
    This class loads in the sentences and CNN image features for image captioning
    that have been taken from Andrej Karpathy's
    `Deep Visual-Semantic Alignments <http://cs.stanford.edu/people/karpathy/deepimagesent/>`_
    work. They are converted to pkl format to avoid using scipy for loading the .mat
    image features.

    The dataset assumes the model takes the precomputed VGG features of an
    image and a sentence converted to its one hot representation. The model
    then transforms both the image and the sentence to the same space and
    prepends the image to the sentence so that it is treated as the first word
    of the sequence to be fed to a RNN.
    """

    end_token = '.'
    image_size = 4096  # Hard code VVG feature size

    def __init__(self, path, max_images=-1):
        """
        Load vocab and image features. Convert sentences to indices

        Args:
            path (str): Directory containing sentences and image features.
            max_images (int): Number of images to load. Set to -1 for max.
        """
        super(ImageCaption, self).__init__(name=None)

        self.path = path
        neon_logger.display('Reading train images and sentences from %s' % self.path)
        self.read_images('train')
        self.load_vocab()

        trainSents, trainImgs = [], []
        for i, img_sent in enumerate(self.iterImageSentencePair()):
            if len(trainSents) > max_images and max_images > 0:
                break
            trainImgs.append(img_sent['image'])
            sent = [self.end_token] + [x for x in img_sent['sentence']['tokens']
                                       if x in self.vocab_to_index]
            trainSents.append(sent[:self.max_sentence_length])

        self.nbatches = len(trainImgs) // self.be.bsz
        self.ndata = self.nbatches * self.be.bsz

        self.X = np.zeros((len(trainSents), self.max_sentence_length))
        self.y = np.zeros((len(trainSents), self.max_sentence_length + 1))
        self.images = np.vstack(trainImgs)

        self.sent_length = np.array([len(x) + 1 for x in trainSents])
        self.sent_ends = np.arange(self.max_sentence_length + 1)[:, np.newaxis]
        for sent_idx, sent in enumerate(trainSents):
            self.X[sent_idx, :len(sent)] = [self.vocab_to_index[word] for word in sent]
        self.y[:, :-1] = self.X

    def load_vocab(self):
        """
        Load vocab and initialize buffers
        Input sentence batch is of dimension (vocab_size, max_sentence_length * batch_size)
        where each column is the 1-hot representation of a word and the first batch_size columns
        are the first words of each sentence.
        """

        sentences = [sent['tokens'] for sent in self.iterSentences()]
        # Flatten list of list of words to one list of words
        words = [word for sentence in sentences for word in sentence]
        # Count words and keep words greater than threshold
        word_counts = Counter(words)

        vocab = [self.end_token] + \
                [word for word in list(word_counts.keys()) if word_counts[word] >= 5]
        self.vocab_size = len(vocab)
        self.vocab_to_index = dict((c, i) for i, c in enumerate(vocab))
        self.index_to_vocab = dict((i, c) for i, c in enumerate(vocab))

        # Compute optional bias vector for initializing final linear layer bias
        word_counts[self.end_token] = len(sentences)
        self.bias_init = np.array([1.0 * word_counts[self.index_to_vocab[i]]
                                   for i in self.index_to_vocab]).reshape((self.vocab_size, 1))
        self.bias_init /= np.sum(self.bias_init)
        self.bias_init = np.log(self.bias_init)
        self.bias_init -= np.max(self.bias_init)

        self.max_sentence_length = max(len(sent) for sent in sentences) + 1

        self.dev_image = self.be.iobuf(self.image_size)
        self.dev_imageT = self.be.empty(self.dev_image.shape[::-1])
        self.dev_X = self.be.iobuf((self.vocab_size, self.max_sentence_length))
        self.dev_y = self.be.iobuf((self.vocab_size, self.max_sentence_length + 1))
        # Create mask to deal with variable length sentences
        self.dev_y_mask = self.be.iobuf((self.vocab_size, self.max_sentence_length + 1))
        self.y_mask = np.zeros(self.dev_y_mask.shape,
                               dtype=np.uint8).reshape(self.vocab_size,
                                                       self.max_sentence_length + 1, -1)
        self.y_mask_reshape = self.y_mask.reshape(self.dev_y_mask.shape)

        self.dev_lbl = self.be.iobuf(self.max_sentence_length, dtype=np.int32)
        self.dev_lblT = self.be.empty(self.dev_lbl.shape[::-1])
        self.dev_lblflat = self.dev_lbl.reshape((1, self.dev_lbl.size))

        self.dev_y_lbl = self.be.iobuf(self.max_sentence_length + 1, dtype=np.int32)
        self.dev_y_lblT = self.be.empty(self.dev_y_lbl.shape[::-1])
        self.dev_y_lblflat = self.dev_y_lbl.reshape((1, self.dev_y_lbl.size))

        self.shape = [self.image_size, (self.vocab_size, self.max_sentence_length)]
        neon_logger.display("Vocab size: %d, Max sentence length: %d" % (self.vocab_size,
                                                                         self.max_sentence_length))

    def read_images(self, split):
        """
        Read sentences and image features from pickled dict

        Args:
            split (str): test or train split
        """
        data_path = os.path.join(self.path, 'features.pkl.gz')
        from neon.util.persist import load_obj
        self.dataset = load_obj(data_path)
        self.sent_data = self.dataset['sents'][split]
        self.features = self.dataset['feats']

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuples, tuples, first tuple contains image features and one hot input sentence
                            second tuple contains one hot target sentence and mask
                            corresponding to 1's up to where each sentence ends and
                            zeros elsewhere after.
        """

        shuf_idx = self.be.rng.permutation(len(self.X))
        self.X, self.y, self.images = (self.X[shuf_idx], self.y[shuf_idx], self.images[shuf_idx])
        self.sent_length = self.sent_length[shuf_idx]

        for batch_idx in xrange(self.nbatches):

            start = batch_idx * self.be.bsz
            end = (batch_idx + 1) * self.be.bsz

            # image_batch = self.images[start:end].T.astype(np.float32, order='C')
            self.dev_imageT.set(self.images[start:end])
            self.dev_image[:] = self.dev_imageT.T

            # X_batch = self.X[start:end].T.astype(np.float32, order='C')
            self.dev_lblT.set(self.X[start:end])
            self.dev_lbl[:] = self.dev_lblT.T
            self.dev_X[:] = self.be.onehot(self.dev_lblflat, axis=0)

            self.y_mask[:] = 1
            sent_lens = self.sent_length[start:end]
            self.y_mask[:, self.sent_ends > sent_lens[np.newaxis, :]] = 0
            self.dev_y_mask[:] = self.y_mask_reshape

            # y_batch = self.y[start:end].T.astype(np.float32, order='C')
            self.dev_y_lblT.set(self.y[start:end])
            self.dev_y_lbl[:] = self.dev_y_lblT.T
            self.dev_y[:] = self.be.onehot(self.dev_y_lblflat, axis=0)
            self.dev_y[:] = self.dev_y * self.dev_y_mask

            yield (self.dev_image, self.dev_X), (self.dev_y, self.dev_y_mask)

    def prob_to_word(self, prob):
        """
        Convert 1 hot probabilities to sentences.

        Args:
            prob (Tensor): Word probabilities of each sentence of batch.
                           Of size (vocab_size, batch_size * (max_sentence_length+1))

        Returns:
            list containing sentences
        """

        sents = []

        if not isinstance(prob, np.ndarray):
            prob = prob.get()
        words = [self.index_to_vocab[x] for x in np.argmax(prob, axis=0).tolist()]

        for sent_index in xrange(self.be.bsz):
            sent = []
            for i in xrange(self.max_sentence_length):
                word = words[self.be.bsz * i + sent_index]
                sent.append(word)
                if (i > 0 and word == self.end_token) or i >= 20:
                    break
            sents.append(" ".join(sent))

        return sents

    def predict(self, model):
        """
        Given a model, generate sentences from this dataset.

        Args:
            model (Model): Image captioning model.

        Returns:
            list, list containing predicted sentences and target sentences
        """
        sents = []
        targets = []
        y = self.be.zeros(self.dev_X.shape)
        for mb_idx, (x, t) in enumerate(self):
            y.fill(0)
            # Repeatedly generate next word in sentence and choose max prob word each time.
            for step in range(1, self.max_sentence_length + 1):
                prob = model.fprop((x[0], y), inference=True).get()[:, :-self.be.bsz].copy()
                pred = np.argmax(prob, axis=0)
                prob.fill(0)
                for i in range(step * self.be.bsz):
                    prob[pred[i], i] = 1
                y[:] = prob
            sents += self.prob_to_word(y)
            # Test set, keep list of targets
            if isinstance(self, ImageCaptionTest):
                targets += t[0]
            # Train set, only 1 target
            else:
                targets.append(t[0])

        return sents, targets

    def bleu_score(self, sents, targets):
        """
        Compute the BLEU score from a list of predicted sentences and reference sentences

        Args:
            sents (list): list of predicted sentences
            targets (list): list of reference sentences where each element is a list of
                            multiple references.
        """

        num_ref = len(targets[0])
        output_file = self.path + '/output'
        reference_files = [self.path + '/reference%d' % i for i in range(num_ref)]
        bleu_script_url = 'https://raw.githubusercontent.com/karpathy/neuraltalk/master/eval/'
        bleu_script = 'multi-bleu.perl'

        neon_logger.display("Writing output and reference sents to dir %s" % self.path)

        output_f = open(output_file, 'w+')
        for sent in sents:
            sent = sent.strip(self.end_token).split()
            output_f.write(" ".join(sent) + '\n')

        reference_f = [open(f, 'w') for f in reference_files]
        for i in range(num_ref):
            for target_sents in targets:
                reference_f[i].write(target_sents[i] + '\n')

        output_f.close()
        [x.close() for x in reference_f]

        owd = os.getcwd()
        os.chdir(self.path)
        if not os.path.exists(bleu_script):
            Dataset.fetch_dataset(bleu_script_url, bleu_script, bleu_script, 6e6)
        bleu_command = 'perl multi-bleu.perl reference < output'
        neon_logger.display("Executing bleu eval script: {}".format(bleu_command))
        os.system(bleu_command)
        os.chdir(owd)

    def _getImage(self, img):
        """
        Get image feature

        Arguments:
            img:

        Returns:

        """
        return self.features[:, img['imgid']]

    def iterSentences(self):
        """Iterate over all sentences"""
        for img in self.sent_data:
            for sent in img['sentences']:
                yield sent

    def iterImageSentencePair(self):
        """Iterate over all image sentence pairs where an image may be repeated"""
        for i, img in enumerate(self.sent_data):
            for sent in img['sentences']:
                out = {}
                out['image'] = self._getImage(img)
                out['sentence'] = sent
                yield out

    def iterImageSentenceGroup(self):
        """Iterate over all image sentence groups"""
        for i, img in enumerate(self.sent_data):
            out = {}
            out['image'] = self._getImage(img)
            out['sentences'] = img['sentences']
            yield out


class ImageCaptionTest(ImageCaption):
    """
    This class loads in image and sentence features for testing.
    """

    def __init__(self, path):
        self.path = path
        neon_logger.display('Reading test images and sentences from %s' % self.path)
        # Load vocab using training set and then load test set
        self.read_images('train')
        self.load_vocab()
        self.read_images('test')

        trainIter = self.iterImageSentenceGroup()
        trainSents, trainImgs = [], []
        for i, img_sent in enumerate(trainIter):
            trainImgs.append(img_sent['image'])
            trainSents.append([' '.join(sent['tokens']) for sent in img_sent['sentences']])

        self.nbatches = len(trainImgs) // self.be.bsz
        self.ndata = self.nbatches * self.be.bsz

        self.images = np.vstack(trainImgs)

        self.ref_sents = trainSents

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple, tuple: first tuple contains image features and empty input Tensor
                          second tuple contains list of reference sentences and
                          placeholder for mask.
        """
        for batch_idx in xrange(self.nbatches):

            start = batch_idx * self.be.bsz
            end = (batch_idx + 1) * self.be.bsz

            image_batch = self.images[start:end, :].T.astype(np.float32, order='C')
            self.dev_image.set(image_batch)

            yield (self.dev_image, self.dev_X), (self.ref_sents[start:end], None)


class Flickr8k(Dataset):
    """
    Flickr8k data set from http://cs.stanford.edu/people/karpathy/cvpr2015.pdf
    """
    def __init__(self, path='.', max_images=-1):
        url = 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption'
        super(Flickr8k, self).__init__('flickr8k.zip',
                                       url,
                                       49165563,
                                       path=path)
        self.max_images = max_images

    def gen_iterators(self):
        data_path = self.load_data()
        self._data_dict = {'train': ImageCaption(path=data_path, max_images=self.max_images)}
        self._data_dict['test'] = ImageCaptionTest(path=data_path)
        return self._data_dict

    def load_data(self):
        return self.load_zip(self.filename, self.size)


class Flickr30k(Dataset):
    """
    Flickr30k data set from http://cs.stanford.edu/people/karpathy/cvpr2015.pdf
    """
    def __init__(self, path='.', max_images=-1):
        url = 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption'
        super(Flickr30k, self).__init__('flickr30k.zip',
                                        url,
                                        49165563,
                                        path=path)
        self.max_images = max_images

    def gen_iterators(self):
        data_path = self.load_data()
        self._data_dict = {'train': ImageCaption(path=data_path, max_images=self.max_images)}
        self._data_dict['test'] = ImageCaptionTest(path=data_path)
        return self._data_dict

    def load_data(self):
        return self.load_zip(self.filename, self.size)


class Coco(Dataset):
    """
    MSCOCO data set from http://cs.stanford.edu/people/karpathy/cvpr2015.pdf
    """
    def __init__(self, path='.', max_images=-1):
        url = 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption'
        super(Coco, self).__init__('coco.zip',
                                   url,
                                   738051031,
                                   path=path)
        self.max_images = max_images

    def load_data(self):
        return self.load_zip(self.filename, self.size)

    def gen_iterators(self):
        data_path = self.load_data()
        self._data_dict = {'train': ImageCaption(path=data_path, max_images=self.max_images)}
        self._data_dict['test'] = ImageCaptionTest(path=data_path)
        return self._data_dict
