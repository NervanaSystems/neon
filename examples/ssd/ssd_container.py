from neon.initializers import Constant, Xavier
from neon.transforms import Rectlin
from neon.layers import Conv, Pooling, BranchNode
from layer import Normalize, PriorBox, DetectionOutput, ConcatTranspose
import numpy as np
from neon.util.persist import load_obj
from neon.layers.container import Tree
from neon.layers.layer import Layer
from collections import OrderedDict
import inspect
import re
from neon.data.datasets import Dataset
import os

# configuration for all the outputs layers
# key is the name of the layer that branches
# the order of the keys is important to model layout
# also see ingest_config below for the default values for some of the fields
# must maintain the layer order here
default_ssd_config = OrderedDict([('conv4_3', {
    'min_sizes': 30.0,
    'max_sizes': 60.0,
    'aspect_ratios': 2.0,
    'step': 8,
    'normalize': True
}), ('fc7', {
    'min_sizes': 60.0,
    'max_sizes': 111.0,
    'aspect_ratios': (2.0, 3.0),
    'step': 16
}), ('conv6_2', {
    'min_sizes': 111.0,
    'max_sizes': 162.0,
    'aspect_ratios': (2.0, 3.0),
    'step': 32
}), ('conv7_2', {
    'min_sizes': 162.0,
    'max_sizes': 213.0,
    'aspect_ratios': (2.0, 3.0),
    'step': 64
}), ('conv8_2', {
    'min_sizes': 213.0,
    'max_sizes': 264.0,
    'aspect_ratios': 2.0,
    'step': 100
}), ('conv9_2', {
    'min_sizes': 264.0,
    'max_sizes': 315.0,
    'aspect_ratios': 2.0,
    'step': 300
})])


class SSD(Tree):
    """
    SSD model is like a Tree, except with additional handling for the output layer, and
    for the prior boxes
    """
    def __init__(self, dataset, ssd_config=None, name='SSD'):
        if ssd_config is None:
            ssd_config = default_ssd_config

        # clean up the layer config, set types and add defaults
        self.ssd_config = self.ingest_config(ssd_config)
        (channels, self.img_h, self.img_w) = dataset.shape

        self.num_classes = dataset.num_classes

        # below method generates the base of the model (VGG + confidence and localization leafs)
        # self.layers - base model (Tree container)
        # self.leafs - list of references to the output leafs of the model
        # self.conv_layers - list of referneces to the conv layers that are
        #                     branch points for the leafs
        (layers, output_config) = self.generate_layers()
        self.output_config = output_config

        # now init the Tree
        super(SSD, self).__init__(layers=layers)
        self.altered_tensors = []

        # self.prior_boxes = [pl['mbox_prior']['layer'] for _, pl in self.output_config.items()]

        # generate the concat layers
        self.concat_loc = ConcatTranspose(name='concat_loc')
        self.concat_conf = ConcatTranspose(name='concat_conf')

        # generate the output layer (used for inference)
        self.output_layer = DetectionOutput(num_classes=self.num_classes)

        self.all_prior_boxes = None  # placeholder for computed prior boxes
        self.all_prior_boxes_dev = None  # placeholder for computed prior boxes

    def get_description(self, get_weights=False, keep_states=False):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only weights are serialized.

        Arguments:
            get_weights (bool, optional): Control whether all parameters are returned or
                                          just weights for serialization.
            keep_states (bool, optional): Control whether all parameters are returned
                                          or just weights for serialization.
        """
        desc = Layer.get_description(self, skip=['layers', 'dataloader', 'dataset'])
        desc['container'] = True
        desc['config']['layers'] = []
        for layer in self.layers:
            desc['config']['layers'].append(layer.get_description(get_weights=get_weights,
                                                                  keep_states=keep_states))
        self._desc = desc
        return desc

    def ingest_config(self, configd):
        # pull in the configuration dict and set the the appropriate instance attributes

        # first some type checking
        for ky in configd:
            # put in default values if not set externally
            configd[ky].setdefault('variance', (0.1, 0.1, 0.2, 0.2))
            configd[ky].setdefault('flip', True)
            configd[ky].setdefault('offset', 0.5)
            configd[ky].setdefault('normalize', False)
            configd[ky].setdefault('clip', False)

            # make sure required calues are set
            for cky in ('min_sizes', 'max_sizes', 'aspect_ratios', 'step'):
                assert cky in configd[ky], '%s must be set manually' % cky

            # convert any not tuples for these fields to tuples
            for ff in ('min_sizes', 'max_sizes', 'aspect_ratios'):
                val = configd[ky][ff]
                if type(val) not in (list, tuple):
                    configd[ky][ff] = (configd[ky][ff],)
                else:
                    configd[ky][ff] = tuple(configd[ky][ff])
        return configd

    def configure(self, in_obj):
        # configure the base model
        super(SSD, self).configure(in_obj)

        self.leafs = self.unnest(self.get_terminal())

        self.prior_boxes = []
        for _, config in self.output_config.items():
            conv_layer = config['conv_layer']
            mbox_prior = config['layers']['mbox_prior']
            mbox_prior.configure([in_obj, conv_layer])
            self.prior_boxes.append(mbox_prior)

        self.concat_loc.configure(self.leafs['mbox_loc'])
        self.concat_conf.configure(self.leafs['mbox_conf'])

        # configure the output layer
        self.prior_boxes = [config['layers']['mbox_prior']
                            for _, config in self.output_config.items()]

        self.output_layer.configure((self.leafs, self.prior_boxes))

    def unnest(self, x):
        outputs = dict()
        for name in ('mbox_loc', 'mbox_conf'):
            output = []
            for layer in self.output_config:
                index = self.output_config[layer]['index'][name]
                if len(index) > 1:
                    output.append(x[index[0]][index[1]])
                else:
                    output.append(x[index[0]])
            outputs[name] = output

        return outputs

    def nest(self, x):
        # x is a tuple of (loc, conf) inputs
        # create a nested list [A, [B, C], D, ...]

        # first create an empty list, then populate it
        list_length = len(self.output_config)*2 - 1
        outputs = [None] * list_length

        for name in ('mbox_loc', 'mbox_conf'):
            for inputs, layer in zip(x[name], self.output_config):
                index = self.output_config[layer]['index'][name]
                if len(index) > 1:
                    if outputs[index[0]] is None:
                        outputs[index[0]] = [None, None]
                    outputs[index[0]][index[1]] = inputs
                else:
                    outputs[index[0]] = inputs

        return outputs

    def allocate(self, shared_outputs=None):
        super(SSD, self).allocate(shared_outputs)

        for prior_box in self.prior_boxes:
            prior_box.allocate()

        self.concat_conf.allocate()
        self.concat_loc.allocate()
        self.output_layer.allocate()

    def generate_layers(self):
        conv_params = {'strides': 1,
                       'padding': 1,
                       'init': Xavier(local=True),
                       'bias': Constant(0),
                       'activation': Rectlin()}

        params = {'init': Xavier(local=True),
                  'bias': Constant(0),
                  'activation': Rectlin()}

        # Set up the model layers
        trunk_layers = []

        # set up 3x3 conv stacks with different feature map sizes
        # use same names as Caffe model for comparison purposes
        trunk_layers.append(Conv((3, 3, 64), name='conv1_1', **conv_params))   # conv1_1
        trunk_layers.append(Conv((3, 3, 64), name='conv1_2', **conv_params))
        trunk_layers.append(Pooling(2, strides=2))
        trunk_layers.append(Conv((3, 3, 128), name='conv2_1', **conv_params))  # conv2_1
        trunk_layers.append(Conv((3, 3, 128), name='conv2_2', **conv_params))
        trunk_layers.append(Pooling(2, strides=2))
        trunk_layers.append(Conv((3, 3, 256), name='conv3_1', **conv_params))  # conv3_1
        trunk_layers.append(Conv((3, 3, 256), name='conv3_2', **conv_params))
        trunk_layers.append(Conv((3, 3, 256), name='conv3_3', **conv_params))
        trunk_layers.append(Pooling(2, strides=2))
        trunk_layers.append(Conv((3, 3, 512), name='conv4_1', **conv_params))  # conv4_1
        trunk_layers.append(Conv((3, 3, 512), name='conv4_2', **conv_params))
        trunk_layers.append(Conv((3, 3, 512), name='conv4_3', **conv_params))

        trunk_layers.append(Pooling(2, strides=2))
        trunk_layers.append(Conv((3, 3, 512), name='conv5_1', **conv_params))  # conv5_1
        trunk_layers.append(Conv((3, 3, 512), name='conv5_2', **conv_params))
        trunk_layers.append(Conv((3, 3, 512), name='conv5_3', **conv_params))
        trunk_layers.append(Pooling(3, strides=1, padding=1))
        trunk_layers.append(Conv((3, 3, 1024), dilation=6, padding=6, name='fc6', **params))  # fc6
        trunk_layers.append(Conv((1, 1, 1024), dilation=1, padding=0, name='fc7', **params))  # fc7

        trunk_layers.append(Conv((1, 1, 256), strides=1, padding=0, name='conv6_1', **params))
        trunk_layers.append(Conv((3, 3, 512), strides=2, padding=1, name='conv6_2', **params))

        trunk_layers.append(Conv((1, 1, 128), strides=1, padding=0, name='conv7_1', **params))
        trunk_layers.append(Conv((3, 3, 256), strides=2, padding=1, name='conv7_2', **params))

        # append conv8, conv9, conv10, etc. (if needed)
        matches = [re.search('conv(\d+)_2', key) for key in self.ssd_config]
        layer_nums = [int(m.group(1)) if m is not None else -1 for m in matches]
        max_layer_num = np.max(layer_nums)

        if max_layer_num is not None:
            for layer_num in range(8, max_layer_num+1):
                trunk_layers.append(Conv((1, 1, 128), strides=1, padding=0,
                                    name='conv{}_1'.format(layer_num), **params))
                trunk_layers.append(Conv((3, 3, 256), strides=1, padding=0,
                                    name='conv{}_2'.format(layer_num), **params))

        layers = []
        output_config = OrderedDict()
        mbox_index = 1

        for layer in self.ssd_config:

            index = self.find_insertion_index(trunk_layers, layer)
            conv_layer = self.get_conv_layer(trunk_layers, index)

            branch_node = BranchNode(name=layer + '_branch')
            trunk_layers.insert(index, branch_node)

            leafs = self.generate_leafs(layer)
            is_terminal = layer == 'conv{}_2'.format(max_layer_num)

            # append leafs to layers
            # mbox_loc_index and mbox_conf_index map to locations
            # in the output list of the model.

            if self.ssd_config[layer]['normalize']:
                branch = self.create_normalize_branch(leafs, branch_node, layer)
                layers.append(branch)
                mbox_loc_index = (mbox_index, 0)
                mbox_conf_index = (mbox_index, 1)
                mbox_index += 1

            else:
                if is_terminal:
                    trunk_layers.append(leafs['mbox_loc'])
                    mbox_loc_index = (0, )
                else:
                    layers.append([branch_node, leafs['mbox_loc']])
                    mbox_loc_index = (mbox_index, )
                    mbox_index += 1

                layers.append([branch_node, leafs['mbox_conf']])
                mbox_conf_index = (mbox_index, )
                mbox_index += 1

            output_config[layer] = {'layers': leafs,
                                    'conv_layer': conv_layer,
                                    'index': {'mbox_conf': mbox_conf_index,
                                              'mbox_loc': mbox_loc_index}}

        layers.insert(0, trunk_layers)

        return layers, output_config

    def get_conv_layer(self, trunk_layers, index):
        return trunk_layers[index - 1][0]

    def find_insertion_index(self, trunk_layers, layer):
        """
        Given a layer name, find the insertion point in trunk_layers
        """
        trunk_names = [l[0].name if isinstance(l, list) else l.name for l in trunk_layers]
        if layer not in trunk_names:
            raise ValueError('{} from ssd_config not found in trunk layers'.format(layer))
        else:
            return trunk_names.index(layer) + 1

    def generate_leafs(self, layer):
        """
        Given a key to the ssd_config, generate the leafs
        """
        config = self.ssd_config[layer]

        leaf_params = {'strides': 1,
                       'padding': 1,
                       'init': Xavier(local=True),
                       'bias': Constant(0)}

        # to match caffe layer's naming
        if config['normalize']:
            layer += '_norm'

        priorbox_args = self.get_priorbox_args(config)
        mbox_prior = PriorBox(**priorbox_args)
        num_priors = mbox_prior.num_priors_per_pixel

        loc_name = layer + '_mbox_loc'
        mbox_loc = Conv((3, 3, 4*num_priors), name=loc_name, **leaf_params)

        conf_name = layer + '_mbox_conf'
        mbox_conf = Conv((3, 3, self.num_classes*num_priors), name=conf_name, **leaf_params)

        return {'mbox_prior': mbox_prior, 'mbox_loc': mbox_loc, 'mbox_conf': mbox_conf}

    def get_priorbox_args(self, config):
        allowed_args = inspect.getargspec(PriorBox.__init__).args
        args = list(set(allowed_args) & set(config.keys()))

        priorbox_args = {key: config[key] for key in args}
        priorbox_args['img_shape'] = (self.img_w, self.img_h)
        return priorbox_args

    def create_normalize_branch(self, leafs, branch_node, layer):
        """
        Append leafs to trunk_layers at the branch_node. If normalize, add a Normalize layer.
        """

        tree_branch = BranchNode(name=layer + '_norm_branch')
        branch1 = [Normalize(init=Constant(20.0), name=layer + '_norm'),
                   tree_branch, leafs['mbox_loc']]

        branch2 = [tree_branch, leafs['mbox_conf']]
        new_tree = Tree([branch1, branch2])

        return [branch_node, new_tree]

    def distribute_tensors(self, x, parallelism='Disabled'):
        for keys in x.keys():
            for tensor in x[keys]:
                altered_tensor = self.be.distribute_data(tensor, parallelism)
                if altered_tensor is not None:  # if altered, track it so we can revert later.
                    self.altered_tensors.append(altered_tensor)

    def revert_tensors(self, tensor_list):
        for tensor in tensor_list:
            self.be.revert_tensor(tensor)

        # reset list of altered tensors
        self.altered_tensors = []

    def fprop(self, inputs, inference=False, beta=0.0):

        self._prior_box_fprop()

        # fprop through the model base
        x = super(SSD, self).fprop(inputs, inference=inference)

        for tensor in x:
            self.be.convert_data(tensor, False)

        x = self.unnest(x)  # reorder x

        # for mgpu, convert to singlenode tensor
        self.distribute_tensors(x, parallelism='Disabled')

        x = (self.concat_loc.fprop(x['mbox_loc']), self.concat_conf.fprop(x['mbox_conf']))

        # TODO: inference and no-inference return different outputs, can we normalize this somehow?
        if inference:
            outputs = self.output_layer.fprop((x, self.all_prior_boxes_dev))
            self.revert_tensors(self.altered_tensors)

            return outputs
        else:
            return (x[0], x[1], self.all_prior_boxes)

    def _prior_box_fprop(self):
        # fprop and stack all the prior boxes

        if self.all_prior_boxes is None:
            priors = [prior_box.fprop(None).get() for prior_box in self.prior_boxes]

            self.all_prior_boxes = np.vstack(priors)
            self.all_prior_boxes_dev = self.be.array(self.all_prior_boxes)

    def bprop(self, error, alpha=1.0, beta=0.0):
        # error will be a tuple with (loc_deltas, conf_deltas) from the
        # the mulitbox loss layer
        # first unconcatenate and unravel the deltas for each layer

        (loc_err, conf_err) = error

        loc_err = self.concat_loc.bprop(loc_err)
        conf_err = self.concat_conf.bprop(conf_err)

        errors = {'mbox_conf': conf_err, 'mbox_loc': loc_err}

        # for mgpu, fragment the tensors
        self.distribute_tensors(errors, parallelism='Data')

        errors = self.nest(errors)

        # errors can go into Tree bprop now
        errors = super(SSD, self).bprop(errors)

        self.revert_tensors(self.altered_tensors)
        return errors


def load_weights(target_layers, source):
    for target in target_layers:
        if hasattr(target, 'W'):
            target.load_weights(source[target.name], load_states=True)
            print(target.name)
        else:
            print("SKIPPING: {}".format(target.name))


def load_caffe_weights(model, file_path):
    pdict = load_obj(file_path)['params']

    #  we match by name with the caffe blobs
    for (pos, layer) in enumerate(model.layers.layers):
        if pos == 1:  # skip conv4_3
            continue
        load_weights(layer.layers, pdict)

    # we handle the tree-in-tree next
    conv4_3_loc = model.layers.layers[1].layers[1].layers[0].layers
    conv4_3_conf = model.layers.layers[1].layers[1].layers[1].layers
    load_weights(conv4_3_loc, pdict)
    load_weights(conv4_3_conf, pdict)


def load_vgg_weights(model, path):
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/'
    filename = 'VGG_ILSVRC_16_layers_fc_reduced_fused_conv_bias.p'
    size = 86046032

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print('De-serializing the pre-trained VGG16 model with dilated convolutions...')
    pdict = load_obj(filepath)

    model_layers = [l for l in model.layers.layers[0].layers]
    # convert source model into dictionary with layer name as keys
    src_layers = {layer['config']['name']: layer for layer in pdict['model']['config']['layers']}

    i = 0
    for layer in model_layers:
        if layer.classnm == 'Convolution_bias' and i < 15:
            # no states in above parameter file
            layer.load_weights(src_layers['Convolution_bias_'+str(i)], load_states=False)
            print('{} loaded from source file'.format(layer.name))
            i += 1
        elif hasattr(layer, 'W'):
            print('Skipping {} layer'.format(layer.name))
