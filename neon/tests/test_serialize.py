# Copyright 2015 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for checking serialization


These tests will run a model in 2 ways:
    (1) run a model for N steps
    (2) run a model k times for N/k steps,
          serializing and deserializing the
          model at each of the k steps
and compare the serialized output files.

The goal is to check whether there are errors
introduced by serializing and deserializing the
model.  This is done by checking for consistency
between the two.
"""

from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest

import logging
import os
import numpy as np
from neon.backends import gen_backend
from neon.util.persist import deserialize


@attr('slow')
class TestSerialization:
    def setup(self):
        logging.basicConfig(level=40)  # ERROR or higher

        self.seed = 1234  # random seed

        self.back_end = 'nervanagpu'

        # check if this is running on Jenkins
        if 'JENKINS_HOME' in os.environ:
            # do not use ~/data with Jenkins
            # need to use the job/build sandbox
            assert 'WORKSPACE' in os.environ
            self.model_path = os.path.join(
                os.environ['WORKSPACE'],
                'data')
        else:
            self.model_path = os.path.join(os.getenv("HOME"), 'data')
        return

    @attr('cuda')
    def test_generator_serialization(self):
        # models to tests
        model_yamls = [
            'toy_serialize_check_base.yml'
            ]

        yaml_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'tests_yamls')

        model_yamls = [os.path.join(yaml_dir, x) for x in model_yamls]

        # this generates the 3 test cases for the 2 different models
        for model_file in model_yamls:
            # run everything on cpu
            yield self.compare_cpu_only, model_file

            # run everything on gpu
            yield self.compare_gpu_only, model_file

            # run model first on gpu then hand off
            # serialized model to cpu b.e. to complete
            yield self.compare_gpu_to_cpu, model_file

            # run model first half on cgpu then hand off
            # serialized model to gpu b.e. to complete
            yield self.compare_cpu_to_gpu, model_file

        return

    # ---------------------------------------------------
    @staticmethod
    def compare_check_points(n, k, onestep_file_root, kstep_file_root,
                             atol=0.0, rtol=0.0):
        # compare the checkpoints saved during the single step run
        # with the files saved during the k step run

        for ind in range(n/k, n, n/k):
            # single step file name
            file_parts = os.path.splitext(onestep_file_root)
            onestep_fn = file_parts[0] + '_cp%d' % ind + file_parts[1]

            # multistep step file name
            file_parts = os.path.splitext(kstep_file_root)
            kstep_fn = file_parts[0] + '%d' % ind + file_parts[1]

            assert TestSerialization.model_compare(
                onestep_fn,
                kstep_fn,
                atol=atol,
                rtol=rtol),\
                'checkpoint files not matching up [%s, %s]' \
                % (onestep_fn, kstep_fn)
        return True

    # ---------------------------------------------------
    def compare_cpu_only(self, config_file):
        # init the model with known seed
        # save init weights locally for check later
        # run model for N steps and serialize model
        # init a new model with same initial weights
        #     - confirm weights are the same
        # run second model for N/k steps k times
        #   serializing model at each N/k steps
        #   and starting next N/k steps by deserializing
        #   file saved at last step
        # compare the final state of the two models
        n = 10  # total number of training epochs
        k = 2   # split across k serial/deserial steps

        # config_file is the yaml that contains basic model params
        # will load experiment from here and alter some
        # parameters programatically

        # run N steps in 1 big step
        be = 'cpu'
        be_args = {'rng_seed': self.seed}
        (fstate_1step, init_w_1step) = self.run_experiment_in_steps(
            config_file,
            n,
            1,
            be,
            **be_args)

        # run N steps in k steps
        (fstate_kstep, init_w_kstep) = self.run_experiment_in_steps(
            config_file,
            n,
            k,
            be,
            **be_args)

        # comapre the initial weights
        assert self.check_init_weights(
            init_w_1step,
            init_w_kstep,
            rtol=0.0,
            atol=0.0),\
            'initial model weights were not the same'

        # compare the final model state for 1 step
        # of N epochs vs. k steps of N/k epochs
        assert self.model_compare(fstate_1step, fstate_kstep),\
            'final states of 1 vs %d step learning runs not matching' % k

        assert self.compare_check_points(
            n,
            k,
            fstate_1step,
            os.path.join(os.path.dirname(fstate_kstep), '%d_step_.prm' % k))
        return

    # ---------------------------------------------------
    @attr('cuda')  # can we do this test by test
    def compare_gpu_only(self, config_file):
        raise SkipTest("This test is currently broken...")
        # same as compare_cpu_only except all is run on the gpu
        n = 10
        k = 2
        atol = 0.0
        rtol = .01

        # run N steps in 1 step
        be = 'gpu'
        be_args = {'rng_seed': self.seed, be: self.back_end}
        (fstate_1step, init_w_1step) = self.run_experiment_in_steps(
            config_file,
            n,
            1,
            be,
            **be_args)

        # run N steps in k steps
        (fstate_kstep, init_w_kstep) = self.run_experiment_in_steps(
            config_file,
            n,
            k,
            be,
            **be_args)

        # comapre the initial weights
        assert self.check_init_weights(
            init_w_kstep,
            init_w_1step,
            rtol=rtol,
            atol=atol),\
            'initial model weights were not the same'

        # compare the final model state for 1 step
        # of N epochs vs. k steps of N/k epochs
        assert self.model_compare(
            fstate_1step,
            fstate_kstep,
            rtol=rtol,
            atol=atol),\
            'final states of 1 vs %d step learning runs not matching' % k
        return

    # ---------------------------------------------------
    @attr('cuda')  # can we do this test by test
    def compare_cpu_to_gpu(self, config_file):
        raise SkipTest("This test is currently broken...")
        # run whole test on cpu and run it 1/2 on cpu the 1/2 on gpu
        # compare the outputs
        n = 10  # for this tests make sure that N/k is not an integer
        atol = 0.0
        rtol = .01

        # run N steps in 2 steps and save the
        # checkpoint which should be at int(N/2)
        be = 'cpu'
        be_args = {'rng_seed': self.seed}
        (fstate_1step, init_w_1step) = self.run_experiment_in_steps(
            config_file,
            n,
            1,
            be,
            **be_args)

        # run N steps in 2 steps first goes on the CPU
        (fstate_half_step, init_w_kstep) = self.run_experiment_in_steps(
            config_file,
            n/2,
            1,
            be,
            **be_args)

        # use the prm file from the 1/2 run above to init
        # this run and finish the full N epochs with GPU
        be = 'gpu'
        be_args = {'rng_seed': self.seed, be: self.back_end}
        (fstate_kstep, temp_init_w) = self.run_experiment_in_steps(
            config_file,
            n,
            1,
            be,
            init_config=fstate_half_step,
            **be_args)

        # comapre the initial weights
        assert self.check_init_weights(
            init_w_kstep,
            init_w_1step,
            rtol=rtol,
            atol=atol),\
            'initial model weights were not the same'

        # compare the final model state for 1 step of N epochs
        # vs. k steps of N/k epochs
        assert self.model_compare(
            fstate_1step,
            fstate_kstep,
            rtol=rtol,
            atol=atol),\
            'cpu->gpu handoff fails'
        return

    # ---------------------------------------------------
    @attr('cuda')  # can we do this test by test
    def compare_gpu_to_cpu(self, config_file):
        raise SkipTest("This test is currently broken...")
        # run whole test on cpu and run it 1/2 on cpu the 1/2 on gpu
        # compare the outputs
        n = 10  # for this tests make sure that N/k is not an integer
        atol = 0.0
        rtol = .01

        # run N steps in 2 steps and save the
        # checkpoint which should be at int(N/2)
        be = 'cpu'
        be_args = {'rng_seed': self.seed}
        (fstate_1step, init_w_1step) = self.run_experiment_in_steps(
            config_file,
            n,
            1,
            be,
            **be_args)

        # run N steps in 2 steps
        # first goes on the CPU
        be = 'gpu'
        be_args = {'rng_seed': self.seed, be: self.back_end}
        (fstate_half_step, init_w_kstep) = self.run_experiment_in_steps(
            config_file,
            n/2,
            1,
            be,
            **be_args)

        # use the prm file from the N/2 run above
        # as init for this run and finish the full
        # N epochs with GPU
        be = 'cpu'
        be_args = {'rng_seed': self.seed}
        (fstate_kstep, temp_init_w) = self.run_experiment_in_steps(
            config_file,
            n,
            1,
            be,
            init_config=fstate_half_step,
            **be_args
            )

        # comapre the initial weights
        assert self.check_init_weights(init_w_kstep, init_w_1step,
                                       rtol=rtol, atol=atol)

        # compare the final model state for 1 step of N epochs
        # vs. k steps of N/k epochs
        assert self.model_compare(fstate_1step, fstate_kstep,
                                  rtol=rtol, atol=atol)
        return

    def run_experiment_in_steps(self, config_file, n, k, be,
                                init_config=None, **be_args):
        # run an experiment for N epochs in k stepe with n/k
        # epochs per step.  Last step will be enough epochs
        # to reach n total epochs.  Between each step the
        # model will be serialized and saved, then reloaded
        # from that saved file at the next step if init_config
        # is not None, then that file will be used #  as the
        # initial deserialize file -  this is used for handing off
        # models run on cpu.gpu backends to running on gpu/cpu backends

        stepsize = n/k
        last_saved_state = None

        for ind in range(1, k+1):
            # run the same learning with N/k epochs k times
            # each time saving and reloading the serialized model state
            if ind == k:
                # in case N/k is a fraction
                # last step will end at epoch N
                end_epoch = n
            else:
                end_epoch = ind*stepsize

            # load up base experiment config
            experiment = deserialize(config_file)

            # run for N/k steps
            experiment.model.num_epochs = end_epoch

            if ind > 1:
                # after step 1 need to load initial config
                # from last runs serialized file
                experiment.model.deserialized_path = last_saved_state
            elif init_config is not None:
                # used the given pickle file to initialize the mdoel
                experiment.model.deserialized_path = init_config

            # save the model to this file
            last_saved_state = os.path.join(self.model_path,
                                            '%d_step_%d.prm' % (k, end_epoch))
            print(last_saved_state)
            if os.path.exists(last_saved_state):
                print('removing %s' % last_saved_state)
                os.remove(last_saved_state)
            experiment.model.serialized_path = last_saved_state

            experiment.model.serialize_schedule = k
            if k == 1:
                # keep copies of all checkpoint files for cp tests
                experiment.model.save_checkpoints = n

            backend = gen_backend(model=experiment.model, **be_args)
            experiment.initialize(backend)

            if ind == 1:
                # save the initial weights for check with other runs
                intial_weights = {}
                for ind, layer in enumerate(experiment.model.layers):
                    if hasattr(layer, 'weights'):  # only checking weights
                        # ensure unique layer names
                        ln = '%s_%d' % (layer.name, ind)
                        intial_weights[ln] = np.copy(
                            layer.weights.asnumpyarray()
                            )
            experiment.run()

        return (last_saved_state, intial_weights)

    # load up two model pkl files and compare the data stored in them
    @staticmethod
    def model_compare(model1_file, model2_file, atol=0.0, rtol=0.0):
        model1 = deserialize(model1_file)
        model2 = deserialize(model2_file)

        assert model1.keys().sort() == model2.keys().sort()

        # remove the epochs from the dictionaries and compare them
        assert model1.pop('epochs_complete') == model2.pop('epochs_complete')

        # for MLP just layers should be left?
        print('checking the 1 versus k step outputs...')
        for ky in model1.keys():
            print(ky)
            assert TestSerialization.layer_compare(model1[ky],
                                                   model2[ky],
                                                   atol=atol,
                                                   rtol=rtol)
        print('OK')

        return True

    # given the dictionary for two layers obtained by
    # deserializing two different pickle files this will
    # compare whether the two layers have the same values.
    @staticmethod
    def layer_compare(layer1, layer2, atol=0.0, rtol=0.0):
        assert layer1.keys().sort() == layer2.keys().sort()
        for ky in layer1.keys():
            print(ky)
            assert isinstance(layer1[ky], type(layer2[ky]))
            assert TestSerialization.val_compare(layer1[ky], layer2[ky],
                                                 atol=atol, rtol=rtol)
        return True

    # compare two objects obtained from serialized files
    # does element-wise compare for numpy ndarray objects
    # and it called recursively on list object
    # abs and rel tolerances can be passed to the "allclose" call
    @staticmethod
    def val_compare(obj1, obj2, atol=0.0, rtol=0.0):
        assert type(obj1) == type(obj2), 'type mismatch'
        if isinstance(obj1, np.ndarray):
            if not np.allclose(obj1, obj2, atol=atol, rtol=rtol):
                maxdiff = np.amax(np.divide(np.absolute(obj1 - obj2),
                                            np.absolute(obj2)))
                assert False, 'layer weight mismatch max %f' % maxdiff
        if isinstance(obj1, list):
            assert len(obj1) == len(obj2), 'list length mismatch'
            for ind in range(len(obj1)):
                assert TestSerialization.val_compare(obj1[ind], obj2[ind],
                                                     atol=atol, rtol=rtol)
        return True

    # takes 2 layer weight dictionaries generated in run_experiment_in_steps
    # and compares the weights to within the rel and abs tol values
    # check the initial weights - should match the 1 step run
    @staticmethod
    def check_init_weights(layer1, layer2, rtol=0.0, atol=0.0):
        assert layer1.keys().sort() == layer2.keys().sort()

        for ky in layer1.keys():
            assert np.allclose(layer1[ky], layer2[ky], atol=atol, rtol=rtol), \
                'initial layer weight mismatch [%s]' % ky
        return True
