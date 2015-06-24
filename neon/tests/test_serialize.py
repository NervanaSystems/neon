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
from nose.tools import nottest

import logging
import os
import sys
import yaml
import numpy as np
from neon.backends import gen_backend
from neon.util.persist import deserialize

import nervanagpu
import cudanet

#@attr('cuda')  # can we do this test by test
class TestSerialization:
    def setup(self):
        logging.basicConfig(level=30)  # ERROR or higher

        self.seed=1234 # random seed

        # was using this to make it easy to check 
        # cudanet but for real test this should be
        # nervanagpu
        self.back_end = 'nervanagpu'


        # check if this is running on Jenkins
        if os.environ.has_key('JENKINS_HOME'):
            # do not use ~/data with Jenkins
            # need to use the job/build sandbox
            assert os.environ.has_key('WORKSPACE')
            self.model_path = os.path.join( os.environ['WORKSPACE'],\
                    'data')
        else:
            self.model_path = os.path.join( os.getenv("HOME"), 'data') #~/data/
        return

    #---------------------------------------------------
    # it appears that nose is ignoring the attr on the yielded functions
    # so this generator has to be set to gpu even though there is 1 cpu only test
    # would prefer to tag each yieled tests seperately
    @attr('cuda')  
    def test_generator(self):
        # models to tests
        ModelYamls = ['toy_serialize_check_base.yml' , \
                            'i1k_serialize_check_base.yml']

        yaml_dir = os.path.join(\
                os.path.dirname(os.path.realpath(__file__)),\
                'tests_yamls')

        ModelYamls = [ os.path.join(yaml_dir, x) for x in ModelYamls ]

        # this generates the 3 test cases for the 2 different models
        for model_file in ModelYamls:
            # run everything on cpu
            yield self.compare_cpuOnly,    model_file

            # run everything on gpu
            yield self.compare_gpuOnly,    model_file

            # run model first on gpu then hand off 
            # serilized model to cpu b.e. to complete
            yield self.compare_gpu_to_cpu, model_file

            # run model first half on cgpu then hand off 
            # serilized model to gpu b.e. to complete
            yield self.compare_cpu_to_gpu, model_file

        return

    #---------------------------------------------------
    def compare_cpuOnly(self,config_file):
        # init the toy model with known seed
        # save init weights locally for check later
        # run model for N steps and serialize model
        # init a new model with same initial weights
        #     - confirm weights are the same
        # run second model for N/k steps k times
        #   serializing model at each N/k steps
        #   and starting next N/k steps by deserilizing
        #   file saved at last step
        # compare the final state of the two models 

        N = 10 # total number of training epochs
        k = 2   # split across k serial/deserial steps

        # config_file is the yaml that contains basic model params
        # will load experiment from here and alter some
        # parameters programatically

        # run N steps in 1 shot 
        be='cpu'
        be_args = {'rng_seed': self.seed}
        (fstate_1shot, init_w_1shot ) = self.run_experiment_in_steps(config_file, N, 1, be, **be_args)

        # run N steps in k shots
        (fstate_kshot, init_w_kshot ) = self.run_experiment_in_steps(config_file, N, k, be, **be_args)

        # comapre the initial weights
        assert self.check_init_weights( init_w_1shot, init_w_kshot , rtol=0.0, atol=0.0 ),\
                'initial model weights were not the same'

        #compare the final model state for 1 step of N epochs vs. k steps of N/k epochs
        assert self.model_compare( fstate_1shot, fstate_kshot ),\
                'final states of 1 vs %d step learning runs not matching' %k
        return

    #---------------------------------------------------
    @attr('cuda')  # can we do this test by test
    def compare_gpuOnly(self,config_file):
        # same as test_toyModel_compare_cpuOnly except all is run on the gpu
        N = 10
        k = 5
        atol=0.0
        rtol=.01

        # run N steps in 1 shot 
        be='gpu'
        be_args = {'rng_seed': self.seed, be : self.back_end }
        (fstate_1shot, init_w_1shot ) = self.run_experiment_in_steps(config_file, N, 1, be, **be_args)

        # run N steps in k shots
        (fstate_kshot, init_w_kshot ) = self.run_experiment_in_steps(config_file, N, k, be, **be_args)

        # comapre the initial weights
        assert self.check_init_weights( init_w_kshot, init_w_1shot , rtol=rtol, atol=atol ),\
                'initial model weights were not the same'

        #compare the final model state for 1 step of N epochs vs. k steps of N/k epochs
        assert self.model_compare( fstate_1shot, fstate_kshot ,rtol=rtol, atol=atol),\
                'final states of 1 vs %d step learning runs not matching' %k
        return

    #---------------------------------------------------
    @attr('cuda')  # can we do this test by test
    def compare_gpuOnly(self, config_file):
        # same as test_toyModel_compare_cpuOnly except all is run on the gpu
        N = 10
        k = 2
        atol=0.0
        rtol=.01

        # run N steps in 1 shot 
        be='gpu'
        be_args = {'rng_seed': self.seed, be : self.back_end }
        (fstate_1shot, init_w_1shot ) = self.run_experiment_in_steps(config_file, N, 1, be, **be_args)

        # run N steps in k shots
        (fstate_kshot, init_w_kshot ) = self.run_experiment_in_steps(config_file, N, k, be, **be_args)


        # comapre the initial weights
        assert self.check_init_weights( init_w_kshot, init_w_1shot , rtol=rtol, atol=atol ),\
                'initial model weights were not the same'

        #compare the final model state for 1 step of N epochs vs. k steps of N/k epochs
        assert self.model_compare( fstate_1shot, fstate_kshot ,rtol=rtol, atol=atol),\
                'final states of 1 vs %d step learning runs not matching' %k
        return


    #---------------------------------------------------
    @attr('cuda')  # can we do this test by test
    def compare_cpu_to_gpu(self, config_file):
        # run whole test on cpu and run it 1/2 on cpu the 1/2 on gpu
        # compare the outputs
        N = 10 # for this tests make sure that N/k is not an integer
        k = 5  # k should be 2 here
        atol=0.0
        rtol=.01

        # run N steps in 2 shots and save the checkpoint which should be at int(N/2)
        be='cpu'
        be_args = {'rng_seed': self.seed}#, be : self.back_end }
        (fstate_1shot, init_w_1shot ) = self.run_experiment_in_steps(config_file, N, 1, be, **be_args)

        # run N steps in 2 shots
        # first goes on the CPU
        (fstate_half_shot, init_w_kshot ) = self.run_experiment_in_steps(config_file, N/2, 1, be, **be_args)
        # use the prm file from the 1/2 run above to init this run and finish the full N epochs with GPU
        be='gpu'
        be_args = {'rng_seed': self.seed, be : self.back_end }
        (fstate_kshot, temp_init_w ) = self.run_experiment_in_steps(config_file, N, 1, be, \
                init_config=fstate_half_shot, **be_args)


        # comapre the initial weights
        assert self.check_init_weights( init_w_kshot, init_w_1shot , rtol=rtol, atol=atol ),\
                'initial model weights were not the same'

        #compare the final model state for 1 step of N epochs vs. k steps of N/k epochs
        assert self.model_compare( fstate_1shot, fstate_kshot ,rtol=rtol, atol=atol),\
                'final states of 1 vs %d step learning runs not matching' %k
        return

    #---------------------------------------------------
    @attr('cuda')  # can we do this test by test
    def compare_gpu_to_cpu(self, config_file ):
        # run whole test on cpu and run it 1/2 on cpu the 1/2 on gpu
        # compare the outputs
        N = 10 # for this tests make sure that N/k is not an integer
        k = 5  # k should be 2 here
        atol=0.0
        rtol=.01

        # run N steps in 2 shots and save the checkpoint which should be at int(N/2)
        be='cpu'
        be_args = {'rng_seed': self.seed}#, be : self.back_end }
        (fstate_1shot, init_w_1shot ) = self.run_experiment_in_steps(config_file, N, 1, be, **be_args)

        # run N steps in 2 shots
        # first goes on the CPU
        be='gpu'
        be_args = {'rng_seed': self.seed, be : self.back_end }
        (fstate_half_shot, init_w_kshot ) = self.run_experiment_in_steps(config_file, N/2, 1, be, **be_args)
        # use the prm file from the 1/2 run above to init this run and finish the full N epochs with GPU
        be='cpu'
        be_args = {'rng_seed': self.seed}
        (fstate_kshot, temp_init_w ) = self.run_experiment_in_steps(config_file, N, 1, be, \
                init_config=fstate_half_shot, **be_args)


        # comapre the initial weights
        assert self.check_init_weights( init_w_kshot, init_w_1shot , rtol=rtol, atol=atol ),\
                'initial model weights were not the same'

        #compare the final model state for 1 step of N epochs vs. k steps of N/k epochs
        assert self.model_compare( fstate_1shot, fstate_kshot ,rtol=rtol, atol=atol),\
                'final states of 1 vs %d step learning runs not matching' %k
        return


    def run_experiment_in_steps(self, config_file, N, k, be, init_config=None, **be_args): 
        # run an experiment for N epochs in k stepe with
        # N/k epochs per step.  Last step will be enough epochs
        # to reach N total epochs.  Between each step the 
        # model will be serialized and saved, then reloaded
        # from that saved file at the next step
        # if init_config is not None, then that file will be used
        #  as the initial deserialize file -  this is used for handing off 
        #  models run on cpu.gpu backends to running on gpu/cpu backends

        stepsize = N/k

        for ind in range(1,k+1):
            # run the same learning with N/k epochs k times
            # each time saving and reloading the serialized model state
            if ind == k:
                end_epoch = N # in case N/k is a fraction, 
                              # last step will end at epoch N
            else:
                end_epoch = ind*stepsize

            # load up base experiment config
            experiment = deserialize( config_file )

            # run for N/k steps
            experiment.model.num_epochs = end_epoch

            if ind > 1:
                # after step 1 need to load initial config from last runs serialized file
                experiment.model.deserialized_path = last_saved_state
            elif (init_config != None):
                # used the given pickle file to initialize the mdoel
                experiment.model.deserialized_path = init_config

            # save the model to this file
            last_saved_state = os.path.join(self.model_path,\
                                        '%d_shot_%d.prm' %(k,end_epoch))
            print last_saved_state
            if os.path.exists(last_saved_state):
                print 'removing %s' %last_saved_state
                os.remove(last_saved_state)
            experiment.model.serialized_path = last_saved_state

            
            experiment.model.serialize_schedule = k
            experiment.model.save_checkpoints = k # keep copies of all checkpoint files for cp tests


            backend = gen_backend(model=experiment.model, **be_args)
            experiment.initialize(backend)

            if ind == 1:
                # save the initial weights for check with other runs
                intial_weights={}
                for ind,layer in enumerate(experiment.model.layers):
                    if hasattr(layer,'weights'): # only looking at weights right now
                        # make sure the names of each layer are unique by adding index
                        intial_weights[ '%s_%d' %(layer.name, ind)  ] = \
                                np.copy( layer.weights.asnumpyarray())

            # run 
            res_2shot_half = experiment.run()

        return ( last_saved_state  , intial_weights )


    # load up two model pkl files and compare the data stored in them
    @staticmethod
    def model_compare( model1_file, model2_file, atol=0.0, rtol=0.0):
        model1 = deserialize( model1_file )
        model2 = deserialize( model2_file )

        assert model1.keys().sort() == model2.keys().sort(), \
                'output state data mismatch'

        # remove the epochs from the dictionaries and compare them
        assert model1.pop('epochs_complete') == model2.pop('epochs_complete'), \
                'mismtach in total epoch count' 

        # for MLP just layers should be left?
        print 'checking the 1 versus k step outputs...'
        for ky in model1.keys():
            print ky
            assert TestSerialization.layer_compare(model1[ky], model2[ky], atol=atol,rtol=rtol), 'Mismatch in layer %s' %ky
        print 'OK'

        return True

    # given the dictionary for two layers obtained by
    # deserializing two different pickle files
    # this will compare whether the two layers have
    # the same values. 
    @staticmethod
    def layer_compare(layer1, layer2, atol=0.0, rtol=0.0):
        assert layer1.keys().sort() == layer2.keys().sort()
        for ky in layer1.keys():
            print ky
            assert type(layer1[ky]) == type(layer2[ky])
            assert TestSerialization.val_compare( layer1[ky], layer2[ky], atol=atol, rtol=rtol)
        return True

    # compare two objects obtained from serialized files
    # does element-wise compare for numpy ndarray objects
    # and it called recursively on list object
    # abs and rel tolerances can be passed to the "allclose" call
    @staticmethod
    def val_compare( obj1, obj2, atol=0.0, rtol=0.0 ):
        assert type(obj1) == type(obj2), 'type mismatch'
        if isinstance(obj1, np.ndarray):
            if not np.allclose(obj1, obj2, atol=atol, rtol=rtol):
                maxdiff = np.amax( np.divide(np.absolute( obj1 - obj2 ) , np.absolute(obj2) ) )
                assert False, 'layer weight mismatch max %f' %maxdiff
        if isinstance(obj1, list):
            assert len(obj1) == len(obj2), 'list length mismatch'
            for ind in range(len(obj1)):
                assert TestSerialization.val_compare( obj1[ind], obj2[ind], atol=atol, rtol=rtol)
        return True

    # takes 2 layer weight dictionaries generated in run_experiment_in_steps
    # and compares the weights to within the rel and abs tol values
    # check the initial weights - should match the 1 shot run
    @staticmethod
    def check_init_weights( layer1, layer2, rtol=0.0, atol=0.0 ):
        assert layer1.keys().sort() == layer2.keys().sort()

        for ky in layer1.keys():
            assert np.allclose(layer1[ky], layer2[ky], atol=atol, rtol=rtol), \
                    'initial layer weight mismatch [%s]' %ky
        return True
