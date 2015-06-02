# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
Experiment in which a model is trained (parameters learned), then performance
is evaluated on the predictions made.
"""

import logging
import os

from neon.util.persist import serialize
from neon.experiments.fit import FitExperiment
from neon.util.param import opt_param
from neon.util.defaults import default_metric

logger = logging.getLogger(__name__)


class FitPredictErrorExperiment(FitExperiment):
    """
    In this `Experiment`, a model is first trained on a training dataset to
    learn a set of parameters, then these parameters are used to generate
    predictions on specified test datasets, and the resulting performance is
    measured then returned.

    Note that a pre-fit model may be loaded depending on serialization
    parameters (rather than learning from scratch).  The same may also apply to
    the datasets specified.

    Keyword Args:
        backend (neon.backends.Backend): The backend to associate with the
                                         datasets to use in this experiment

    TODO:
        add other params
    """
    def __init__(self, **kwargs):
        super(FitPredictErrorExperiment, self).__init__(**kwargs)
        opt_param(self, ['diagnostics'], {'timing': False, 'ranges': False})
        opt_param(self, ['metrics'], default_metric())
        opt_param(self, ['predictions'], {})

    def initialize(self, backend):
        if self.live:
            if not hasattr(self.dataset, 'live'):
                raise AttributeError('This dataset does not support '
                                     'live inference')
            self.model.batch_size = 1
            self.dataset.live = True
        super(FitPredictErrorExperiment, self).initialize(backend)

    def save_results(self, dataset, setname, data, dataname):
        out_dir = os.path.join(dataset.repo_path, dataset.__class__.__name__)
        if hasattr(dataset, 'save_dir'):
            out_dir = dataset.save_dir
        out_dir = os.path.expandvars(os.path.expanduser(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        filename = os.path.join(out_dir, '{}-{}.pkl'.format(setname, dataname))
        serialize(data.asnumpyarray().T, filename)

    def run(self):
        """
        Actually carry out each of the experiment steps.

        Returns:
            dict: of inference_metric names, each entry of which is a dict
                  containing inference_set name keys, and actual metric values
        """
        result = dict()
        # if the experiment includes timing diagnostics, decorate backend
        if self.diagnostics['timing']:
            self.backend.flop_timing_init(self.diagnostics['decorate_fc'],
                                          self.diagnostics['decorate_conv'],
                                          self.diagnostics['decorate_ew'])
            self.model.timing_plots = True

        # if the experiment includes parameter statistics
        if self.diagnostics['ranges']:
            from neon.diagnostics import ranges_decorators
            rd = ranges_decorators.Decorators(backend=self.backend,
                                              verbosity=self.
                                              diagnostics['verbosity'])
            rd.decorate(function_list=self.diagnostics)

        # Load the data and train the model.
        super(FitPredictErrorExperiment, self).run()
        if self.live:
            self.predict_live()
            return

        # switch to inference mode
        self.model.set_train_mode(False)

        # Generate and save predictions
        for pred_set in self.predictions:
            if not self.dataset.has_set(pred_set):
                logger.warning("Unable to generate '%s' predictions, no "
                               "equivalent dataset partition" % pred_set)
                continue

            outputs, targets = self.model.predict_fullset(self.dataset,
                                                          pred_set)
            self.save_results(self.dataset, pred_set, outputs, 'inference')
            self.save_results(self.dataset, pred_set, targets, 'targets')
            # update any metrics for this set while we have this info
            if pred_set in self.metrics:
                for m in self.metrics[pred_set]:
                    m.add(targets, outputs)

        # Report error metrics.
        for metric_set in self.metrics:
            if not self.dataset.has_set(metric_set):
                logger.warning("Unable to generate '%s' metrics, no "
                               "equivalent dataset partition" % metric_set)
                continue
            if metric_set not in result:
                result[metric_set] = dict()
            if metric_set not in self.predictions:
                for outputs, targets in self.model.predict_generator(
                        self.dataset, metric_set):
                    # update metrics for this set while we have this info
                    for m in self.metrics[metric_set]:
                        m.add(targets, outputs)
            for m in self.metrics[metric_set]:
                metric_name = str(m)
                logger.info('%s set %s %.5f', metric_set, metric_name,
                            m.report())
                result[metric_set][metric_name] = m.report()

        # visualization (if so requested)
        if self.diagnostics['timing']:
            from neon.diagnostics import timing_plots as tp
            tp.print_performance_stats(self.backend, logger)
        if self.diagnostics['ranges']:
            from neon.diagnostics import ranges_plots as rp
            rp.print_param_stats(self.backend, logger,
                                 self.diagnostics['prefix'])
        self.dataset.unload()
        return result

    def predict_live(self):
        self.model.predict_live_init(self.dataset)
        logger.info('Ready to perform live inference')
        while True:
            try:
                result = self.model.predict_live()
                self.dataset.process_result(result.asnumpyarray())
            except KeyboardInterrupt:
                logger.info('Execution interrupted.')
                self.dataset.unload()
                break
