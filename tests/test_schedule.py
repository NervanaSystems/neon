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

import numpy as np

from neon.optimizers import (Schedule, ExpSchedule, PowerSchedule, StepSchedule,
                             ShiftSchedule)
from utils import allclose_with_out


def test_schedule(backend_default):
    """
    Test constant rate, fixed step and various modes of programmable steps.
    """
    lr_init = 0.1

    # default scheduler has a constant learning rate
    sch = Schedule()
    for epoch in range(10):
        lr = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        assert lr == lr_init

    # test a uniform step schedule
    step_config = 2
    change = 0.5
    sch = Schedule(step_config=step_config, change=change)
    for epoch in range(10):
        lr = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        # test a repeated call for the same epoch
        lr2 = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        # print epoch, lr, lr2
        assert allclose_with_out(lr, lr_init * change**(np.floor(epoch // step_config)))
        assert allclose_with_out(lr2, lr_init * change**(np.floor(epoch // step_config)))

    # test a list step schedule
    sch = Schedule(step_config=[2, 3], change=.1)
    assert allclose_with_out(.1, sch.get_learning_rate(learning_rate=.1, epoch=0))
    assert allclose_with_out(.1, sch.get_learning_rate(learning_rate=.1, epoch=1))
    assert allclose_with_out(.01, sch.get_learning_rate(learning_rate=.1, epoch=2))
    # test a repeated call for the same epoch
    assert allclose_with_out(.01, sch.get_learning_rate(learning_rate=.1, epoch=2))
    assert allclose_with_out(.001, sch.get_learning_rate(learning_rate=.1, epoch=3))
    assert allclose_with_out(.001, sch.get_learning_rate(learning_rate=.1, epoch=4))


def test_step_schedule(backend_default):
    """
    Test the StepSchedule class
    """
    step_config = [1, 4, 5]
    change = [0.1, 0.3, 0.4]
    sch = StepSchedule(step_config=step_config, change=change)

    target_lr = [1.0, 0.1, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4, 0.4]

    for e, lr in enumerate(target_lr):
        assert allclose_with_out(lr, sch.get_learning_rate(learning_rate=1.0, epoch=e))


def test_power_schedule(backend_default):
    """
    Test the PowerSchedule class
    """
    sch = PowerSchedule(step_config=2, change=0.5)

    target_lr = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125]

    for e, lr in enumerate(target_lr):
        assert allclose_with_out(lr, sch.get_learning_rate(learning_rate=1.0, epoch=e))


def test_exp_schedule(backend_default):
    """
    Test exponential learning rate schedule
    """
    lr_init = 0.1
    decay = 0.01
    sch = ExpSchedule(decay)
    for epoch in range(10):
        lr = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        assert allclose_with_out(lr, lr_init / (1. + decay * epoch))


def test_shift_schedule(backend_default):
    """
    Test binary shift learning rate schedule
    """
    lr_init = 0.1
    interval = 1
    sch = ShiftSchedule(interval)
    for epoch in range(10):
        lr = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        assert allclose_with_out(lr, lr_init / (2 ** epoch))
