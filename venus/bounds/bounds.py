# ************
# File: bounds.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: class for bounds of nodes.
# ************

from distutils.command.config import config
import numpy as np
import torch
from venus.common.configuration import Config
from venus.gemmpy.calculation_apis import *

class Bounds:
    def __init__(self, config: Config = None, upper=None, lower=None):
        self.config = config
        self.lower = lower
        self.upper = upper
        self.lower_intv = None
        self.upper_intv = None
        if config is not None and config.interval_in_bs is True and config.gpu_in_bs is False:
            self.lower_intv = float_array2np_interval_cpu_array(self.lower.flatten().reshape(lower.flatten().size()[0], -1))
            self.upper_intv = float_array2np_interval_cpu_array(self.upper.flatten().reshape(lower.flatten().size()[0], -1))
        if config is not None and config.interval_in_bs is True and config.gpu_in_bs is True:
            self.lower_intv = torch_float_array2pinterval_gpu_array(self.lower.flatten().reshape(lower.flatten().size()[0], -1))
            self.upper_intv = torch_float_array2pinterval_gpu_array(self.upper.flatten().reshape(lower.flatten().size()[0], -1))

    def get_lower(self):
        if self.config is None or self.config.interval_in_bs is False:
            return self.lower.flatten()
        return self.lower_intv

    def get_upper(self):
        if self.config is None or self.config.interval_in_bs is False:
            return self.upper.flatten()
        return self.upper_intv

    def normalise(self, mean, std):
        """
        Normalises the bounds

        Arguments:
            mean:
                normalisation mean
            std:
                normalisation standard deviation
        Returns
            None
        """
        self.lower = ( self.lower - mean ) / std
        self.upper = ( self.upper - mean ) / std

    def clip(self, min_value, max_value):
        """
        Clips the  bounds

        Arguments:
            min_value:
                valid lower bound
            max_value:
                valid upper bound
        Returns:
            None
        """
        self.lower = np.clip(self.lower, min_value, max_value) 
        self.upper = np.clip(self.upper, min_value, max_value)
        if self.upper_intv is not None and self.config.gpu_in_bs is False:
            self.upper_intv = np.clip(self.upper_intv, min_value, max_value)
            self.lower_intv = np.clip(self.lower_intv, min_value, max_value)
        if self.upper_intv is not None and self.config.gpu_in_bs is True:
            self.upper_intv = torch.clip(self.upper_intv, min_value, max_value)
            self.lower_intv = torch.clip(self.lower_intv, min_value, max_value)


    def get_range(self):
        """
        Returns the range of the bounds.
        """
        return self.upper - self.lower

    def copy(self):
        """
        Copies the bounds.
        """
        self.detach()
        lower = self.lower.clone() if self.lower is not None else None
        upper = self.upper.clone() if self.upper is not None else None

        return Bounds(self.config, upper, lower)

    def detach(self):
        """
        Detaches the bounds. 
        """
        if self.lower is not None:
            self.lower.detach()
        if self.upper is not None:
            self.upper.detach()


