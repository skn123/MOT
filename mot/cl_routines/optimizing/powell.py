import os
from pkg_resources import resource_filename
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Powell(AbstractParallelOptimizer):

    default_patience = 2

    def __init__(self, bracket_gold=None, glimit=None, reset_method=None, patience=None, **kwargs):
        """Use the Powell method to calculate the optimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            bracket_gold (double): the default ratio by which successive intervals are magnified in Bracketing
            glimit (double): the maximum magnification allowed for a parabolic-fit step in Bracketing
            reset_method (str): one of 'EXTRAPOLATED_POINT' or 'RESET_TO_IDENTITY' lower case or upper case.
        """
        patience = patience or self.default_patience

        optimizer_settings = kwargs.pop('optimizer_settings', {})
        optimizer_settings['bracket_gold'] = bracket_gold
        optimizer_settings['glimit'] = glimit
        optimizer_settings['reset_method'] = reset_method

        option_defaults = {'bracket_gold': 1.618034, 'glimit': 100.0, 'reset_method': 'reset_to_identity'}

        def get_value(option_name, default):
            value = optimizer_settings.get(option_name, None)
            if value is None:
                return default
            return value

        for option, default in option_defaults.items():
            optimizer_settings.update({option: get_value(option, default)})

        super(Powell, self).__init__(patience=patience, optimizer_settings=optimizer_settings, **kwargs)

    def _get_worker_generator(self, *args):
        return lambda cl_environment: PowellWorker(cl_environment, *args)


class PowellWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        params = {'NMR_PARAMS': self._nmr_params, 'PATIENCE': self._parent_optimizer.patience}

        for option, value in self._optimizer_settings.items():
            params.update({option.upper(): value})
        params['RESET_METHOD'] = 'POWELL_RESET_METHOD_' + params['RESET_METHOD'].upper()

        body = open(os.path.abspath(resource_filename('mot', 'data/opencl/powell.pcl')), 'r').read()
        if params:
            body = body % params
        return body

    def _get_optimizer_call_name(self):
        return 'powell'
