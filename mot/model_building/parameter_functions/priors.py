import numpy as np
from mot.cl_data_type import CLDataType

__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterPrior(object):
    """The priors are used during model sampling.

    These priors should be in the

    They indicate the a priori information one has about a parameter.
    """

    def get_prior_function(self):
        """Get the prior function as a CL string. This should include include guards (#ifdef's).

        This should follow the signature:

        .. code-block: c

            mot_float_type <prior_fname>(mot_float_type parent_parameter,
                                         mot_float_type lower_bound,
                                         mot_float_type upper_bound,
                                         <sub-parameters>)

        That is, the parent parameter and it lower and upper bound is given next to the optional parameters
        defined in this prior.

        Returns:
            str: The cl function
        """

    def get_prior_function_name(self):
        """Get the name of the prior function call.

         This is used by the model builder to construct the call to the prior function.

         Returns:
            str: name of the function
        """

    def get_parameters(self):
        """Get the additional parameters featured in this prior.

        This can return a list of additional parameters to be used in the model function.

        Returns:
            list of CLFunctionParameter: the list of function parameters to be added to the list of
                parameters of the enclosing model.
        """
        return []


class UniformPrior(AbstractParameterPrior):
    """The uniform prior is always 1."""

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_UNIFORM
            #define PRIOR_UNIFORM

            mot_float_type prior_uniform_prior(const mot_float_type parameter,
                                               const mot_float_type lower_bound,
                                               const mot_float_type upper_bound){
                return 1;
            }

            #endif //PRIOR_UNIFORM
        '''

    def get_prior_function_name(self):
        return 'prior_uniform_prior'


class UniformWithinBoundsPrior(AbstractParameterPrior):
    """This prior is 1 within the upper and lower bound of the parameter, 0 outside."""

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_UNIFORM_WITHIN_BOUNDS
            #define PRIOR_UNIFORM_WITHIN_BOUNDS

            mot_float_type prior_uniform_within_bounds(const mot_float_type parameter,
                                                       const mot_float_type lower_bound,
                                                       const mot_float_type upper_bound){

                return (parameter < lower_bound || parameter > upper_bound) ? 0.0 : 1.0;
            }

            #endif //PRIOR_UNIFORM_WITHIN_BOUNDS
        '''

    def get_prior_function_name(self):
        return 'prior_uniform_within_bounds'


class AbsSinPrior(AbstractParameterPrior):
    """The fabs(sin(x)) prior."""

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_ABSSIN
            #define PRIOR_ABSSIN

            mot_float_type prior_abs_sin(const mot_float_type parameter,
                                               const mot_float_type lower_bound,
                                               const mot_float_type upper_bound){
                return fabs(sin(parameter));
            }

            #endif //PRIOR_ABSSIN
        '''

    def get_prior_function_name(self):
        return 'prior_abs_sin'


class AbsSinHalfPrior(AbstractParameterPrior):
    """The fabs(sin(x)/2.0) prior."""

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_ABSSIN_HALF
            #define PRIOR_ABSSIN_HALF

            mot_float_type prior_abs_sin_half(const mot_float_type parameter,
                                               const mot_float_type lower_bound,
                                               const mot_float_type upper_bound){
                return fabs(sin(parameter)/2.0);
            }

            #endif //PRIOR_ABSSIN_HALF
        '''

    def get_prior_function_name(self):
        return 'prior_abs_sin_half'


class NormalPDF(AbstractParameterPrior):

    def get_parameters(self):
        from mot.model_building.cl_functions.parameters import FreeParameter
        return [FreeParameter(CLDataType.from_string('mot_float_type'), 'mu', True, 0, -np.inf, np.inf,
                              sampling_prior=UniformPrior()),
                FreeParameter(CLDataType.from_string('mot_float_type'), 'sigma', False, 1, -np.inf, np.inf,
                              sampling_prior=UniformPrior())]

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_NORMALPDF
            #define PRIOR_NORMALPDF

            mot_float_type prior_normal_pdf(const mot_float_type parameter,
                                            const mot_float_type lower_bound,
                                            const mot_float_type upper_bound,
                                            const mot_float_type prior_mean,
                                            const mot_float_type prior_sigma){

                return exp(-pown(parameter - prior_mean, 2) / (2 * pown(prior_sigma, 2)))
                        / (prior_sigma * sqrt(2 * M_PI));

            }

            #endif //PRIOR_NORMALPDF
        '''

    def get_prior_function_name(self):
        return 'prior_normal_pdf'
