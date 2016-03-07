from ...cl_functions import RanluxCL
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SimulatedAnnealing(AbstractParallelOptimizer):

    default_patience = 500

    def __init__(self, cl_environments, load_balancer, use_param_codec=False, patience=None,
                 optimizer_options=None, **kwargs):
        """Use Simulated Annealing to calculate the optimum.

        This does not use the parameter codec, even if set to True. This because the priors (should) already
        cover the bounds.

        This implementation uses an adapted Metropolis Hasting algorithm to find the optimum. This being a sampling
        routine it does require that the model is a implementation of SampleModelInterface.

        Args:
            patience (int):
                Used to set the maximum number of samples to patience*(number_of_parameters+1)
            optimizer_options (dict): the optimization options. Contains:
                - proposal_update_intervals (int): the interval by which we update the proposal std.

        """
        patience = patience or self.default_patience
        super(SimulatedAnnealing, self).__init__(cl_environments, load_balancer, False, patience=patience, **kwargs)
        optimizer_options = optimizer_options or {}
        self.proposal_update_intervals = optimizer_options.get('proposal_update_intervals', 50)

    def _get_worker_generator(self, *args):
        return lambda cl_environment: SimulatedAnnealingWorker(cl_environment, *args, patience=self.patience,
                                                               proposal_update_intervals=self.proposal_update_intervals)


class SimulatedAnnealingWorker(AbstractParallelOptimizerWorker):

    def __init__(self, *args, **kwargs):
        self.patience = kwargs.pop('patience')
        self.proposal_update_intervals = kwargs.pop('proposal_update_intervals')

        super(SimulatedAnnealingWorker, self).__init__(*args, **kwargs)
        self._use_param_codec = False

    def _get_optimizer_cl_code(self):
        kernel_source = self._get_evaluate_function()

        rand_func = RanluxCL()
        kernel_source += '#define RANLUXCL_LUX 4' + "\n"
        kernel_source += rand_func.get_cl_header()
        kernel_source += rand_func.get_cl_code()

        kernel_source += self._model.get_log_prior_function('getLogPrior')
        kernel_source += self._model.get_proposal_function('getProposal')
        kernel_source += self._model.get_proposal_parameters_update_function('updateProposalParameters')

        if not self._model.is_proposal_symmetric():
            kernel_source += self._model.get_proposal_logpdf('getProposalLogPDF')

        kernel_source += self._model.get_log_likelihood_function('getLogLikelihood', full_likelihood=False)

        kernel_source += self._get_sampling_code()
        return kernel_source

    def _get_optimizer_call_name(self):
        return 'simulated_annealing'

    def _get_sampling_code(self):
        nrm_adaptable_proposal_parameters = len(self._model.get_proposal_parameter_values())
        adaptable_proposal_parameters_str = '{' + ', '.join(map(str, self._model.get_proposal_parameter_values())) + '}'
        acceptance_counters_between_proposal_updates = '{' + ', '.join('0' * self._nmr_params) + '}'

        kernel_source = '''
            void _update_proposals(mot_float_type* const proposal_parameters, uint* const ac_between_proposal_updates,
                                   uint* const proposal_update_count){

                *proposal_update_count += 1;

                if(*proposal_update_count == ''' + str(self.proposal_update_intervals) + '''){
                    updateProposalParameters(ac_between_proposal_updates,
                                             ''' + str(self.proposal_update_intervals) + ''',
                                             proposal_parameters);

                    for(int i = 0; i < ''' + str(nrm_adaptable_proposal_parameters) + '''; i++){
                        ac_between_proposal_updates[i] = 0;
                    }

                    *proposal_update_count = 0;
                }
            }

            void _update_state(mot_float_type* const x,
                               void* rand_settings,
                               double* const current_likelihood,
                               mot_float_type* const current_prior,
                               const optimize_data* const data,
                               mot_float_type* const proposal_parameters,
                               uint * const ac_between_proposal_updates){

                mot_float_type new_prior;
                double new_likelihood;
                double bayesian_f;
                mot_float_type old_x;

                #pragma unroll 1
                for(int k = 0; k < ''' + str(self._nmr_params) + '''; k++){

                    old_x = x[k];
                    x[k] = getProposal(k, x[k], (ranluxcl_state_t*)rand_settings, proposal_parameters);

                    new_prior = getLogPrior(x);

                    if(exp(new_prior) > 0){
                        new_likelihood = getLogLikelihood(data, x);
        '''
        if self._model.is_proposal_symmetric():
            kernel_source += '''
                        bayesian_f = exp((new_likelihood + new_prior) - (*current_likelihood + *current_prior));
                '''
        else:
            kernel_source += '''
                        mot_float_type x_to_prop = getProposalLogPDF(k, old_x, x[k], proposal_parameters);
                        mot_float_type prop_to_x = getProposalLogPDF(k, x[k], x[k], proposal_parameters);

                        bayesian_f = exp((new_likelihood + new_prior + x_to_prop) -
                            (*current_likelihood + *current_prior + prop_to_x));
                '''
        kernel_source += '''
                        if(new_likelihood > *current_likelihood || rand(rand_settings) < bayesian_f){
                            *current_likelihood = new_likelihood;
                            *current_prior = new_prior;
                            ac_between_proposal_updates[k]++;
                        }
                        else{
                            x[k] = old_x;
                        }
                    }
                    else{
                        x[k] = old_x;
                    }
                }
            }

            int simulated_annealing(mot_float_type* const x, const void* const data, void* rand_settings){
                uint proposal_update_count = 0;

                mot_float_type proposal_parameters[] = ''' + adaptable_proposal_parameters_str + ''';
                uint ac_between_proposal_updates[] = ''' + acceptance_counters_between_proposal_updates + ''';

                double current_likelihood = getLogLikelihood((optimize_data*)data, x);
                mot_float_type current_prior = getLogPrior(x);

                rand(rand_settings);

                for(uint i = 0; i < ''' + str(self.patience * (self._nmr_params + 1)) + '''; i++){
                    _update_state(x, rand_settings, &current_likelihood, &current_prior,
                                  data, proposal_parameters, ac_between_proposal_updates);
                    _update_proposals(proposal_parameters, ac_between_proposal_updates, &proposal_update_count);
                }

                return 1;
            }
        '''
        return kernel_source

    def _uses_random_numbers(self):
        return True