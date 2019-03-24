from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Struct, LocalMemory
from mot.library_functions import SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2019-03-24'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class SimulatedAnnealing(SimpleCLLibrary):

    def __init__(self, eval_func, nmr_parameters,
                 patience=10, state_update_func=None, annealing_schedule=None):
        """The simulated annealing implementation.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the function we want to optimize, Should be of signature:
                ``double evaluate(mot_float_type* x, void* data_void);``
            nmr_parameters (int): the number of parameters in the model, this will be hardcoded in the method
            patience (int): translates to the number of iterations in the annealing procedure
            state_update_func (StateUpdateFunc): the method used to advance the state of the annealing process
            annealing_schedule (AnnealingSchedule): the method used to cool down the annealing process
        """
        self._nmr_parameters = nmr_parameters
        self._state_update_func = state_update_func or AMWG()
        self._annealing_schedule = annealing_schedule or Linear()

        nmr_iterations = patience * (nmr_parameters + 1)

        super().__init__('''
            int simulated_annealing(mot_float_type* model_parameters, void* data, void* sa_data){
            
                double* temperature = ((_simulated_annealing_data*)sa_data)->temperature;
                *temperature = ''' + str(self._annealing_schedule.get_starting_temperature(nmr_iterations)) + ''';
                
                double* current_fval = ((_simulated_annealing_data*)sa_data)->current_fval;
                double* best_fval = ((_simulated_annealing_data*)sa_data)->best_fval;
                mot_float_type* best_x = ((_simulated_annealing_data*)sa_data)->best_x;
                
                *current_fval = ''' + eval_func.get_cl_function_name() + '''(model_parameters, data);
                *best_fval = *current_fval;
                
                if(get_local_id(0) == 0){
                    for(uint k = 0; k < ''' + str(self._nmr_parameters) + '''; k++){
                        best_x[k] = model_parameters[k];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                for(ulong i = 0; i < ''' + str(nmr_iterations) + '''; i++){
                    ''' + state_update_func.get_cl_function(eval_func, nmr_parameters).get_cl_function_name() + '''(
                        model_parameters, current_fval, i, *temperature, 
                        ((_simulated_annealing_data*)sa_data)->state_update_data, data);
                    
                    if(get_local_id(0) == 0){
                        if(*current_fval < *best_fval){
                            *best_fval = *current_fval;
                            for(uint k = 0; k < ''' + str(self._nmr_parameters) + '''; k++){
                                best_x[k] = model_parameters[k];
                            }
                        }
                    }
                    
                    ''' + annealing_schedule.get_cl_function(nmr_iterations).get_cl_function_name() + '''(
                        temperature, i, ((_simulated_annealing_data*)sa_data)->annealing_schedule_data);
                }
                
                if(get_local_id(0) == 0){
                    for(uint k = 0; k < ''' + str(self._nmr_parameters) + '''; k++){
                        model_parameters[k] = best_x[k];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                return 0;                         
            }
        ''', dependencies=[eval_func, state_update_func.get_cl_function(eval_func, nmr_parameters),
                           self._annealing_schedule.get_cl_function(nmr_iterations)])

    def get_kernel_data(self):
        return {'sa_data': Struct({
            'temperature': LocalMemory('double', nmr_items=1),
            'current_fval': LocalMemory('double', nmr_items=1),
            'best_fval': LocalMemory('double', nmr_items=1),
            'best_x': LocalMemory('mot_float_type', nmr_items=self._nmr_parameters),
            'state_update_data': self._state_update_func.get_kernel_data(self._nmr_parameters),
            'annealing_schedule_data': self._annealing_schedule.get_kernel_data()},
            '_simulated_annealing_data')}


def get_state_update_func(method_name):
    """Factory method to get a state update functions by name.

    Args:
        method_name (str): the name of the state update function to get

    Returns:
        class: class reference of type :class:`StateUpdateFunc`.
    """
    if method_name == 'AMWG':
        return AMWG
    raise ValueError('The state update function with the name "{}" could not be found.'.format(method_name))


def get_annealing_schedule(method_name):
    """Factory method to get an annealing schedule function by name.

    Args:
        method_name (str): the name of the annealing schedule

    Returns:
        class: class reference of type :class:`AnnealingSchedule`.
    """
    if method_name == 'Linear':
        return Linear
    raise ValueError('The annealing schedule with the name "{}" could not be found.'.format(method_name))


class StateUpdateFunc:
    """Information class for the state update functions."""

    def get_cl_function(self, eval_func, nmr_parameters):
        """Get the CL function for updating the parameter state.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the evaluation function
            nmr_parameters (int): the number of parameters in the model

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for updating the state
        """
        raise NotImplementedError()

    def get_kernel_data(self, nmr_parameters):
        """Get the kernel data needed by this state update function.

        Args:
            nmr_parameters (int): the number of parameters in the model

        Returns:
            mot.lib.kernel_data.Struct: the structure with the kernel data for this state update function
        """
        raise NotImplementedError()


class AnnealingSchedule:
    """Information class for the annealing schedule functions."""

    def get_cl_function(self, nmr_iterations):
        """Get the CL function for updating the temperature.

        Args:
            nmr_iterations (int): the number of iterations in the annealing process

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for updating the temperature
        """
        raise NotImplementedError()

    def get_kernel_data(self):
        """Get the kernel data needed by this annealing schedule function.

        Returns:
            mot.lib.kernel_data.Struct: the structure with the kernel data for this annealing schedule function
        """
        raise NotImplementedError()

    def get_starting_temperature(self, nmr_iterations):
        """Get the starting temperature for this annealing schedule.

        Args:
            nmr_iterations (int): the number of iterations in the annealing process

        Returns:
            float: the starting temperature
        """
        raise NotImplementedError()


class AMWG:

    def __init__(self):
        """This uses the Adaptive Metropolis-Within-Gibbs (AMWG) MCMC algorithm [1] for updating the annealing state.

        References:
            [1] Roberts GO, Rosenthal JS. Examples of adaptive MCMC. J Comput Graph Stat. 2009;18(2):349-367.
                doi:10.1198/jcgs.2009.06134.
        """
        self._target_acceptance_rate = 0.44
        self._batch_size = 50
        self._damping_factor = 1
        self._min_val = 1e-15
        self._max_val = 1e3

    def get_cl_function(self, eval_func, nmr_parameters):
        update_proposal_state = SimpleCLFunction.from_string('''
            void _updateProposalState(
                    mot_float_type* x, 
                    ulong current_iteration, 
                    mot_float_type* proposal_stds,
                    mot_float_type* acceptance_counter){
                
                if(current_iteration > 0 && current_iteration % ''' + str(self._batch_size) + ''' == 0){
                    mot_float_type delta = sqrt(1.0/
                            (''' + str(self._damping_factor) + ''' * 
                                (current_iteration / ''' + str(self._batch_size) + ''')));
                    
                    for(uint k = 0; k < ''' + str(nmr_parameters) + '''; k++){
                        if(acceptance_counter[k] / (mot_float_type)''' + str(self._batch_size) + ''' 
                                > ''' + str(self._target_acceptance_rate) + '''){
                            proposal_stds[k] *= exp(delta);
                        }
                        else{
                            proposal_stds[k] /= exp(delta);
                        }
        
                        proposal_stds[k] = clamp(proposal_stds[k], 
                                                 (mot_float_type)''' + str(self._min_val) + ''', 
                                                 (mot_float_type)''' + str(self._max_val) + ''');
        
                        acceptance_counter[k] = 0;
                    }
                }             
            }
        ''')

        return SimpleCLFunction.from_string('''
             void _AMWG_state_update_function(
                    mot_float_type* x,
                    double* current_fval,
                    ulong iteration,
                    double temperature,
                    void* state_update_data,
                    void* data){
                
                bool is_first_work_item = get_local_id(0) == 0;
                
                mot_float_type* proposal_stds = ((_AMWG_state_update_data*)state_update_data)->proposal_stds;
                mot_float_type* acceptance_counter = ((_AMWG_state_update_data*)state_update_data)->acceptance_counter;
                
                if(is_first_work_item){
                    if(iteration == 0){
                        for(uint k = 0; k < ''' + str(nmr_parameters) + '''; k++){
                            proposal_stds[k] = 1;
                            acceptance_counter[k] = 0;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                double tmp;
                mot_float_type new_fval;
            
                for(uint k = 0; k < ''' + str(nmr_parameters) + '''; k++){
                    if(is_first_work_item){
                        tmp = x[k];
                        x[k] += frandn() * proposal_stds[k];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    new_fval = ''' + eval_func.get_cl_function_name() + '''(x, data);
                    
                    if(is_first_work_item){
                        if(new_fval < *current_fval || frand() < exp(-(new_fval - *current_fval) / temperature)){
                            *current_fval = new_fval;
                            acceptance_counter[k]++;
                        }
                        else{
                            x[k] = tmp;
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                
                if(is_first_work_item){
                    _updateProposalState(x, iteration, proposal_stds, acceptance_counter);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
             }
        ''', dependencies=[eval_func, update_proposal_state])

    def get_kernel_data(self, nmr_parameters):
        return Struct({
            'proposal_stds': LocalMemory('mot_float_type', nmr_parameters),
            'acceptance_counter': LocalMemory('mot_float_type', nmr_parameters)},
            '_AMWG_state_update_data')


class Linear(AnnealingSchedule):

    def __init__(self, starting_temperature=10):
        """Simple linear annealing schedule.

        This will linearly decrease with each iteration, such that at the end of all iterations the starting temperature
        is near one. That is, this method follows the rule :math:`t_i = t_0 - (t_0 / it_{max}) * i`, where
        :math:`t_{i}` is the temperature at iteration :math:`i`, :math:`it_{max}` is the maximum number of iterations
        and :math:`i` is the current number of iterations.

        Args:
            starting_temperature (float): the starting temperature
        """
        self._starting_temperature = starting_temperature

    def get_cl_function(self, nmr_iterations):
        return SimpleCLFunction.from_string('''
            void _Linear_annealing_schedule(double* temperature, ulong iteration, void* annealing_schedule_data){
                ulong starting_temperature = ''' + str(self._starting_temperature) + ''';
                ulong nmr_iterations = ''' + str(nmr_iterations) + ''';
                
                *temperature = starting_temperature - (starting_temperature / nmr_iterations) * iteration;
            }        
        ''')

    def get_starting_temperature(self, nmr_iterations):
        return self._starting_temperature

    def get_kernel_data(self):
        return Struct({}, '_linear_annealing_schedule_data')

