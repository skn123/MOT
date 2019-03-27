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
                 patience=10, initial_temperature=10, restart_interval=200,
                 state_update_func=None, annealing_schedule=None):
        """Function minimization using simulated annealing.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the function we want to optimize, Should be of signature:
                ``double evaluate(mot_float_type* x, void* data_void);``
            nmr_parameters (int): the number of parameters in the model, this will be hardcoded in the method
            patience (int): translates to the number of iterations in the annealing procedure
            initial_temperature (float): the initial temperature
            restart_interval (float): restart the system after this many iterations (controls the inner annealing loop).
            state_update_func (StateUpdateFunc): the method used to advance the state of the annealing process
            annealing_schedule (AnnealingSchedule): the method used to cool down the annealing process
        """
        self._nmr_parameters = nmr_parameters
        self._state_update_func = state_update_func or Fast()
        self._annealing_schedule = annealing_schedule or Exponential()

        nmr_iterations = patience * (nmr_parameters + 1)

        super().__init__('''
            int simulated_annealing(
                    mot_float_type* model_parameters, 
                    mot_float_type* lower_bounds,
                    mot_float_type* upper_bounds,
                    void* data, 
                    void* sa_data){
                
                const float initial_temperature = ''' + str(initial_temperature) + ''';
                
                uint* annealing_parameter = ((_simulated_annealing_data*)sa_data)->annealing_parameter;
                float* temperature        = ((_simulated_annealing_data*)sa_data)->temperature;
                double* current_fval      = ((_simulated_annealing_data*)sa_data)->current_fval;
                double* best_fval         = ((_simulated_annealing_data*)sa_data)->best_fval;
                mot_float_type* best_x    = ((_simulated_annealing_data*)sa_data)->best_x;
                
                *current_fval = ''' + eval_func.get_cl_function_name() + '''(model_parameters, data);
                *best_fval = *current_fval;
                
                if(get_local_id(0) == 0){
                    for(uint k = 0; k < ''' + str(self._nmr_parameters) + '''; k++){
                        best_x[k] = model_parameters[k];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                ''' + str(self._annealing_schedule.get_init_function().get_cl_function_name()) + '''(
                    initial_temperature,                    
                    ((_simulated_annealing_data*)sa_data)->annealing_schedule_data);
                
                ''' + state_update_func.get_init_function(eval_func, nmr_parameters).get_cl_function_name() + '''(
                        model_parameters, current_fval, *temperature, 
                        ((_simulated_annealing_data*)sa_data)->state_update_data, data);
                
                for(uint i = 0; i < ''' + str(nmr_iterations) + '''; i++){
                    for(uint k = 0; k < ''' + str(restart_interval) + '''; k++){
                        ''' + state_update_func.get_cl_function(eval_func, nmr_parameters).get_cl_function_name() + '''(
                            model_parameters, current_fval, k, *temperature, lower_bounds, upper_bounds,
                            ((_simulated_annealing_data*)sa_data)->state_update_data, data);
                        
                        ''' + annealing_schedule.get_cl_function().get_cl_function_name() + '''(
                            temperature, k + 1, initial_temperature, 
                            ((_simulated_annealing_data*)sa_data)->annealing_schedule_data);
                        
                        if(get_local_id(0) == 0){
                            if(*current_fval < *best_fval){
                                *best_fval = *current_fval;
                                for(uint k = 0; k < ''' + str(self._nmr_parameters) + '''; k++){
                                    best_x[k] = model_parameters[k];
                                }
                            }
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    
                    // restart
                    if(get_local_id(0) == 0){
                        *current_fval = *best_fval;
                        for(uint k = 0; k < ''' + str(self._nmr_parameters) + '''; k++){
                            model_parameters[k] = best_x[k];
                        }
                        *temperature = initial_temperature;
                        
                        ''' + str(self._annealing_schedule.get_init_function().get_cl_function_name()) + '''(
                            initial_temperature,                    
                            ((_simulated_annealing_data*)sa_data)->annealing_schedule_data);
                        
                        ''' + state_update_func.get_init_function(eval_func, nmr_parameters).get_cl_function_name() +
                            '''(model_parameters, current_fval, *temperature, 
                                ((_simulated_annealing_data*)sa_data)->state_update_data, data);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                
                if(get_local_id(0) == 0){
                    for(uint k = 0; k < ''' + str(self._nmr_parameters) + '''; k++){
                        model_parameters[k] = best_x[k];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                return 0;                         
            }
        ''', dependencies=[eval_func,
                           self._state_update_func.get_cl_function(eval_func, nmr_parameters),
                           self._state_update_func.get_init_function(eval_func, nmr_parameters),
                           self._annealing_schedule.get_cl_function(),
                           self._annealing_schedule.get_init_function()])

    def get_kernel_data(self):
        return {'sa_data': Struct({
            'temperature': LocalMemory('float', nmr_items=1),
            'annealing_parameter': LocalMemory('uint', nmr_items=1),
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
    elif method_name == 'Fast':
        return Fast
    elif method_name == 'Boltz':
        return BoltzUpdate
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
    elif method_name == 'Exponential':
        return Exponential
    elif method_name == 'Boltz':
        return BoltzSchedule
    raise ValueError('The annealing schedule with the name "{}" could not be found.'.format(method_name))


class StateUpdateFunc:
    """Information class for the state update functions."""

    def get_init_function(self, eval_func, nmr_parameters):
        """Get the CL function to initialize this state update method.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the evaluation function
            nmr_parameters (int): the number of parameters in the model

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for the initialization. Should have signature:

            .. code-block:: c

                void <name>(
                    mot_float_type* x,
                    double* current_fval,
                    float temperature,
                    void* state_update_data,
                    void* data);
        """
        raise NotImplementedError()

    def get_cl_function(self, eval_func, nmr_parameters):
        """Get the CL function for updating the parameter state.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the evaluation function
            nmr_parameters (int): the number of parameters in the model

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for updating the state. Should have signature:

            .. code-block:: c

                void <name>(
                    mot_float_type* x,
                    double* current_fval,
                    uint iteration,
                    float temperature,
                    mot_float_type* lower_bounds,
                    mot_float_type* upper_bounds,
                    void* state_update_data,
                    void* data);
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

    def get_init_function(self):
        """Get the CL function to initialize this annealing schedule.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for the initialization. Should have signature:

            .. code-block:: c

                void <name>(float initial_temperature, void* annealing_schedule_data);
        """
        raise NotImplementedError()

    def get_cl_function(self):
        """Get the CL function for updating the temperature.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for updating the temperature. Should have signature:

            .. code-block:: c

                void <name>(float* temperature, uint iteration,
                            float initial_temperature, void* annealing_schedule_data);
        """
        raise NotImplementedError()

    def get_kernel_data(self):
        """Get the kernel data needed by this annealing schedule function.

        Returns:
            mot.lib.kernel_data.Struct: the structure with the kernel data for this annealing schedule function
        """
        raise NotImplementedError()


class Fast(StateUpdateFunc):

    def __init__(self):
        """Sets the length equals to the current temperature, with an uniform random direction.

        This method shifts the proposal, if necessary, to stay within bounds. Each infeasible component of the proposal
        is shifted to a random value between the violated bound and the (feasible) value at the previous iteration.
        """

    def get_init_function(self, eval_func, nmr_parameters):
        return SimpleCLFunction.from_string('''
             void _Fast_state_update_function_init(
                    mot_float_type* x,
                    double* current_fval,
                    float temperature,
                    void* state_update_data,
                    void* data){
            }
        ''')

    def get_cl_function(self, eval_func, nmr_parameters):
        return SimpleCLFunction.from_string('''
             void _Fast_state_update_function(
                    mot_float_type* x,
                    double* current_fval,
                    uint iteration,
                    float temperature,
                    mot_float_type* lower_bounds,
                    mot_float_type* upper_bounds,
                    void* state_update_data,
                    void* data){
                
                bool is_first_work_item = get_local_id(0) == 0;
                
                mot_float_type* x_old = ((_Fast_state_update_data*)state_update_data)->x_old;
                
                if(is_first_work_item){
                    for(uint k = 0; k < ''' + str(nmr_parameters) + '''; k++){
                        x_old[k] = x[k];
                        x[k] += frand() * temperature;
                        
                        if(x[k] > upper_bounds[k]){
                            x[k] = x_old[k] + frand() * (upper_bounds[k] - x_old[k]);
                        }
                        if(x[k] < lower_bounds[k]){
                            x[k] = lower_bounds[k] + frand() * (x_old[k] - lower_bounds[k]);
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                double new_fval = ''' + eval_func.get_cl_function_name() + '''(x, data);
                
                if(is_first_work_item){
                    if(new_fval < *current_fval 
                            || frand() < (1 / (1 + exp((new_fval - *current_fval) / temperature)))){
                        *current_fval = new_fval;
                    }    
                    else{
                        for(uint k = 0; k < ''' + str(nmr_parameters) + '''; k++){
                            x[k] = x_old[k];
                        }   
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
             }
        ''', dependencies=[eval_func])

    def get_kernel_data(self, nmr_parameters):
        return Struct({
            'x_old': LocalMemory('mot_float_type', nmr_parameters)},
            '_Fast_state_update_data')


class BoltzUpdate(StateUpdateFunc):

    def __init__(self):
        """Sets the length equals to the square root of the current temperature, with an uniform random direction.

        This method shifts the proposal, if necessary, to stay within bounds. Each infeasible component of the proposal
        is shifted to a random value between the violated bound and the (feasible) value at the previous iteration.
        """

    def get_init_function(self, eval_func, nmr_parameters):
        return SimpleCLFunction.from_string('''
             void _Boltz_state_update_function_init(
                    mot_float_type* x,
                    double* current_fval,
                    float temperature,
                    void* state_update_data,
                    void* data){
            }
        ''')

    def get_cl_function(self, eval_func, nmr_parameters):
        return SimpleCLFunction.from_string('''
             void _Boltz_state_update_function(
                    mot_float_type* x,
                    double* current_fval,
                    uint iteration,
                    float temperature,
                    mot_float_type* lower_bounds,
                    mot_float_type* upper_bounds,
                    void* state_update_data,
                    void* data){

                bool is_first_work_item = get_local_id(0) == 0;

                mot_float_type* x_old = ((_Boltz_state_update_data*)state_update_data)->x_old;

                if(is_first_work_item){
                    for(uint k = 0; k < ''' + str(nmr_parameters) + '''; k++){
                        x_old[k] = x[k];
                        x[k] += frand() * log(temperature);

                        if(x[k] > upper_bounds[k]){
                            x[k] = x_old[k] + frand() * (upper_bounds[k] - x_old[k]);
                        }
                        if(x[k] < lower_bounds[k]){
                            x[k] = lower_bounds[k] + frand() * (x_old[k] - lower_bounds[k]);
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                double new_fval = ''' + eval_func.get_cl_function_name() + '''(x, data);

                if(is_first_work_item){
                    if(new_fval < *current_fval 
                            || frand() < (1 / (1 + exp((new_fval - *current_fval) / temperature)))){
                        *current_fval = new_fval;
                    }    
                    else{
                        for(uint k = 0; k < ''' + str(nmr_parameters) + '''; k++){
                            x[k] = x_old[k];
                        }   
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
             }
        ''', dependencies=[eval_func])

    def get_kernel_data(self, nmr_parameters):
        return Struct({
            'x_old': LocalMemory('mot_float_type', nmr_parameters)},
            '_Boltz_state_update_data')


class AMWG(StateUpdateFunc):

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

    def get_init_function(self, eval_func, nmr_parameters):
        return SimpleCLFunction.from_string('''
             void _AMWG_state_update_function_init(
                    mot_float_type* x,
                    double* current_fval,
                    float temperature,
                    void* state_update_data,
                    void* data){

                mot_float_type* proposal_stds = ((_AMWG_state_update_data*)state_update_data)->proposal_stds;
                uint* acceptance_counter = ((_AMWG_state_update_data*)state_update_data)->acceptance_counter;

                if(get_local_id(0) == 0){
                    for(uint k = 0; k < ''' + str(nmr_parameters) + '''; k++){
                        proposal_stds[k] = 1;
                        acceptance_counter[k] = 0;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        ''')

    def get_cl_function(self, eval_func, nmr_parameters):
        update_proposal_state = SimpleCLFunction.from_string('''
            void _updateProposalState(
                    mot_float_type* x, 
                    uint current_iteration, 
                    mot_float_type* proposal_stds,
                    uint* acceptance_counter){
                
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
                    uint iteration,
                    float temperature,
                    mot_float_type* lower_bounds,
                    mot_float_type* upper_bounds,
                    void* state_update_data,
                    void* data){
                
                bool is_first_work_item = get_local_id(0) == 0;
                
                mot_float_type* proposal_stds = ((_AMWG_state_update_data*)state_update_data)->proposal_stds;
                uint* acceptance_counter = ((_AMWG_state_update_data*)state_update_data)->acceptance_counter;                
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
                        if(new_fval < *current_fval 
                                || frand() < (1 / (1 + exp((new_fval - *current_fval) / temperature)))){
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
            'acceptance_counter': LocalMemory('uint', nmr_parameters)},
            '_AMWG_state_update_data')


class Linear(AnnealingSchedule):

    def __init__(self, reduction_factor=0.01):
        """Simple linear annealing schedule.

        Sets the temperature at any point to the initial temperature divided by the annealing iteration counter.

        Args:
            reduction_factor (float): the linear reduction factor
        """
        self._reduction_factor = reduction_factor

    def get_init_function(self):
        return SimpleCLFunction.from_string('''
            void _Linear_annealing_schedule_init(float initial_temperature, void* annealing_schedule_data){
            }        
        ''')

    def get_cl_function(self):
        return SimpleCLFunction.from_string('''
            void _Linear_annealing_schedule(float* temperature, uint iteration, 
                                            float initial_temperature, void* annealing_schedule_data){
                if(get_local_id(0) == 0){
                    *temperature = initial_temperature / iteration;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }        
        ''')

    def get_kernel_data(self):
        return Struct({}, '_linear_annealing_schedule_data')


class Exponential(AnnealingSchedule):

    def __init__(self, reduction_factor=0.98):
        """Simple linear annealing schedule.

        This will set the temperature to an exponentially decreased temperature. That is, it sets
        the temperature to ``starting_temperature * reduction_factor^i`` with i the annealing iteration counter.

        Args:
            reduction_factor (float): the exponential reduction factor
        """
        self._reduction_factor = reduction_factor

    def get_init_function(self):
        return SimpleCLFunction.from_string('''
            void _Exponential_annealing_schedule_init(float initial_temperature, void* annealing_schedule_data){
            }        
        ''')

    def get_cl_function(self):
        return SimpleCLFunction.from_string('''
            void _Exponential_annealing_schedule(float* temperature, uint iteration, 
                                                 float initial_temperature, void* annealing_schedule_data){
                if(get_local_id(0) == 0){
                    *temperature = initial_temperature * pown(''' + str(self._reduction_factor) + ''', iteration);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }        
        ''')

    def get_kernel_data(self):
        return Struct({}, '_linear_annealing_schedule_data')


class BoltzSchedule(AnnealingSchedule):

    def __init__(self):
        """Simple linear annealing schedule.

        This will set the temperature to an exponentially decreased temperature. That is, it sets
        the temperature to ``starting_temperature * log(i)`` with i the annealing iteration counter.
        """

    def get_init_function(self):
        return SimpleCLFunction.from_string('''
            void _Boltz_annealing_schedule_init(float initial_temperature, void* annealing_schedule_data){
            }        
        ''')

    def get_cl_function(self):
        return SimpleCLFunction.from_string('''
            void _Boltz_annealing_schedule(float* temperature, uint iteration, 
                                           float initial_temperature, void* annealing_schedule_data){
                if(get_local_id(0) == 0){
                    *temperature = initial_temperature / log((double)iteration);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }        
        ''')

    def get_kernel_data(self):
        return Struct({}, '_linear_annealing_schedule_data')
