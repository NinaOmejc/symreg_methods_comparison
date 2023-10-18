import numpy as np
from scipy.integrate import solve_ivp


class System():

    """
    A class for dynamical system.

    Attributes:
        - name           (string)               arbitrary name of the system (e.g. 'vdp')
        - state_vars     (list of strings)      state or system variables that are described by ODE (e.g. ['x', 'y'])
        - model          (list of strings)      model of the system in the form of right-hand sides of ODEs (e.g. ['C*x', 'x+y'])
        - model_params   (dictionary)           parameters of the model for each of the ODEs (e.g. {C:2})

    Optional attributes:
        - init_bounds    (list of lists)        interval bounds for each state variable from which random
                                                initial states will be drawn. Default = [[-5, 5]] * num_of_state_vars
        - init_condition (string)               either 'none' (default) or 'integers'; Special condition for integer
                                                state variables (only used for lotka-volterra).

    Methods:
        - simulate                              uses solve_ivp function (scipy library) to simulate a system of ODEs
        - get_inits                             randomly samples initial states based on init_bounds, by default from
                                                uniform distribution.
    """

    def __init__(self, name, state_vars, model, model_params={}, **kwargs):

        self.name = name
        self.state_vars = state_vars
        self.model = model
        self.model_params = model_params
        self.init_bounds = kwargs.get('init_bounds', [[-5, 5]] * len(self.state_vars))
        self.init_condition = kwargs.get('init_condition', 'none')
        self.param_bounds = kwargs.get('param_bounds', [-5, 5])

    def simulate(self, inits, time_start=0, time_end=10, time_step=0.01):
        time_span = [time_start, time_end]
        times = np.arange(time_start, time_end, time_step)

        parameters = tuple(self.model_params.keys())
        param_vals = tuple(self.model_params.values())

        # transform model from string to function using eval and lambda.
        model_function = []
        for expr in self.model:
            expr_edited = check_expr(expr)
            model_function.append(eval(f'lambda t, {", ".join(self.state_vars)}, {", ".join(parameters)}: ' + expr_edited))
        def rhs(t, y, *param_keys):
            return [model_function[i](t, *y, *param_keys) for i in range(len(model_function))]

        simulation_results = solve_ivp(rhs, t_span=time_span, y0=inits, t_eval=times, args=param_vals,
                                       method='LSODA', rtol=1e-12, atol=1e-12)

        if simulation_results['success'] == False:
            print('Simulation failed. Data were not generated.')
            return None

        return simulation_results

    def get_inits(self):
        if self.init_condition == 'integers':
            inits = [np.random.randint(low=i[0], high=i[1], size=(1,)) for i in self.init_bounds]
        else:
            inits = [np.random.uniform(low=i[0], high=i[1], size=(1,)) for i in self.init_bounds]
        return [i[0] for i in inits]


def check_expr(expr):
    # check if expression contains any of the following functions and replace them with numpy/scipy equivalents.

    replacements = {'exp': 'np.exp', 'log': 'np.log', 'sqrt': 'np.sqrt',
                    'tan': 'np.tan', 'sin': 'np.sin', 'cos': 'np.cos', 'cot': '1/np.tan'}

    for func, replacement in replacements.items():
        if func in expr:
            expr = expr.replace(func, replacement)

    return expr
