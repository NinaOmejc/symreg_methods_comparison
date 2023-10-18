from src.utils.system import System
import numpy as np

"""
A collection of Systems objects that are in the benchmark. 
At the end of the script, there is also a collection of initial conditions, that were used for each of the 4 runs.
"""

# bacterial respiration (pg 288, strogatz textbook)
sys_bacres = System(name="bacres",
                    state_vars=["x", "y"],
                    model=["B - x - ((x * y) / (q*x**2 + 1))",
                           "A - (x * y / (q*x**2 + 1))"],
                    model_params={'B': 20, 'A':10, 'q':0.5},
                    init_bounds=[[2, 8], [9, 11]],
                    param_bounds=[-20, 20],
                    )

sys_barmag = System(name="barmag",
                    state_vars=["x", "y"],
                    model=["K * sin(x-y) - sin(x)",
                           "K * sin(y-x) - sin(y)"],
                    model_params={'K': 0.5},
                    init_bounds=[[0, 2*np.pi], [0, 2*np.pi]],
                    param_bounds=[-5, 5]
                    )

sys_glider = System(name="glider",
                    state_vars=["x", "y"],
                    model=["-D * x**2 - sin(y)",
                           "x - (cos(y)/x)"],
                    model_params= {'D': 0.05},
                    init_bounds=[[2, 8], [-3, 3]],
                    param_bounds=[-5, 5]
                    )

sys_lv = System(name="lv",
                state_vars=["x", "y"],
                model=["x * (A - x - B*y)",
                       "y * (C  - x - y)"],
                model_params= {'A': 3, 'B': 2, 'C': 2},
                init_bounds=[[1, 10], [1, 10]],
                init_condition='integers',
                param_bounds=[-5, 5]
                )

sys_predprey = System(name="predprey",
                state_vars=["x", "y"],
                model=["x*(b - x - y/(1+x))",
                       "y*(x/(1+x) - a*y)"],
                model_params= {'a': 0.075, 'b': 4},
                init_bounds=[[2, 8], [9, 11]],
                param_bounds=[-5, 5]
                )

sys_shearflow = System(name="shearflow",
                       state_vars=["x", "y"],
                       model=["cot(y) * cos(x)",
                               "(cos(y)**2 + A*sin(y)**2) * sin(x)"],
                       model_params= {'A': 0.1},
                       init_bounds=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]],
                       param_bounds=[-5, 5]
                       )

sys_vdp = System(name="vdp",
                state_vars=["x", "y"],
                model=["y",
                       "-x - M*y*(x**2 - 1)"],
                model_params={'M': 2},
                init_bounds=[[-5, 5], [-5, 5]],
                param_bounds=[-5, 5]
                )

sys_stl = System(name="stl",
                 state_vars=["x", "y"],
                 model=["a*x - w*y - x*(x**2 + y**2)",
                       "w*x + a*y - y*(x**2 + y**2)"],
                 model_params= {'a': 1, 'w': 3},
                 init_bounds=[[-5, 5], [-5, 5]],
                 param_bounds=[-5, 5]
                 )

sys_cphase = System(name="cphase",
                    state_vars=["x", "y"],
                    model=["A*sin(x) + B*sin(y) + W*sin(K*t) + R",
                           "C*sin(x) + D*sin(y) + E"],
                    model_params= {'A': 0.8, 'B': 0.8, 'W': 2.5, 'K': 2*np.pi*0.0015, 'R': 2,
                                   'C': 0, 'D': 0.6, 'E': 4.53},
                    init_bounds=[[-5, 5], [-5, 5]],
                    param_bounds=[-5, 5]
                    )

sys_lorenz = System(name="lorenz",
                    state_vars=["x", "y", "z"],
                    model=["S*(-x + y)",
                           "R*x - x*z - y",
                           "B*z + x*y"],
                    model_params= {'S': 10, 'R': 28, 'B': -8/3},
                    init_bounds=[[-5, 5], [-5, 5], [-5, 5]],
                    param_bounds=[-30, 30]
                    )


systems_collection = {
    sys_bacres.name: sys_bacres,
    sys_barmag.name: sys_barmag,
    sys_glider.name: sys_glider,
    sys_lv.name: sys_lv,
    sys_predprey.name: sys_predprey,
    sys_shearflow.name: sys_shearflow,
    sys_vdp.name: sys_vdp,
    sys_stl.name: sys_stl,
    sys_cphase.name: sys_cphase,
    sys_lorenz.name: sys_lorenz
}


