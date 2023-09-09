from samplers.dpm_solver import DPM_Solver

def dpm_solver(model):
    DPM_Solver(model_fn=model, noise_schedule=None, 
                                algorithm_type="dpmsolver++")  
def dpm_solver_plus(model):
    DPM_Solver(model_fn=model, noise_schedule=None, 
                                algorithm_type="dpmsolver")