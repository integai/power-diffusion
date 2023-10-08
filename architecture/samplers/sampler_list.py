from samplers.dpm_solver import DPM_Solver
def sampler_choose(model=None, sampler=str()):
    if sampler == "dpm_solver":
        return dpm_solver(model_fn=model)
    elif sampler == "dpm_solver_plus":
        return dpm_solver_plus(model_fn=model)
def dpm_solver(model):
    DPM_Solver(model_fn=model, noise_schedule=None, 
                                algorithm_type="dpmsolver++")  
def dpm_solver_plus(model):
    DPM_Solver(model_fn=model, noise_schedule=None, 
                                algorithm_type="dpmsolver")