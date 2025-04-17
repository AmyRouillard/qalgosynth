from qalgosynth.main import run
import numpy as np
import os
from qalgosynth.params import get_params

scheduler = (
    lambda generation: (np.cos(generation / 300 * np.pi) + 1) / 2 * (1 - 0.25) + 0.25
)
scheduler_p = (
    lambda generation: (1 - np.cos(generation / 300 * np.pi)) / (2) * (0.25 - 0.05)
    + 0.05
)

if __name__ == "__main__":

    N_cpus = os.cpu_count()  # number of CPUs to use for parallel processing
    n_runs = 1  # number of runs to perform (sequentially)

    run_type = "grover"  # "deutsch_jozsa"  # "qft"  #
    run_size = "small"  # "medium"  # "large"  # "huge" #

    output_dir = f"~/Desktop/"  # output directory for the results

    # setting to true will make the run time shorter and the number of batches smaller
    debug = True  # False

    #####################################

    params = get_params(run_type, run_size)

    initial_motif_length = 2
    if run_type == "deutsch_jozsa" or run_type == "grover":
        if run_size == "small":
            time_out = 60 * 60 * 2
        elif run_size == "medium":
            time_out = 60 * 60 * 4
        elif run_size == "large":
            time_out = 60 * 60 * 8
        elif run_size == "huge":
            time_out = 60 * 60 * 8
    elif run_type == "qft":
        if run_size == "small":
            time_out = 60 * 60 * 4
        elif run_size == "medium":
            time_out = 60 * 60 * 8
        elif run_size == "large":
            time_out = 60 * 60 * 12
        elif run_size == "huge":
            time_out = 60 * 60 * 12
            initial_motif_length = 1

    for j in range(0, n_runs):

        if run_type == "deutsch_jozsa":
            repetition_range_list = [1 for k in params["training_dict"].keys()]
        else:
            repetition_range_list = [k for k in params["training_dict"].keys()]

        fixed_params = {
            "path_out": lambda text, run_size="", run_j="": output_dir
            + f"{text}_{N_cpus}/{run_size}/{run_j}/",
            "max_generations": None,
            "time_out": 60 * 10 if debug else time_out,
            "parallel": True,
            "n_workers": N_cpus,
            "repetition_range_list": repetition_range_list,
            "stopping_threshold": 0.99,
            "parameterized": True if run_type in ["qft"] else False,
            "seed": 42,
            "initial_population_path": None,
            "allow_duplicates": True,
            "exp_exp_tradeoff_scheduler": scheduler,
            "pressure_scheduler": scheduler_p,
            "initial_population_size": 200,
            "initial_motif_length": initial_motif_length,
            "batch_size": (
                2 if debug else 20
            ),  # 20 batches used with 14 cpus, adjust based on available cpus
            "max_pool": None,
        }

        # update params with fixed params
        params.update(fixed_params)

        if params["n_workers"] == 1 and params["parallel"]:
            print("Warning: n_workers=1, but parallel=True. Setting parallel=False.")
            params["parallel"] = False

        run(params, run_type, run_size, f"run_{j}", overwrite=True)
