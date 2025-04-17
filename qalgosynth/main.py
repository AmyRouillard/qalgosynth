## The "correct_threshold" determines when a prediction for a given training point is considered correct,
## i.e. |<y|y\hat>| > "correct_threshold". This determines the "percentage_correct".
## The "fitness" can take other factors into account, such as the fidelity, length and number of symbols used in the circuit.
## Stopping condition for the genetic algorithm is when the best individual has a "percentage_correct" greater than
## the "stopping_threshold".
## The reason for this is that the appropriate "stopping_threshold" for the fitness is not always evident.
## To optimism for "fitness" set "stopping_threshold" to None and use the "time_out" parameter to stop the genetic algorithm.


import os
import dill
import inspect
import time
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from qalgosynth.evolutionary_algorithm import EvolutionaryCircuitOptimizer
from qalgosynth.problemsetup import (
    ProblemSetupBase,
    ProblemSetupParameterized,
    ProblemSetupMultiNQ,
)
from qalgosynth.utils import __add_nodes_edges__

# from hierarqcal import (
#     Qinit,
#     plot_circuit,
# )


def post_processing(
    params,
    motif,
    best_ind,
    population,
    path_out,
    clean=True,
    generation=None,
):

    population_dict = {g.id: g for g in population}
    if clean:

        if params["parameterized"]:

            # params["train_N"] = 200
            # params["train_tol"] = 1e-6
            # params["train_verbose"] = True
            keys_p = [
                x
                for x in inspect.signature(
                    ProblemSetupParameterized.__init__
                ).parameters
                if x != "self"
            ]
        else:
            keys_p = [
                x
                for x in inspect.signature(ProblemSetupBase.__init__).parameters
                if x != "self"
            ]

        keys_pm = [
            x
            for x in inspect.signature(ProblemSetupMultiNQ.__init__).parameters
            if x != "self"
        ]

        # initialise the problem setup multi nq with keys_pm and kwargs for problem setup base
        p = ProblemSetupMultiNQ(
            **{k: params[k] for k in keys_pm if k in params},
            **{k: params[k] for k in keys_p if k in params},
        )

        cleaned = p.clean_up_motif(
            motif=motif,
            motif_id=best_ind.genotype,
            attribute_threshold=getattr(best_ind, "fitness"),
            attribute="fitness",
            repetitions_list=best_ind.repetitions_list,
            id=best_ind.id,
            generation=generation,
        )
    else:
        cleaned = None

    # nq = 3
    # target = "1" * nq
    # hierq = Qinit(len(params["additional_ancilla"]) + len(params["base_layout"](nq)))

    if cleaned is None:
        history, all_parents = __add_nodes_edges__(
            best_ind, [], [best_ind], population_dict
        )
        # for m in best_ind.genotype:
        #     if isinstance(m, type(lambda x: x)):
        #         hierq += m(target)
        #     else:
        #         hierq += m
    else:
        history, all_parents = __add_nodes_edges__(
            cleaned, [], [cleaned], population_dict
        )

        # for m in cleaned.genotype:
        #     if isinstance(m, type(lambda x: x)):
        #         hierq += m(target)
        #     else:
        #         hierq += m

    info_dict = {
        "id": best_ind.id if cleaned is None else cleaned.id,
        "edge_tuples": history,
        "sub_population": all_parents,
    }

    # fig, ax = plot_circuit(hierq, plot_width=25)
    # fig.savefig(path_out + "imgs/best_circuit.png")

    with open(path_out + f"evolution_tree_best.dill", "wb") as f:
        dill.dump(info_dict, f)


def run(
    params,
    run_name="test",
    params_i="",
    run_j="",
    rng=np.random.default_rng(),
    overwrite=False,
):
    """
    Example of minimum params:
    params = {
        "path_out": lambda run_name, params_i, run_j: f"../data/{run_name}/{params_i}/{run_j}/",
        "max_generations": 10,
        "time_out": None,
        "stopping_threshold": None,
        "initial_population_size": 10,
        "initial_population_path": None,
        "batch_size": 10,
        "max_pool": 10,
        "parallel": True,
        "n_workers": 10,
        "repetition_range_list": [2, 3],
        "parameterized": False,
    }
    """

    params["run_name"] = run_name

    path_out = params["path_out"](params["run_name"], params_i, run_j)

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    if not os.path.exists(path_out + "population/"):
        os.makedirs(path_out + "population/")
    if not os.path.exists(path_out + "imgs/"):
        os.makedirs(path_out + "imgs/")

    if os.path.exists(path_out + "params.dill") and not overwrite:
        # load params
        with open(path_out + "params.dill", "rb") as f:
            params = dill.load(f)
        print("Overwrite warning: Params loaded from file.")
    else:
        with open(path_out + "params.dill", "wb") as f:
            dill.dump(params, f)

    population = []
    previous_generation = 0
    previous_generations = []
    for file in os.listdir(path_out + "population/"):
        previous_generations.append(int(file.split("_")[1].split(".")[0]))
        with open(path_out + "population/" + file, "rb") as f:
            population.extend(dill.load(f))

    if len(previous_generations) > 0:
        previous_generation = max(previous_generations)

    keys_e = [
        x
        for x in inspect.signature(EvolutionaryCircuitOptimizer.__init__).parameters
        if x != "self"
    ]
    e = EvolutionaryCircuitOptimizer(
        population=population,
        random_number_generator=rng,
        **{k: params[k] for k in keys_e if k in params},
    )

    if params["parameterized"]:
        keys_p = [
            x
            for x in inspect.signature(ProblemSetupParameterized.__init__).parameters
            if x != "self"
        ]
    else:
        keys_p = [
            x
            for x in inspect.signature(ProblemSetupBase.__init__).parameters
            if x != "self"
        ]

    keys_pm = [
        x
        for x in inspect.signature(ProblemSetupMultiNQ.__init__).parameters
        if x != "self"
    ]

    # initialise the problem setup multi nq with keys_pm and kwargs for problem setup base
    p = ProblemSetupMultiNQ(
        **{k: params[k] for k in keys_pm if k in params},
        **{k: params[k] for k in keys_p if k in params},
    )

    def evaluate_genotype_wrapper(kwargs):
        return p.evaluate_genotype(**kwargs)

    additional_evaluation_params = {
        "repetition_range_list": params["repetition_range_list"],
    }

    allow_duplicates = params["allow_duplicates"]
    max_generations = params["max_generations"]
    time_out = params["time_out"]
    threshold = params["stopping_threshold"]
    fields = [
        "generation",
        "creation_time",
        "evaluation_time",
        "save_time",
        "current_runtime",
        "population_size",
        "percentage_unique",
        "motif_seq",
    ]

    ### If no population is loaded from file, generate a random population
    start_ = time.time()
    if len(e.population) == 0:
        # if "initial_motif_lengths" not in params:
        #     lengths = [1] * params["initial_population_size"]
        # else:
        #     lengths = rng.choice(
        #         params["initial_motif_lengths"],
        #         size=params["initial_population_size"],
        #         p=params["initial_motif_length_probabilities"],
        #     )
        # initial_population = e.get_genotypes_random(
        #     lengths,
        #     parent_id=("0:seed",),
        #     **additional_evaluation_params,
        # )

        # initial_population = e.get_genotypes_unit_length(
        #     params["initial_population_size"],
        #     parent_id=("0:seed",),
        #     **additional_evaluation_params,
        # )

        initial_population = e.get_genotypes_length_n(
            params["initial_population_size"],
            n=params["initial_motif_length"],
            parent_id=("0:seed",),
            **additional_evaluation_params,
        )

        # load the initial population from a file
        if params["initial_population_path"] is not None:
            with open(params["initial_population_path"], "rb") as f:
                addition_population = dill.load(f)

            for ind in addition_population:
                tmp = {
                    "motif": e.get_motif_from_id(ind.genotype),
                    "motif_id": ind.genotype,
                    "parent_ids": (
                        "0:ext",
                        ind.id,
                    ),
                }
                tmp.update(additional_evaluation_params)

                initial_population.append(tmp)

        initial_population_unevaluated = []
        initial_population_evaluated = []
        current_motif_ids = []
        for ind in initial_population:
            # check is the genotype is already in the population
            index = e.population_dict.get("".join(ind["motif_id"]), None)
            if "".join(ind["motif_id"]) in current_motif_ids:
                initial_population_evaluated.append(ind)
            else:
                initial_population_unevaluated.append(ind)
                current_motif_ids.append("".join(ind["motif_id"]))
    creation_time = time.time() - start_

    ### Evaluate the initial population
    start = time.time()
    if len(e.population) == 0:
        if params["parallel"]:
            with Pool(params["n_workers"]) as pool:
                results = pool.map(
                    evaluate_genotype_wrapper, initial_population_unevaluated
                )
        else:
            results = []
            for ind in initial_population_unevaluated:
                results.append(evaluate_genotype_wrapper(ind))

        results = [x for x in results if x is not None]
        e.update_population(results)

        if allow_duplicates:
            results = []
            for ind in initial_population_evaluated:
                genotype_tmp = e.population[e.population_dict["".join(ind["motif_id"])]]
                results.append(p.copy_eval(genotype_tmp, ind["parent_ids"]))
            e.population.extend(results)

    evaluation_time = time.time() - start

    ### Save the initial population
    start = time.time()
    if previous_generation == 0:
        # save the whole initial population
        with open(path_out + f"population/population_{0}.dill", "wb") as f:
            dill.dump(e.population, f)
    save_time = time.time() - start

    ### Get current best individual
    best_ind = sorted(e.population, key=lambda x: x.fitness)[-1]
    ### Print info of the current run status
    info = [
        previous_generation,
        creation_time,
        evaluation_time,
        save_time,
        time.time() - start_,
        len(e.population),
        len(e.population_dict) / len(e.population),
        "".join([x[0] for x in best_ind.genotype]),
    ]  # + [f"{getattr(best_ind, x)}" for x in p.attributes]
    att = [getattr(best_ind, x) for x in p.attributes]
    att = [f"{a:.4f}" if isinstance(a, float) else a for a in att]
    att = [
        (
            [round(x, 4) for x in a]
            if isinstance(a, list) and isinstance(a[0], float)
            else a
        )
        for a in att
    ]
    att = [f'"{a}"' if isinstance(a, list) else a for a in att]
    att = [str(a) for a in att]
    info = [f"{i:.4f}" if isinstance(i, float) else f"{i}" for i in info] + att
    # text = "\t".join([f"{x}: {y}\n" for x, y in zip(fields + p.attributes, info)])
    # print(text)

    ### Create a csv file to store the results
    if not os.path.exists(path_out + "results.csv"):
        with open(path_out + "results.csv", "w") as f:
            f.write(",".join(fields + p.attributes) + "\n" + ",".join(info) + "\n")

    generation = 1 + previous_generation

    if os.path.exists(path_out + f"population/population_{generation}.dill"):
        raise ValueError(f"population/population_{generation}.dill exists.")

    # del (
    #     initial_population,
    #     initial_population_unevaluated,
    #     initial_population_evaluated,
    #     current_motif_ids,
    # )

    while True:
        start = time.time()
        new_genotypes = []
        for _ in range(params["batch_size"]):
            new_genotypes.extend(
                e.get_genotypes_tournament(
                    generation=generation,
                    exp_exp_tradeoff=params["exp_exp_tradeoff_scheduler"](generation),
                    pressure=params["pressure_scheduler"](generation),
                    max_pool=params["max_pool"],
                    **additional_evaluation_params,
                )
            )

            # for _ in range(2):  # generate two random motifs
            #     # Add a random motif of a random length determined by the best individual
            #     length = rng.integers(1, best_ind.length) if best_ind.length > 1 else 1
            #     new_genotypes.extend(
            #         e.get_genotypes_random(
            #             [length],
            #             parent_id=(str(generation) + ":seed", best_ind.id),
            #             **additional_evaluation_params,
            #         )
            #     )
        population_unevaluated = []
        population_evaluated = []
        current_motif_ids = []
        for ind in new_genotypes:
            # check is the genotype is already in the population
            index = e.population_dict.get("".join(ind["motif_id"]), None)
            if index is not None:
                population_evaluated.append(ind)
            elif "".join(ind["motif_id"]) in current_motif_ids:
                pass  # don't add copies in the new batch to the pool
            else:
                population_unevaluated.append(ind)
                current_motif_ids.append("".join(ind["motif_id"]))
        creation_time = time.time() - start

        start = time.time()
        if params["parallel"]:
            with Pool(params["n_workers"]) as pool:
                results = pool.map(evaluate_genotype_wrapper, population_unevaluated)
        else:
            results = [evaluate_genotype_wrapper(ind) for ind in population_unevaluated]
        results = [x for x in results if x is not None]
        e.update_population(results)

        results_add = []
        if allow_duplicates:
            for ind in population_evaluated:
                genotype_tmp = e.population[e.population_dict["".join(ind["motif_id"])]]
                results_add.append(p.copy_eval(genotype_tmp, ind["parent_ids"]))
            e.population.extend(results_add)

        evaluation_time = time.time() - start

        start = time.time()
        # save only the new data
        with open(path_out + f"population/population_{generation}.dill", "wb") as f:
            dill.dump(results + results_add, f)
        save_time = time.time() - start

        # write progress to results.csv
        best_ind = sorted(e.population, key=lambda x: x.fitness)[-1]
        info = [
            generation,
            creation_time,
            evaluation_time,
            save_time,
            time.time() - start_,
            len(e.population),
            len(e.population_dict) / len(e.population),
            "".join([x[0] for x in best_ind.genotype]),
        ]  # + [f"{getattr(best_ind, x)}" for x in p.attributes]
        att = [getattr(best_ind, x) for x in p.attributes]
        att = [f"{a:.4f}" if isinstance(a, float) else a for a in att]
        att = [
            (
                [round(x, 4) for x in a]
                if isinstance(a, list) and isinstance(a[0], float)
                else a
            )
            for a in att
        ]
        att = [f'"{a}"' if isinstance(a, list) else a for a in att]
        att = [str(a) for a in att]
        info = [f"{i:.4f}" if isinstance(i, float) else f"{i}" for i in info] + att

        # append info to the csv file
        with open(path_out + "results.csv", "a") as f:
            f.write(",".join(info) + "\n")

        if (
            threshold is not None
            and (
                best_ind.percentage_correct
                >= threshold  # best_ind.fitness >= threshold and
            )
            and (params["parameterized"] or max(best_ind.oracle_calls_list) == 1)
        ):

            text = "\t".join(
                [f"{x}: {y}\n" for x, y in zip(fields + p.attributes, info)]
            )
            print(text)
            print("Post processing...")
            try:
                post_processing(
                    params,
                    e.get_motif_from_id(best_ind.genotype),
                    best_ind,
                    e.population,
                    path_out,
                    generation + 1,
                )
            except Exception as e:
                print("Bug in post processing.")
                print(e)

            break

        if time_out is not None and time.time() - start_ > time_out:
            print(f"Timeout after {time_out} seconds.")

            try:
                post_processing(
                    params,
                    e.get_motif_from_id(best_ind.genotype),
                    best_ind,
                    e.population,
                    path_out,
                    clean=False,
                )
            except Exception as e:
                print("Bug in post processing.")
                print(e)
            break

        if max_generations is not None and generation >= max_generations:
            print(f"Max generations {max_generations} reached.")

            try:
                post_processing(
                    params,
                    e.get_motif_from_id(best_ind.genotype),
                    best_ind,
                    e.population,
                    path_out,
                    clean=False,
                )
            except Exception as e:
                print("Bug in post processing.")
                print(e)

            break

        if generation % 1000 == 0:
            try:
                post_processing(
                    params,
                    e.get_motif_from_id(best_ind.genotype),
                    best_ind,
                    e.population,
                    path_out,
                    clean=False,
                )
            except Exception as e:
                print("Bug in post processing.")
                print(e)

        generation += 1
