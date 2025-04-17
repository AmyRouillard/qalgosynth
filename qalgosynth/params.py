from qalgosynth.train_sets import (
    get_train_set_deutsch_jozsa,
    get_train_set_qft,
    get_train_set_grover,
)
from qalgosynth.duetsch_jozsa_oracle import DJ_oracle_naked
from qalgosynth.grover_oracle import U_oracle_naked


def get_params(task, size):
    if task not in ["qft", "deutsch_jozsa", "grover"]:
        raise ValueError(f"Task {task} not supported")
    if size not in ["small", "medium", "large", "huge"]:
        raise ValueError(f"Size {size} not supported")

    if task == "deutsch_jozsa":
        if size == "small":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("Hadamard", None, "H", None),
                        ("X",),
                        ("Z",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1"]]),
                "merge_within": [["!"]],
                "share_weights": [[False]],
                "edge_order": [[[1]]],
            }

            motif_probabilities = [
                0.0,
                0.0,
                0.4,
                0.4,
                0.2,
            ]  # "Qmask", "Qunmask", "Qcycle", "Qpivot", "Oracle"

        elif size == "medium":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("CZ",),
                        ("CNOT",),
                        ("MultiControlledX", None, "MCX", -1),
                        ("Hadamard", None, "H", None),
                        ("X",),
                        ("Z",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*"]]),
                "merge_within": [["!"]],
                "share_weights": [[False]],
                "edge_order": [[[1]]],
            }
            motif_probabilities = [0.1, 0.1, 0.3, 0.3, 0.2]

        elif size == "large":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("CZ",),
                        ("CY",),
                        ("CNOT",),
                        ("MultiControlledX", None, "MCX", -1),
                        ("Hadamard", None, "H", None),
                        ("X",),
                        ("Z",),
                        ("Y",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False]],
                "edge_order": [[[1]]],
            }

            motif_probabilities = [0.1, 0.1, 0.3, 0.3, 0.2]

        elif size == "huge":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("CZ",),
                        ("CY",),
                        ("CNOT",),
                        ("MultiControlledX", None, "MCX", -1),
                        ("Hadamard", None, "H", None),
                        ("X",),
                        ("Z",),
                        ("Y",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 2, 1)]],
                "step": [[x for x in range(1, 1 + 2, 1)]],
                "offset": [[x for x in range(2)]],
                "boundary": [["open", "periodic"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 2, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*", "!0", "0!"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False]],
                "edge_order": [[[1], [-1]]],
            }

            motif_probabilities = [0.1, 0.1, 0.3, 0.3, 0.2]

        params = {
            ### Run set up:
            # "initial_population_size": 500,
            # "initial_population_path": None,
            # "initial_motif_lengths": [1, 2, 3],
            # "initial_motif_length_probabilities": [0.2, 0.6, 0.2],
            # "batch_size": 4,
            # "max_pool": None,
            ### ProblemSetupMultiNQ
            "training_dict": {
                nq: get_train_set_deutsch_jozsa(nq) for nq in range(2, 5)
            },
            ### ProblemSetupBase
            "penalty_fn": lambda nq, n_edges, resultant_fidelities, oracle_calls: 0.1
            * (
                resultant_fidelities
                + 0.025 * n_edges / nq
                + 0.05 * oracle_calls * int(oracle_calls > 1)
            ),
            "reward_fn": lambda mean_fidelity: mean_fidelity,
            "correct_threshold": 0.99,
            ### EvolutionaryCircuitOptimizer
            "pressure": 0.05,  # 0.0001,
            "base_layout": lambda n=1: ["z"] * n,
            "additional_ancilla": [],
            "motif_probabilities": motif_probabilities,
            "oracle": DJ_oracle_naked,
            "discrete_variables": discrete_variables,
            "motif_size_limit": None,
        }

    elif task == "qft":
        if size == "small":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("ControlledPhaseShift", None, "CP", None),
                        ("Hadamard", None, "H", None),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False, True]],
                "edge_order": [[[1]]],
            }
        elif size == "medium":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("ControlledPhaseShift", None, "CP", None),
                        ("CRX",),
                        ("CRZ",),
                        ("CRY",),
                        ("RX",),
                        ("RZ",),
                        ("RZ",),
                        ("Hadamard", None, "H", None),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False, True]],
                "edge_order": [[[1]]],
            }

        elif size == "large":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("ControlledPhaseShift", None, "CP", None),
                        ("CRX",),
                        ("CRZ",),
                        ("CRY",),
                        ("RX",),
                        ("RZ",),
                        ("RZ",),
                        ("Hadamard", None, "H", None),
                        ("X",),
                        ("Z",),
                        ("Y",),
                        ("CNOT",),
                        ("CZ",),
                        ("CY",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False, True]],
                "edge_order": [[[1]]],
            }

        elif size == "huge":

            discrete_variables = {
                "mapping_names": [
                    [
                        ("ControlledPhaseShift", None, "CP", None),
                        ("CRX",),
                        ("CRZ",),
                        ("CRY",),
                        ("RX",),
                        ("RZ",),
                        ("RZ",),
                        ("Hadamard", None, "H", None),
                        ("X",),
                        ("Z",),
                        ("Y",),
                        ("CNOT",),
                        ("CZ",),
                        ("CY",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 2, 1)]],
                "step": [[x for x in range(1, 1 + 2, 1)]],
                "offset": [[x for x in range(2)]],
                "boundary": [["open", "periodic"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 2, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*", "!0", "0!"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False, True]],
                "edge_order": [[[1], [-1]]],
            }

        params = {
            ### Run set up:
            # "initial_population_size": [200],  # [250, 2500],
            # "initial_population_path": [None],
            # "initial_motif_lengths": [[1, 2, 3]],
            # "initial_motif_length_probabilities": [[0.2, 0.6, 0.2]],
            # "batch_size": [4],
            # "max_pool": [None],
            ### ProblemSetupMultiNQ
            "training_dict": {nq: get_train_set_qft(nq) for nq in range(2, 5)},
            ### ProblemSetupBase
            "penalty_fn": lambda nq, n_edges, num_symbols, resultant_fidelities: 0.1
            * (resultant_fidelities + 0.05 * (n_edges + num_symbols) / nq),
            "reward_fn": lambda mean_fidelity: mean_fidelity,
            "correct_threshold": 0.99,
            "train_lr": 0.05,
            "train_N": 100,
            "train_verbose": False,
            "train_tol": 1e-5,
            ### EvolutionaryCircuitOptimizer
            "pressure": 0.05,
            "base_layout": lambda n=1: ["z"] * n,
            "additional_ancilla": [],
            "motif_probabilities": [0.3, 0.1, 0.3, 0.3],
            "discrete_variables": discrete_variables,
            "motif_size_limit": None,
        }

    elif task == "grover":
        if size == "small":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("MultiControlledX", None, "MCX", -1),
                        ("Hadamard", None, "H", None),
                        ("X",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*"]]),
                "merge_within": [["!"]],
                "share_weights": [[False]],
                "edge_order": [[[1]]],
            }

            motif_probabilities = [0.0, 0.0, 0.4, 0.4, 0.2]

        elif size == "medium":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("MultiControlledX", None, "MCX", -1),
                        ("Hadamard", None, "H", None),
                        ("X",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False]],
                "edge_order": [[[1]]],
            }
            motif_probabilities = [0.1, 0.1, 0.3, 0.3, 0.2]

        elif size == "large":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("CZ",),
                        ("CY",),
                        ("CNOT",),
                        ("MultiControlledX", None, "MCX", -1),
                        ("Hadamard", None, "H", None),
                        ("X",),
                        ("Z",),
                        ("Y",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 1, 1)]],
                "step": [[x for x in range(1, 1 + 1, 1)]],
                "offset": [[x for x in range(1)]],
                "boundary": [["open"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False]],
                "edge_order": [[[1]]],
            }
            motif_probabilities = [0.1, 0.1, 0.3, 0.3, 0.2]

        elif size == "huge":
            discrete_variables = {
                "mapping_names": [
                    [
                        ("CZ",),
                        ("CY",),
                        ("CNOT",),
                        ("MultiControlledX", None, "MCX", -1),
                        ("Hadamard", None, "H", None),
                        ("X",),
                        ("Z",),
                        ("Y",),
                    ]
                ],
                "stride": [[x for x in range(1, 1 + 2, 1)]],
                "step": [[x for x in range(1, 1 + 2, 1)]],
                "offset": [[x for x in range(2)]],
                "boundary": [["open", "periodic"]],
                "strides": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 1, 1)],
                    [0],
                ],
                "steps": [
                    [x for x in range(1, 1 + 1, 1)],
                    [x for x in range(1, 1 + 2, 1)],
                    [1],
                ],
                "offsets": [[0], [0], [0]],
                "boundaries": [["open"], ["open"], ["periodic"]],
                "global_pattern": ([["*1", "1*", "!0", "0!"]]),
                "merge_within": [["*1", "1*", "!"]],
                "share_weights": [[False]],
                "edge_order": [[[1], [-1]]],
            }
            motif_probabilities = [0.1, 0.1, 0.3, 0.3, 0.2]

        params = {
            ### Run set up:
            # "initial_population_size": [2500],
            # "initial_population_path": [None],
            # "initial_motif_lengths": [[1, 2, 3]],
            # "initial_motif_length_probabilities": [[0.2, 0.6, 0.2]],
            # "batch_size": [4],
            # "max_pool": [None],
            ### ProblemSetupMultiNQ
            "training_dict": {
                nq: get_train_set_grover(
                    nq,
                )
                for nq in range(3, 6)
            },
            ### ProblemSetupBase
            "penalty_fn": lambda nq, n_edges, resultant_fidelities, oracle_calls: 0.01
            * (
                resultant_fidelities
                + 0.025 * n_edges / nq
                + 0.05 * oracle_calls * int(oracle_calls > 1)
            ),
            "reward_fn": lambda mean_fidelity: mean_fidelity,
            "correct_threshold": 0.90,
            ### EvolutionaryCircuitOptimizer
            "pressure": 0.05,
            "base_layout": lambda n=1: ["z"] * n,
            "additional_ancilla": [],
            "motif_probabilities": motif_probabilities,
            "oracle": U_oracle_naked,
            "discrete_variables": discrete_variables,
            "motif_size_limit": None,
        }

    return params
