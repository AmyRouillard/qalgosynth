# TODO:

# Note:
# Qmask is restricted to global_pattern only for mutation

import numpy as np
from collections import namedtuple
import inspect
from itertools import product

from qalgosynth.utils import get_unitary_from_penny_gate

import hierarqcal
from hierarqcal import (
    Qinit,
    Qhierarchy,
    Qunitary,
    Qmask,
    Qunmask,
    Qcycle,
    Qpermute,
    Qpivot,
    Qmotif,
    Qmotifs,
    get_tensor_as_f,
    plot_circuit,
)


class MotifPopulation:
    """
    Class to generate random motifs for the hierarchical quantum circuit architecture search.
    This should not have to be edited if the problem changes.

    Parameters:
    ----------
    base_layout (function): Function to generate the base layout of the quantum circuit.
    additional_ancilla (list): List of additional ancilla qubits to add to the circuit.
    motif_probabilities (list): List of probabilities for each motif to be selected.
    oracle (Qunitary): Oracle to be used in the circuit.
    discrete_variables (dict): Dictionary of discrete variables to be used in the motifs.

    Properties:
    ----------
    oracle (Qunitary): Oracle to be used in the circuit.
    available_motifs (list): List of available motifs to be used in the circuit.
    motif_probabilities (list): List of probabilities for each motif to be selected.
    base_layout (function): Function to generate the base layout of the quantum circuit.
    bit_layout (function): Function to generate the bit layout of the quantum circuit.
    additional_ancilla (list): List of additional ancilla qubits to add to the circuit.
    nq (function): Function to get the number of qubits in the circuit.
    discrete_variables (dict): Dictionary of discrete variables to be used in the motifs.
    hyperparam_functions (dict): Dictionary of functions to generate hyperparameters for the motifs.
    fixed_discrete_variables (list): List of fixed discrete variables.
    mutation_options (list): List of mutation options for the motifs.

    Public Methods:
    ---------------
    get_random_motif: Get a random motif from the available motifs.
    mutate_motif: Mutate a motif by changing a random property.


    """

    def __init__(
        self,
        base_layout=lambda n=1: ["az", "z"] * n,
        additional_ancilla=[],
        motif_probabilities=None,
        oracle=None,
        discrete_variables=None,
        random_number_generator=None,
        nq=None,
    ):

        self.oracle = oracle
        self.available_motifs = (
            ["Qmask", "Qunmask", "Qcycle", "Qpivot", "oracle"]
            if self.oracle is not None
            else ["Qmask", "Qunmask", "Qcycle", "Qpivot"]
        )
        self.motif_probabilities = motif_probabilities
        if random_number_generator is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = random_number_generator

        if self.motif_probabilities is not None and len(
            self.motif_probabilities
        ) != len(self.available_motifs):
            raise ValueError(
                "motif_probabilities must be same length as available_motifs, available_motifs: ",
                self.available_motifs,
            )

        self.base_layout = base_layout

        self.bit_layout = (
            lambda n=1: [
                f"{b}{i}" for i in range(n - 1, -1, -1) for b in self.base_layout()
            ]
            + additional_ancilla
        )

        self.additional_ancilla = additional_ancilla

        self.nq = lambda n: (
            len(self.bit_layout(n))
            if nq is None
            else lambda n=nq: len(self.bit_layout(n))
        )

        if discrete_variables is not None:
            self.discrete_variables = discrete_variables
        else:
            self.discrete_variables = {}

        mapping_names = (
            self.discrete_variables["mapping_names"][0]
            if "mapping_names" in list(discrete_variables.keys())
            else []
        )

        # mapping_names = [
        #     m(self.nq()) if isinstance(m, type(lambda x: x)) else m
        #     for m in mapping_names
        # ]

        available_mappings = [
            get_unitary_from_penny_gate(*args) for args in mapping_names
        ]

        # TODO: fix bug where "mapping" grows (parallelism?)
        # if "mapping" in self.discrete_variables.keys():
        #     self.discrete_variables["mapping"][0] += available_mappings
        # else:
        self.discrete_variables["mapping"] = [available_mappings]

        # Hyperparam_Info = namedtuple("Hyperparam_Info", ["function"])
        # self.hyperparam_functions = {
        #     key: Hyperparam_Info(
        #         lambda opts=options: (
        #             [opt[self.rng.choice(range(len(opt)))] for opt in opts]
        #             if len(opts) > 1
        #             else opts[0][self.rng.choice(range(len(opts[0])))]
        #         )
        #     )
        #     for key, options in self.discrete_variables.items()
        #     if key not in ["mapping_names"]
        # }

        # TODO combine motif_params_dict and N_opt into a single dict with a named tuple
        # Hyperparam_Info = namedtuple("Hyperparam_Info", ["param_names", "N_opts", "iterator"])

        self.motif_params_dict = {}
        for Q in self.available_motifs:
            if Q == "Qcycle":
                tmp = [
                    x
                    for x in inspect.signature(Qcycle.__init__).parameters
                    if x not in ["self", "kwargs"]
                ] + ["mapping", "share_weights", "edge_order"]
            elif Q == "Qpivot":
                tmp = [
                    x
                    for x in inspect.signature(Qpivot.__init__).parameters
                    if x not in ["self", "kwargs"]
                ] + ["mapping", "share_weights", "edge_order"]
            elif Q == "Qmask":
                # tmp = [
                #     x
                #     for x in inspect.signature(Qmask.__init__).parameters
                #     if x not in ["self", "kwargs"]
                # ] + ["mapping", "share_weights", "edge_order"]
                tmp = ["global_pattern"]
            elif Q == "Qunmask":
                tmp = []
            elif Q == "oracle":
                tmp = []
            else:
                raise ValueError(f"Q={Q} not recognized")

            tmp = [
                x for x in self.discrete_variables if x in tmp
            ]  # and not in self.fixed_discrete_variables

            ## TODO: *Debug (check if motif has prob>0 else skip)
            # if  self.motif_probabilities[self.available_motifs.index(Q)]>0:
            self.motif_params_dict[Q] = tmp

        self.fixed_discrete_variables = []
        for p in self.motif_params_dict.values():
            for key, options in self.discrete_variables.items():
                if key in p:
                    if (
                        len(
                            [
                                x[0] if len(x) == 1 else list(x)
                                for x in product(*options)
                            ]
                        )
                        == 1
                    ):
                        self.fixed_discrete_variables.append(key)
            # Some fixed options might not be default so do not remove them,
            # but mutate will ignore them
            # self.motif_params_dict[n] = [x for x in q if x not in fixed_discrete_variables]

        self.mutation_options_func = lambda motif_name: [
            x
            for x in self.motif_params_dict[motif_name]
            if x not in self.fixed_discrete_variables
        ]

        self.N_opts = {}
        for n, p in self.motif_params_dict.items():
            self.N_opts[n] = [
                len([x[0] if len(x) == 1 else list(x) for x in product(*options)])
                for key, options in self.discrete_variables.items()
                if key in p
            ]

        self.param_pad = max(
            [max([len(str(x)) for x in v] + [0]) for v in self.N_opts.values()]
        )
        self.id_pad = max([len(v) for v in self.N_opts.values()]) * self.param_pad

        id_iterator = [[range(n) for n in v] for v in self.N_opts.values()]

        self.id_dict = {}
        for ind, n, iter in zip(
            range(len(self.available_motifs)), self.available_motifs, id_iterator
        ):

            # TODO: *Debug
            if self.motif_probabilities[ind] > 0:
                v_params = [
                    [
                        x[0] if len(x) == 1 else list(x)
                        for x in product(*self.discrete_variables[p])
                    ]
                    for p in self.motif_params_dict[n]
                ]

                for x in product(*iter):
                    tmp = "".join(
                        ["0" * (self.param_pad - len(str(i))) + str(i) for i in x]
                    )
                    tmp = "0" * (self.id_pad - len(tmp)) + tmp

                    value = [v[i] for i, v in zip(x, v_params)]

                    self.id_dict[str(ind) + tmp] = value

        if self.oracle is not None:
            self.oracle_id = (
                str(self.available_motifs.index("oracle")) + "0" * self.id_pad
            )

    def get_motif_from_id(self, id_tuple):

        motif_params = [
            (
                (self.available_motifs[int(id[0])], self.id_dict[id])
                if id in self.id_dict.keys()
                else None
            )
            for id in id_tuple
        ]

        if None in motif_params:
            return None
        else:
            motif = Qmotifs()
            for m in motif_params:
                motif_name = m[0]
                if motif_name == "Qunmask":
                    kwargs = {"global_pattern": "previous"}
                    motif += getattr(hierarqcal, motif_name)(
                        **kwargs,
                    )
                elif motif_name == "oracle":
                    motif += self.oracle
                else:
                    kwargs = dict(
                        zip(
                            self.motif_params_dict[motif_name],
                            m[1],
                        )
                    )
                    motif += getattr(hierarqcal, motif_name)(
                        **kwargs,
                    )

            return motif

    def __get_motif_params__(self, motif):

        motif_name = motif.__class__.__name__

        if motif_name == "Qunmask":
            return {"global_pattern": motif.global_pattern}
        # elif motif_name == "Qmask":
        #     return {"global_pattern": motif.global_pattern}
        else:
            motif_params = {
                hyperparam: vars(motif)[hyperparam]
                for hyperparam in self.mutation_options_func[motif_name]
                # set(
                #     [x for x in inspect.signature(motif.__init__).parameters]
                #     + ["mapping", "share_weights", "edge_order"]
                # )
                # & set([k for k in self.hyperparam_functions.keys()])
            }

        return motif_params

    def __get_base_motifs__(self, max_motifs=200):

        motif_type = [self.available_motifs[int(m[0])] for m in self.id_dict.keys()]
        # if len(motif_type) > max_motifs:
        #     raise ValueError(
        #         f"Number of motifs {len(motif_type)} is greater than max_motifs {max_motifs}. Increase max_motifs to at least {len(motif_type)}."
        #     )
        motif_type_count = [motif_type.count(m) for m in self.available_motifs]
        motif_count = [p * max_motifs for p in self.motif_probabilities]

        # check that motif_type_count < motif_count for all motifs
        if not all([m <= n for m, n in zip(motif_type_count, motif_count)]):
            # raise warning instead of error
            print(
                f"Number of motifs {motif_type_count} is greater than max_motifs {motif_count}. Adjust motif probabilities or max number of motifs."
            )
            # raise ValueError(
            #     f"Number of motifs {motif_type_count} is greater than max_motifs {motif_count}. Adjust motif probabilities or max number of motifs."
            # )

        # new_count = []
        motifs = []
        for i, m in enumerate(self.available_motifs):
            new_motifs = [
                k for k in self.id_dict.keys() if self.available_motifs[int(k[0])] == m
            ]
            N = int(motif_count[i] // motif_type_count[i])
            r = int(motif_count[i] - N * motif_type_count[i])
            if r > 0:
                additionsl_motifs = list(self.rng.choice(new_motifs, r))
            else:
                additionsl_motifs = []
            new_motifs = new_motifs * int(N) + additionsl_motifs
            new_motifs = [Qmotifs() + (m,) for m in new_motifs]
            # new_count += [len(new_motifs)]
            motifs.extend(new_motifs)

        # probs = [x/len(motifs) for x in new_count]
        # assert np.allclose(probs, self.motif_probabilities)

        return motifs

    def get_random_motif(
        self,
        motif_name=None,
    ):

        if motif_name is None:
            motif_name = np.random.choice(
                self.available_motifs, p=self.motif_probabilities
            )

        i = self.available_motifs.index(motif_name)
        motifs = [m for m in self.id_dict.keys() if m[0] == str(i)]
        motif_id = self.rng.choice(motifs)

        # id_iterator = [range(n) for n in self.N_opts[motif_name]]
        # tmp = "".join(
        #     [
        #         "0" * (self.param_pad - len(str(i))) + str(i)
        #         for i in [self.rng.choice(x) for x in id_iterator]
        #     ]
        # )
        # motif_id = str(i) + "0" * (self.id_pad - len(tmp)) + tmp

        return Qmotifs() + (motif_id,)

    def mutate_motif(self, motif_id, property=None):
        """
        Mutate motif by randomly selecting a property to change and changing it to a new value.
        A new randomly generated motif is returned if the input motif is a:
        - Qunmask
        - Oracle
        - A property to mutate cannot be found

        Parameters:
        ----------
        motif (Qmotif): Motif to be mutated.

        """

        if len(motif_id) != self.id_pad + 1:
            raise ValueError(f"motif_id must be length {self.id_pad + 1}")

        motif_name = self.available_motifs[int(motif_id[0])]
        motif_properties = motif_id[1:]

        if property is not None and property not in self.motif_params_dict[motif_name]:
            raise ValueError(f"{property} is not a property of {motif_name}")

        if motif_name == "Qunmask":
            return self.get_random_motif()
        elif motif_name == "oracle":
            return self.get_random_motif()
        else:
            if property is None:
                property = self.rng.choice(self.mutation_options_func(motif_name))
            property_ind = self.motif_params_dict[motif_name].index(property)

            id_iterator = [range(n) for n in self.N_opts[motif_name]]
            offset = (
                self.id_pad - len(self.motif_params_dict[motif_name]) * self.param_pad
            )
            old_value = motif_properties[
                property_ind + offset : property_ind + offset + self.param_pad
            ]
            new_value = self.rng.choice(
                [x for x in id_iterator[property_ind] if x != int(old_value)]
            )

            new_motif_id = (
                motif_id[0]
                + motif_properties[: property_ind + offset]
                + "0" * (self.param_pad - len(str(new_value)))
                + str(new_value)
                + motif_properties[property_ind + offset + self.param_pad :]
            )

            return Qmotifs() + (new_motif_id,)
