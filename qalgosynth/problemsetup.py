import numpy as np
from collections import namedtuple
import uuid
import inspect
import pennylane as qml

from collections import Counter
import torch
from torch.func import grad
import time

from qalgosynth.utils import JSD_uniform_t, JSD_uniform_np

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


class ProblemSetupMultiNQ:
    def __init__(self, parameterized, training_dict, **kwargs):
        self.training_dict = training_dict
        self.parameterized = parameterized

        if self.parameterized:
            self.problem_setups = {
                nq: ProblemSetupParameterized(
                    train_set=self.training_dict[nq], **kwargs
                )
                for nq in training_dict.keys()
            }
        else:
            self.problem_setups = {
                nq: ProblemSetupBase(train_set=self.training_dict[nq], **kwargs)
                for nq in training_dict.keys()
            }

        self.GenotypeInformation = namedtuple(
            "Task_Information",
            [
                "fitness",  # required
                "percentage_correct",  # required
                "fitness_list",
                "mean_fidelity_list",
                "percentage_correct_list",
                "length",  # required
                "n_edges_list",
                "num_symbols_list",
                "oracle_calls_list",
                "repetitions_list",
                "id",  # required
                "parent_ids",  # required
                "genotype",  # required
                "symbols_list",
            ],
        )

        exclude_log = [
            # "fitness",  # required
            # "percentage_correct",  # required
            # "fitness_list",
            "mean_fidelity_list",
            "percentage_correct_list",
            # "length",
            "n_edges_list",
            "num_symbols_list",
            "oracle_calls_list",
            # "repetitions_list",
            # "id",  # required
            # "parent_ids",  # required
            "genotype",  # required
            "symbols_list",
        ]
        self.attributes = [
            f"{x}" for x in self.GenotypeInformation._fields if x not in exclude_log
        ]

    def evaluate_genotype(
        self,
        motif=None,
        motif_id=None,
        parent_ids=None,
        repetitions_list=None,
        repetition_range_list=None,
        symbols_list=None,
        symbols_0_list=None,
    ):
        if motif is None:
            return None

        if repetitions_list is None:
            repetitions_list = [None] * len(self.problem_setups.keys())

        if repetition_range_list is None:
            repetition_range_list = [None] * len(self.problem_setups.keys())

        if self.parameterized:
            if symbols_list is None:
                symbols_list = [None] * len(self.problem_setups.keys())

            if symbols_0_list is None:
                symbols_0_list = [None] * len(self.problem_setups.keys())

            evaluations = [
                self.problem_setups[nq].evaluate_genotype(
                    motif=motif,
                    motif_id=motif_id,
                    parent_ids=parent_ids,
                    repetitions=repetitions,
                    repetition_range=repetition_range,
                    symbols=symbols,
                    symbols_0=symbols_0,
                )
                for nq, repetitions, repetition_range, symbols, symbols_0 in zip(
                    self.problem_setups.keys(),
                    repetitions_list,
                    repetition_range_list,
                    symbols_list,
                    symbols_0_list,
                )
            ]
        else:
            evaluations = [
                self.problem_setups[nq].evaluate_genotype(
                    motif=motif,
                    motif_id=motif_id,
                    parent_ids=parent_ids,
                    repetitions=repetitions,
                    repetition_range=repetition_range,
                )
                for nq, repetitions, repetition_range in zip(
                    self.problem_setups.keys(),
                    repetitions_list,
                    repetition_range_list,
                )
            ]

        # if any evaluation is None, return None
        if any([e is None for e in evaluations]):
            return None

        genotype_id = uuid.uuid4().hex

        genotype = self.GenotypeInformation(
            fitness=np.min([e.fitness for e in evaluations]),
            percentage_correct=np.min([e.percentage_correct for e in evaluations]),
            fitness_list=[e.fitness for e in evaluations],
            mean_fidelity_list=[e.mean_fidelity for e in evaluations],
            percentage_correct_list=[e.percentage_correct for e in evaluations],
            length=evaluations[0].length,
            n_edges_list=[e.n_edges for e in evaluations],
            num_symbols_list=[
                e.num_symbols if hasattr(e, "num_symbols") else None
                for e in evaluations
            ],
            oracle_calls_list=[
                e.oracle_calls if hasattr(e, "oracle_calls") else None
                for e in evaluations
            ],
            repetitions_list=[e.repetitions for e in evaluations],
            id=genotype_id,
            parent_ids="--".join(parent_ids) if parent_ids is not None else None,
            genotype=motif_id,
            symbols_list=[
                e.symbols if hasattr(e, "symbols") else None for e in evaluations
            ],
        )

        return genotype

    def clean_up_motif(
        self,
        motif,
        motif_id,
        attribute_threshold,
        attribute,
        repetitions_list=None,
        repetition_range_list=None,
        id=None,
        generation=None,
    ):

        if generation is not None:
            desc = str(generation) + ":cleaned"
        else:
            desc = "?:cleaned"

        best_motif = motif
        best_motif_id = motif_id

        best_out = None
        n = len(best_motif) - 2
        for i in range(n):
            new_motif = Qmotifs()
            # skip the i and i+1 motifs
            for j, m in enumerate(best_motif):
                if j != i and j != i + 1:
                    new_motif += m
            new_motif_id = Qmotifs()
            for j, m in enumerate(best_motif_id):
                if j != i and j != i + 1:
                    new_motif_id += (m,)

            out = self.evaluate_genotype(
                motif=new_motif,
                motif_id=new_motif_id,
                parent_ids=(desc,) if id is None else (desc, id),
                repetitions_list=repetitions_list,
                repetition_range_list=repetition_range_list,
                symbols_list=None,
                symbols_0_list=None,
            )

            if out is not None and getattr(out, attribute) >= attribute_threshold:
                best_motif = new_motif
                best_motif_id = new_motif_id
                attribute_threshold = getattr(out, attribute)
                best_out = out

        n = len(best_motif) - 1
        for i in range(n):
            new_motif = Qmotifs()
            # skip the i and i+1 motifs
            for j, m in enumerate(best_motif):
                if j != i:
                    new_motif += m

            new_motif_id = Qmotifs()
            for j, m in enumerate(best_motif_id):
                if j != i:
                    new_motif_id += (m,)

            out = self.evaluate_genotype(
                motif=new_motif,
                motif_id=new_motif_id,
                parent_ids=(desc,) if id is None else (desc, id),
                repetitions_list=repetitions_list,
                repetition_range_list=repetition_range_list,
                symbols_list=None,
                symbols_0_list=None,
            )

            if out is not None and getattr(out, attribute) >= attribute_threshold:
                best_motif = new_motif
                best_motif_id = new_motif_id
                attribute_threshold = getattr(out, attribute)
                best_out = out

        return best_out

    def copy_eval(self, genotype, parent_ids=None):
        genotype_id = uuid.uuid4().hex

        return self.GenotypeInformation(
            fitness=genotype.fitness,
            percentage_correct=genotype.percentage_correct,
            fitness_list=genotype.fitness_list,
            mean_fidelity_list=genotype.mean_fidelity_list,
            percentage_correct_list=genotype.percentage_correct_list,
            length=genotype.length,
            n_edges_list=genotype.n_edges_list,
            num_symbols_list=genotype.num_symbols_list,
            oracle_calls_list=genotype.oracle_calls_list,
            repetitions_list=genotype.repetitions_list,
            id=genotype_id,
            parent_ids="--".join(parent_ids) if parent_ids is not None else None,
            genotype=genotype.genotype,
            symbols_list=genotype.symbols_list,
        )


class ProblemSetupParameterized:
    """
    ProblemSetupBase is a class that defines the problem setup for the evolutionary algorithm.
    The user must define the following methods:
    - evaluate_genotype (callable): A function that takes a genotype as input and returns a namedtuple GenotypeInformation.

    The user must also define the following attributes:
    - GenotypeInformation (namedtuple): A namedtuple that defines the information that is stored for each genotype. The namedtuple should contain at least the following fields:
        - fitness (float): The fitness of the genotype.
        - id (str): A unique identifier for the genotype.
        - genotype (list): The genotype of the circuit.
    - attributes

    Args:
    - length_penalty_fn (callable): A function that takes the length of the genotype as input and returns a penalty. (Penalty is subtracted from fitness. Default is 0.)
    - oracle_penalty_fn (callable): A function that takes the number of oracle calls as input and returns a penalty. (Penalty is subtracted from fitness. Default is 0.)
    - train_set (dict): A dictionary of the form {0:{'input':input_state_vector, 'output':output_state_vector}. Training points can be for varying numbers of qubits, the correct circuit size is automatically determined. Output state vectors must be a list of state vectors. (Default is None.) From the train set the attributes self.input_state_vectors and self.output_state_vectors.
    - exclude_log (list): A list of attributes to exclude from the log. (Default is ['genotype'].)

    Important notes:
    - We assume that the train set uses a fixed number of qubits for all input state vectors.
    - does not work with lambda funtions, i.e. oracles. (No kwargs in train_set.)

    """

    def __init__(
        self,
        penalty_fn=None,
        reward_fn=lambda mean_fidelity: mean_fidelity,
        train_set=None,
        correct_threshold=0.95,
        train_lr=0.05,
        train_N=200,
        train_verbose=False,
        train_tol=1e-6,
        time_limit=10,
    ):

        self.penalty_fn = penalty_fn
        self.reward_fn = reward_fn
        self.reward_fn_params = inspect.signature(self.reward_fn).parameters
        if self.penalty_fn is None:
            self.penalty_fn_params = []
        else:
            self.penalty_fn_params = inspect.signature(self.penalty_fn).parameters

        self.correct_threshold = correct_threshold

        self.GenotypeInformation = namedtuple(
            "Task_Information",
            [
                "fitness",  # required
                "mean_fidelity",
                "percentage_correct",
                "length",
                "n_edges",
                "num_symbols",
                "repetitions",
                "id",  # required
                "parent_ids",  # required
                "genotype",  # required
                "symbols",
            ],
        )

        exclude_log = ["genotype", "symbols"]

        self.attributes = [
            f"{x}" for x in self.GenotypeInformation._fields if x not in exclude_log
        ]

        if train_set is not None:
            if isinstance(train_set, dict):
                # TODO: check that train_set matches the train_set format
                self.input_state_vectors = [
                    train_set[k]["input"] for k in train_set.keys()
                ]
                self.output_state_vectors = [
                    train_set[k]["output"] for k in train_set.keys()
                ]
            else:
                raise ValueError(
                    "train_set must be a dictionary of the form:\n{0:{'input':input_state_vector, 'output':output_state_vector, 'kwargs':{'x':'0'}}...}"
                )

            nqs = [int(np.log2(len(input))) for input in self.input_state_vectors]
            # check all nqs are the same
            if len(set(nqs)) == 1:
                self.nq = nqs[0]
            else:
                raise ValueError("All input state vectors must have the same length.")

        else:
            self.input_state_vectors = None
            self.output_state_vectors = None
            self.nq = None

        self.train_lr = train_lr
        self.train_N = train_N
        self.train_verbose = train_verbose
        self.train_tol = train_tol
        self.time_limit = time_limit

    def __get_circuit__(self, circ, input_state):
        """
        Get the circuit from the circuit description.

        Parameters
        ----------
        circ : Hierarqcal object
            Circuit description.
        input_state : torch.tensor
            Input state vector.

        Returns
        -------
        qnode : callable
            Circuit as a qnode.
        """
        dev = qml.device("default.qubit.torch", wires=circ.tail.Q)

        @qml.qnode(dev, interface="torch")
        def circuit():

            # set input state
            qml.QubitStateVector(input_state, wires=circ.tail.Q)

            # execute the circuit
            circ(backend="pennylane")

            return qml.state()

        return circuit

    def __fitness__(self, kwargs):
        """
        Compute the fitness of the motif.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the reward and penalty functions.

        Returns
        -------
        fitness : float
            Fitness of the motif.
        """
        reward_args = []
        penalty_args = []
        # Extract required arguments for reward function
        for param in self.reward_fn_params:
            if param not in kwargs:
                raise KeyError(f"Missing required key for reward function: {param}")
            reward_args.append(kwargs[param])

        # Extract required arguments for penalty function
        for param in self.penalty_fn_params:
            if param not in kwargs:
                raise KeyError(f"Missing required key for penalty function: {param}")

            if param == "resultant_fidelities":
                penalty_args.append(JSD_uniform_t(kwargs[param]))
            else:
                penalty_args.append(kwargs[param])

        if self.penalty_fn is not None:

            return self.reward_fn(*reward_args) - self.penalty_fn(*penalty_args)
        else:
            return self.reward_fn(*reward_args)

    def __get_resultant_state__(self, circ, input_state):
        """
        Get the resultant vector state for the given input state. Use initialise circuit as input, not the motif.

        Parameters
        ----------
        circ : Qinit object
            Circuit description.
        input_state : torch.tensor
            Input state vector.

        Returns
        -------
        resultant_state : torch.tensor
            Resultant state vector.
        """
        try:
            resultant_state = self.__get_circuit__(circ, input_state)()
            return resultant_state
        except:
            return None

    def __get_n_edges__(self, nq, motif):
        """
        Get the number of gates (edges) in the circuit.

        Parameters
        ----------
        nq : int
            Number of qubits.
        motif : tuple
            Circuit motif.
        """
        circ = Qinit(nq) + motif  # + Qunmask("all")
        for _ in range(
            sum([isinstance(x, Qmask) for x in circ])
            - sum([isinstance(x, Qunmask) for x in circ])
        ):
            circ += Qunmask("previous")

        n_edges = 0
        for m in circ:
            n_edges += len(m.E)

        return n_edges

    def __evaluate__(self, circ):
        """
        Evaluate the initilised circuit for all training data.

        Parameters
        ----------
        circ : Hierarqcal object
            Circuit description that has been Qinit'ed

        Returns
        -------
        results_fid : torch.tensor
            Fidelities of the resultant state vectors.
        """

        results_fid = torch.zeros(len(self.input_state_vectors))
        count = 0
        for input, output in zip(self.input_state_vectors, self.output_state_vectors):
            resultant_state = self.__get_resultant_state__(circ, input)

            # convert output to torch tensor
            # output = [torch.tensor(out, dtype=torch.complex128) for out in output]
            # convert output using as_tensor

            if resultant_state is not None:

                if isinstance(output, list):
                    output = [
                        torch.as_tensor(out, dtype=torch.complex128) for out in output
                    ]
                else:
                    output = [torch.as_tensor(output, dtype=torch.complex128)]

                # inner_products = torch.zeros(len(output), dtype=torch.complex128)
                # for i, o in enumerate(output):
                #     inner_products[i] = torch.vdot(o, resultant_state)
                # results_fid[count] = torch.sqrt(
                #     torch.vdot(inner_products, inner_products).real
                # )
                results_fid[count] = torch.square(
                    torch.abs(torch.vdot(output[0], resultant_state))
                )
                for o in output[1:]:
                    results_fid[count] = torch.add(
                        results_fid[count],
                        torch.square(torch.abs(torch.vdot(o, resultant_state))),
                    )
            else:
                results_fid[count] = -torch.tensor(float("inf"))
            count += 1

        return results_fid

    def evaluate_genotype(
        self,
        motif=None,
        motif_id=None,
        parent_ids=None,
        repetitions=None,
        repetition_range=None,
        symbols=None,
        symbols_0=None,
        rng=np.random.default_rng(42),
    ):

        if motif is None:
            return None

        if self.input_state_vectors is None:
            raise ValueError("train_set must be defined to evaluate genotypes.")

        length = len(Qmotifs() + motif)

        if repetitions is None:
            repetitions = 1

        mask_count = 0
        for m in motif:
            if isinstance(m, Qmask):
                mask_count += 1
            if isinstance(m, Qunmask):
                mask_count -= 1
        mask_count = max(0, mask_count)

        # motif_1 = motif + Qunmask("all")
        motif_1 = motif + (Qunmask("previous"),) * mask_count
        circ_1 = Qinit(self.nq) + motif_1
        symb_freq = Counter(circ_1.get_symbols())
        num_symbols = len(symb_freq.keys())
        n_edges = 0
        for m in circ_1:
            n_edges += len(m.E)

        # if Qmask of Qunmask are edges present?
        # if n_edges == 0:
        #     return None

        # assuming the symbols are the same for all repetitions
        # populate_symbols = (
        #     lambda symbols, n: [
        #         symbols[list(symb_freq.keys()).index(s)] for s in circ_1.get_symbols()
        #     ]
        #     * n
        # )
        # try:

        # Important to unmask only at the very end of the circuit not in between repetitions!
        # motif_n = motif * repetitions + Qunmask("all")
        motif_n = (
            motif * repetitions + (Qunmask("previous"),) * mask_count * repetitions
        )
        circ_n = Qinit(self.nq) + motif_n
        # assuming the symbols are NOT the same for all repetitions
        symb_freq_n = Counter(circ_n.get_symbols())
        num_symbols_n = len(symb_freq_n.keys())
        populate_symbols = lambda symbols, n: [
            symbols[list(symb_freq_n.keys()).index(s)] for s in circ_n.get_symbols()
        ]
        if num_symbols > 0:
            if symbols is None:
                symbols_trained = self.train(
                    motif=motif,
                    n_symbols=num_symbols_n,
                    n_edges=n_edges,
                    repetitions=repetitions,
                    populate_symbols=populate_symbols,
                    symbols=(
                        symbols_0[: len(symb_freq_n)] if symbols_0 is not None else None
                    ),
                    rng=rng,
                )
            else:
                symbols_trained = torch.as_tensor(symbols)
            all_symbols = populate_symbols(symbols_trained, repetitions)
            circ_n.set_symbols(all_symbols)
            results_fid = self.__evaluate__(circ_n)
        else:
            symbols_trained = None
            results_fid = self.__evaluate__(circ_n)

        if torch.isnan(results_fid).any():
            return None

        # mean_fidelity = torch.mean(results_fid).item()
        # convert to numpy array
        # results_fid = results_fid.detach().numpy()
        # mean_fidelity = np.mean(results_fid)
        # percentage_correct = len(
        #     [f for f in results_fid if f > self.correct_threshold]
        # ) / len(results_fid)

        mean_fidelity = torch.mean(results_fid)

        percentage_correct = torch.mean((results_fid > self.correct_threshold).float())

        kwargs = {
            "resultant_fidelities": results_fid,
            "mean_fidelity": mean_fidelity,
            "percentage_correct": percentage_correct,
            "length": length,
            "n_edges": n_edges,
            "num_symbols": num_symbols,
            "repetitions": repetitions,
            "nq": self.nq,
        }
        fitness = self.__fitness__(kwargs)

        # if torch.isnan(fitness).any():
        #     return None

        genotype_id = uuid.uuid4().hex

        genotype = self.GenotypeInformation(
            fitness=fitness.item(),
            mean_fidelity=mean_fidelity.item(),
            percentage_correct=percentage_correct.item(),
            length=length,
            n_edges=n_edges,
            num_symbols=num_symbols,
            repetitions=repetitions,
            id=genotype_id,
            parent_ids="--".join(parent_ids) if parent_ids is not None else None,
            genotype=motif_id,
            symbols=(symbols_trained.detach() if symbols_trained is not None else None),
        )

        if repetition_range is not None and repetition_range > 1:
            for reps in range(2, repetition_range + 1):
                genotype_tmp = self.evaluate_genotype(
                    motif=motif,
                    motif_id=motif_id,
                    parent_ids=parent_ids,
                    repetitions=reps,
                    symbols_0=(
                        torch.cat([genotype.symbols.detach() for _ in range(reps)])
                        if genotype.symbols is not None
                        else None
                    ),
                )
                if genotype_tmp is None:
                    break
                else:
                    if genotype_tmp.fitness > genotype.fitness:
                        genotype = genotype_tmp
                    else:
                        return genotype
            return genotype
        else:
            return genotype

    def objective_function(
        self, motif, num_symbols, n_edges, repetitions, populate_symbols, symbols
    ):
        length = len(Qmotifs() + motif)

        circ = Qinit(self.nq) + motif * repetitions
        all_symbols = populate_symbols(symbols, repetitions)
        circ.set_symbols(all_symbols)

        results_fid = self.__evaluate__(circ)

        mean_fidelity = torch.mean(
            results_fid
        )  # torch.abs(torch.sum(results_fid))/len(self.input_state_vectors) #

        percentage_correct = torch.mean(
            (results_fid > self.correct_threshold).float()
        ).requires_grad_(True)
        # percentage_correct_tensor = percentage_correct.clone().detach().requires_grad_(True)

        # fn = grad(self.__fitness__, argnums=(0, 1, 2, 3), has_aux=True)
        # loss = - fn(mean_fidelity, percentage_correct, length, num_symbols)

        kwargs = {
            "resultant_fidelities": results_fid,
            "mean_fidelity": mean_fidelity,
            "percentage_correct": percentage_correct,
            "length": length,
            "n_edges": n_edges,
            "num_symbols": num_symbols,
            "repetitions": repetitions,
            "nq": self.nq,
        }
        loss = -self.__fitness__(kwargs)

        # mean_fidelity.retain_grad()
        # percentage_correct.retain_grad()

        # loss.backward()

        # if torch.isnan(loss).any():
        #     print("loss", loss)
        # if torch.isnan(symbols.grad).any():
        #     print("symbols", symbols)
        # if torch.isnan(mean_fidelity.grad).any():
        #     print("mean_fidelity", mean_fidelity)
        # if torch.isnan(percentage_correct.grad).any():
        #     print("percentage_correct", percentage_correct)

        return loss

    def train(
        self,
        motif,
        n_symbols,
        n_edges,
        repetitions,
        populate_symbols,
        symbols=None,
        rng=np.random.default_rng(42),
    ):

        # run time limit 1min

        # store tmp loss to compare if it is below tol, set initial loss to inf
        loss_tmp = torch.tensor(float("inf"))

        attempt = 0
        max_attempts = 10
        if symbols is None:
            # symbols = torch.rand(n_symbols, requires_grad=True)
            # start at a small value
            # symbols = rng.uniform(0, 2 * np.pi * 0.01, n_symbols)
            symbols = rng.uniform(-np.pi / 2, np.pi / 2, n_symbols)
            symbols = torch.as_tensor(symbols).requires_grad_(True)
        else:
            # copy of symbols
            symbols = torch.as_tensor(symbols).requires_grad_(True)
        if len(symbols) != n_symbols:
            # symbols = torch.rand(n_symbols, requires_grad=True)
            # start at a small value
            symbols = rng.uniform(0, 2 * np.pi * 0.01, n_symbols)
            symbols = torch.as_tensor(symbols).requires_grad_(True)

        # start_time
        start_time = time.time()
        opt = torch.optim.Adam([symbols], lr=self.train_lr)
        for it in range(self.train_N):
            opt.zero_grad()
            loss = self.objective_function(
                motif, n_symbols, n_edges, repetitions, populate_symbols, symbols
            )
            loss.backward()
            # tmp_grad = symbols.grad.clone().detach()
            tmp = symbols.clone().detach()
            opt.step()

            # check if symbols contain nan
            if torch.isnan(symbols).any():
                # break out of training and set sybmols to tmp
                symbols = tmp.clone().detach()
                # print(f"At step {it}: Nan in symbols, break")
                if it - attempt > 0 or attempt > max_attempts:
                    break
                else:
                    attempt += 1
                    # symbols = torch.rand(n_symbols, requires_grad=True)
                    symbols = rng.uniform(0, 2 * np.pi, n_symbols)
                    symbols = torch.as_tensor(symbols).requires_grad_(True)
                    # print(f"Attempt {attempt}")

            if self.train_verbose:
                if it % 10 == 0:
                    print(f"Loss at step {it}: {loss}")

            if torch.abs(loss - loss_tmp) < self.train_tol:
                break
            loss_tmp = loss

            if time.time() - start_time > self.time_limit:
                break

        return symbols

    def __get_resultant_states__(self, motif, symbols=None, threshold=1e-10):

        mask_count = 0
        for m in motif:
            if isinstance(m, Qmask):
                mask_count += 1
            if isinstance(m, Qunmask):
                mask_count -= 1
        mask_count = max(0, mask_count)

        circ = Qinit(self.nq) + motif
        symb_freq = Counter(circ.get_symbols())
        populate_symbols = lambda sym: [
            sym[list(symb_freq.keys()).index(s)] for s in circ.get_symbols()
        ]
        if symbols is not None:
            all_symbols = populate_symbols(symbols)
            circ.set_symbols(all_symbols)
        else:
            if len(symb_freq) > 0:
                raise ValueError("Symbols must be provided.")

        res_states_bits = []
        for r in [
            self.__get_resultant_state__(circ, input)
            for input in self.input_state_vectors
        ]:
            tmp = []
            for s in range(2**self.nq):
                if np.abs(r[s]) > threshold:
                    tmp.append(f"{format(s, f'0{self.nq}b')}: {r[s]:.2f}")
            res_states_bits.append(tmp)

        return res_states_bits

    def clean_up_motif(
        self,
        motif,
        motif_id,
        attribute_threshold,
        attribute,
        repetitions=None,
        repetition_range=None,
        id=None,
        generation=None,
    ):
        """
        Cleaning procedure for a motif:
        1. Skip the i-th and i+1-th motif and evaluate the fitness of the resulting circuit.
        2. Accept the new circuit if the attribute is above the threshold, update the threshold.
        3. Skip the i-th motif and evaluate the fitness of the resulting circuit.
        4. Accept the new circuit if the attribute is above the threshold, update the threshold.
        """
        if generation is not None:
            desc = str(generation) + ":cleaned"
        else:
            desc = "?:cleaned"

        best_motif = motif
        best_motif_id = motif_id
        best_out = None
        n = len(best_motif) - 2
        for i in range(n):
            new_motif = Qmotifs()
            # skip the i and i+1 motifs
            for j, m in enumerate(best_motif):
                if j != i and j != i + 1:
                    new_motif += m

            new_motif_id = Qmotifs()
            for j, m in enumerate(best_motif_id):
                if j != i and j != i + 1:
                    new_motif_id += (m,)

            out = self.evaluate_genotype(
                motif=new_motif,
                motif_id=new_motif_id,
                parent_ids=(desc,) if id is None else (desc, id),
                repetitions=repetitions,
                repetition_range=repetition_range,
                symbols=None,
                symbols_0=None,
            )

            if out is not None and getattr(out, attribute) >= attribute_threshold:
                best_motif = new_motif
                best_motif_id = new_motif_id
                attribute_threshold = getattr(out, attribute)
                best_out = out

        n = len(best_motif) - 1
        for i in range(n):
            new_motif = Qmotifs()
            # skip the i and i+1 motifs
            for j, m in enumerate(best_motif):
                if j != i:
                    new_motif += m

            new_motif_id = Qmotifs()
            for j, m in enumerate(best_motif_id):
                if j != i:
                    new_motif_id += (m,)

            out = self.evaluate_genotype(
                motif=new_motif,
                motif_id=new_motif_id,
                parent_ids=(desc,) if id is None else (desc, id),
                repetitions=repetitions,
                repetition_range=repetition_range,
                symbols=None,
                symbols_0=None,
            )

            if out is not None and getattr(out, attribute) >= attribute_threshold:
                best_motif = new_motif
                best_motif_id = new_motif_id
                attribute_threshold = getattr(out, attribute)
                best_out = out

        return best_out

    def copy_eval(self, genotype, parent_ids=None):

        genotype_id = uuid.uuid4().hex

        return self.GenotypeInformation(
            fitness=genotype.fitness,
            mean_fidelity=genotype.mean_fidelity,
            percentage_correct=genotype.percentage_correct,
            length=genotype.length,
            n_edges=genotype.n_edges,
            num_symbols=genotype.num_symbols,
            repetitions=genotype.repetitions,
            id=genotype_id,
            parent_ids="--".join(parent_ids) if parent_ids is not None else None,
            genotype=genotype.genotype,
            symbols=genotype.symbols,
        )


class ProblemSetupBase:
    """
    ProblemSetupBase is a class that defines the problem setup for the evolutionary algorithm.
    The user must define the following methods:
    - evaluate_genotype (callable): A function that takes a genotype as input and returns a namedtuple GenotypeInformation.


    The user must also define the following attributes:
    - GenotypeInformation (namedtuple): A namedtuple that defines the information that is stored for each genotype. The namedtuple should contain at least the following fields:
        - fitness (float): The fitness of the genotype.
        - id (str): A unique identifier for the genotype.
        - genotype (list): The genotype of the circuit.
    - attributes

    Args:
    - length_penalty_fn (callable): A function that takes the length of the genotype as input and returns a penalty. (Penalty is subtracted from fitness. Default is 0.)
    - oracle_penalty_fn (callable): A function that takes the number of oracle calls as input and returns a penalty. (Penalty is subtracted from fitness. Default is 0.)
    - train_set (dict): A dictionary of the form {0:{'input':input_state_vector, 'output':output_state_vector, 'kwargs':{'x':'0'}}...}. Training points can be for varying numbers of qubits, the correct circuit size is automatically determined. Output state vectors must be a list of state vectors. (Default is None.) From the train set the attributes self.input_state_vectors, self.output_state_vectors, and self.eval_kwargs are constructed.
    - exclude_log (list): A list of attributes to exclude from the log. (Default is ['genotype'].)

    """

    def __init__(
        self,
        penalty_fn=None,
        reward_fn=lambda mean_fidelity: mean_fidelity,
        train_set=None,
        correct_threshold=0.95,
    ):

        self.penalty_fn = penalty_fn
        self.reward_fn = reward_fn
        self.reward_fn_params = inspect.signature(self.reward_fn).parameters
        self.penalty_fn_params = (
            inspect.signature(self.penalty_fn).parameters
            if self.penalty_fn is not None
            else []
        )

        self.correct_threshold = correct_threshold

        self.GenotypeInformation = namedtuple(
            "Task_Information",
            [
                "fitness",  # required
                "mean_fidelity",  # required
                "percentage_correct",  # required
                "length",
                "n_edges",
                "oracle_calls",
                "repetitions",
                "id",  # required
                "parent_ids",  # required
                "genotype",  # required
            ],
        )

        exclude_log = ["genotype"]
        self.attributes = [
            f"{x}" for x in self.GenotypeInformation._fields if x not in exclude_log
        ]

        if train_set is not None:
            if isinstance(train_set, dict):
                # TODO: check that train_set matches the train_set format
                self.input_state_vectors = [
                    train_set[k]["input"] for k in train_set.keys()
                ]
                self.output_state_vectors = [
                    train_set[k]["output"] for k in train_set.keys()
                ]
                self.eval_kwargs = (
                    [train_set[k]["kwargs"] for k in train_set.keys()]
                    if "kwargs" in train_set[0].keys()
                    else [{} for _ in train_set.keys()]
                )

                if "expected_fidelity" in train_set[0].keys():
                    self.expected_fidelity = [
                        train_set[k]["expected_fidelity"] for k in train_set.keys()
                    ]
                else:
                    self.expected_fidelity = None
            else:
                raise ValueError(
                    "train_set must be a dictionary of the form:\n{0:{'input':input_state_vector, 'output':output_state_vector, 'kwargs':{'x':'0'}}...}"
                )

            nqs = [int(np.log2(len(input))) for input in self.input_state_vectors]
            # check all nqs are the same
            if len(set(nqs)) == 1:
                self.nq = nqs[0]
            else:
                raise ValueError("All input state vectors must have the same length.")

        else:
            self.input_state_vectors = None
            self.output_state_vectors = None
            self.eval_kwargs = None
            self.expected_fidelity = None

            self.nq = None

    def __get_circuit__(self, circ, input_state):
        """
        Get the circuit from the circuit description.

        Parameters
        ----------
        circ : Hierarqcal object
            Circuit description.
        input_state : torch.tensor
            Input state vector.

        Returns
        -------
        qnode : callable
            Circuit as a qnode.
        """
        dev = qml.device("default.qubit.torch", wires=circ.tail.Q)

        @qml.qnode(dev, interface="torch")
        def circuit():

            # set input state
            qml.QubitStateVector(input_state, wires=circ.tail.Q)

            # execute the circuit
            circ(backend="pennylane")

            return qml.state()

        return circuit

    def __fitness__(self, kwargs):
        """
        Compute the fitness of the motif.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the reward and penalty functions.

        Returns
        -------
        fitness : float
            Fitness of the motif.
        """
        reward_args = []
        penalty_args = []
        # Extract required arguments for reward function
        for param in self.reward_fn_params:
            if param not in kwargs:
                raise KeyError(f"Missing required key for reward function: {param}")
            reward_args.append(kwargs[param])

        # Extract required arguments for penalty function
        for param in self.penalty_fn_params:
            if param not in kwargs:
                raise KeyError(f"Missing required key for penalty function: {param}")

            if param == "resultant_fidelities":
                penalty_args.append(JSD_uniform_np(kwargs[param]))
            else:
                penalty_args.append(kwargs[param])
        if self.penalty_fn is not None:
            return self.reward_fn(*reward_args) - self.penalty_fn(*penalty_args)
        else:
            return self.reward_fn(*reward_args)

    def __get_resultant_state__(self, motif, input_state, **kwargs):
        """
        Use motif as input, circuit is initialized with nq given by the input_state.
        """
        try:

            circ = Qinit(self.nq)
            for m in motif:
                if isinstance(m, type(lambda x: x)):
                    circ += m(**kwargs)
                else:
                    circ += m
            for _ in range(
                sum([isinstance(x, Qmask) for x in circ])
                - sum([isinstance(x, Qunmask) for x in circ])
            ):
                circ += Qunmask("previous")

            resultant_state = self.__get_circuit__(circ, input_state)()
            return resultant_state
        except:
            return None

    def __get_n_edges__(self, nq, motif, kwargs):
        circ = Qinit(nq)
        count = 0
        for m in motif:
            if isinstance(m, type(lambda x: x)):
                # circ += m(**kwargs)
                # don't count edges in oracle
                pass
            else:
                circ += m
                count += 1

        n_edges = 0
        if count > 0:
            for _ in range(
                sum([isinstance(x, Qmask) for x in circ])
                - sum([isinstance(x, Qunmask) for x in circ])
            ):
                circ += Qunmask("previous")

            for m in circ:
                n_edges += len(m.E)

        return n_edges

    def __evaluate__(self, motif):

        results_fid = []
        for input, output, kwargs in zip(
            self.input_state_vectors,
            self.output_state_vectors,
            self.eval_kwargs,
        ):
            resultant_state = self.__get_resultant_state__(motif, input, **kwargs)

            if resultant_state is not None:
                # inner_products = np.zeros(len(output), dtype=complex)
                # for i, o in enumerate(output):
                #     inner_products[i] = np.vdot(o, resultant_state)
                # fid = np.sqrt(np.vdot(inner_products, inner_products).real)
                fid = np.abs(np.vdot(output, resultant_state)) ** 2  #
            else:
                fid = None
            results_fid.append(fid)

            # if fid < 1:
            #     pass

        if self.expected_fidelity is not None:
            results_fid = [
                np.abs(1.0 - ef - fid) if fid is not None else None
                for fid, ef in zip(results_fid, self.expected_fidelity)
            ]

        return results_fid

    def evaluate_genotype(
        self,
        motif=None,
        motif_id=None,
        parent_ids=None,
        repetitions=None,
        repetition_range=None,
        rng=np.random.default_rng(42),
    ):

        if motif is None:
            return None

        if self.input_state_vectors is None:
            raise ValueError("train_set must be defined to evaluate genotypes.")

        n_edges = self.__get_n_edges__(self.nq, motif, self.eval_kwargs[0])

        # if Qmask of Qunmask are edges present?
        # # if n_edges contains zero, the circuit is invalid
        # if 0 == n_edges:
        #     return None

        oracle_calls = 0
        for m in motif:
            if isinstance(m, type(lambda x: x)):
                oracle_calls += 1

        length = len(Qmotifs() + motif)

        if repetitions is None:
            repetitions = 1

        results_fid = self.__evaluate__(motif * repetitions)

        if None in results_fid:
            return None

        mean_fidelity = np.mean(
            results_fid
        )  # np.abs(np.sum(results_fid))/len(self.input_state_vectors) #
        percentage_correct = len(
            [f for f in results_fid if f > self.correct_threshold]
        ) / len(results_fid)

        kwargs = {
            "resultant_fidelities": results_fid,
            "mean_fidelity": mean_fidelity,
            "percentage_correct": percentage_correct,
            "length": length,
            "n_edges": n_edges,
            "oracle_calls": oracle_calls,
            "repetitions": repetitions,
            "nq": self.nq,
        }
        fitness = self.__fitness__(kwargs)

        # # check for nan in fitness
        # if np.isnan(fitness):
        #     return None

        genotype_id = uuid.uuid4().hex

        genotype = self.GenotypeInformation(
            fitness=fitness,
            mean_fidelity=mean_fidelity,
            percentage_correct=percentage_correct,
            length=length,
            n_edges=n_edges,
            oracle_calls=oracle_calls,
            repetitions=repetitions,
            id=genotype_id,
            parent_ids="--".join(parent_ids) if parent_ids is not None else None,
            genotype=motif_id,
        )

        if repetition_range is not None and repetition_range > 1:
            for reps in range(2, repetition_range + 1):
                genotype_tmp = self.evaluate_genotype(
                    motif=motif,
                    motif_id=motif_id,
                    parent_ids=parent_ids,
                    repetitions=reps,
                    rng=rng,
                )
                if genotype_tmp is None:
                    break
                else:
                    if genotype_tmp.fitness > genotype.fitness:
                        genotype = genotype_tmp
                    else:
                        return genotype
            return genotype
        else:
            return genotype

    def __get_resultant_states__(self, motif, symbols=None, threshold=1e-10):
        nqs = []
        res_states = []
        for input, kwargs in zip(self.input_state_vectors, self.eval_kwargs):
            resultant_state = self.__get_resultant_state__(motif, input, **kwargs)
            res_states.append(resultant_state)
            nqs.append(int(np.log2(len(input))))

        res_states_bits = []
        for r, total_nq in zip(res_states, nqs):
            tmp = []
            for s in range(2**total_nq):
                if np.abs(r[s]) > threshold:
                    tmp.append(f"{format(s, f'0{total_nq}b')}: {r[s]:.2f}")
            res_states_bits.append(tmp)

        return res_states_bits

    def clean_up_motif(
        self,
        motif,
        motif_id,
        attribute_threshold,
        attribute,
        repetitions=None,
        repetition_range=None,
        id=None,
        generation=None,
    ):

        if generation is not None:
            desc = str(generation) + ":cleaned"
        else:
            desc = "?:cleaned"

        """
        Cleaning procedure for a motif:
        1. Skip the i-th and i+1-th motif and evaluate the fitness of the resulting circuit.
        2. Accept the new circuit if the attribute is above the threshold, update the threshold.
        3. Skip the i-th motif and evaluate the fitness of the resulting circuit.
        4. Accept the new circuit if the attribute is above the threshold, update the threshold.
        """

        best_motif = motif
        best_motif_id = motif_id

        best_out = None
        n = len(best_motif) - 2
        for i in range(n):
            new_motif = Qmotifs()
            # skip the i and i+1 motifs
            for j, m in enumerate(best_motif):
                if j != i and j != i + 1:
                    new_motif += m

            new_motif_id = Qmotifs()
            for j, m in enumerate(best_motif_id):
                if j != i and j != i + 1:
                    new_motif_id += (m,)

            out = self.evaluate_genotype(
                motif=new_motif,
                motif_id=motif_id,
                parent_ids=(desc,) if id is None else (desc, id),
                repetitions=repetitions,
                repetition_range=repetition_range,
                symbols=None,
                symbols_0=None,
            )

            if out is not None and getattr(out, attribute) >= attribute_threshold:
                best_motif = new_motif
                best_motif_id = new_motif_id
                attribute_threshold = getattr(out, attribute)
                best_out = out

        n = len(best_motif) - 1
        for i in range(n):
            new_motif = Qmotifs()
            # skip the i and i+1 motifs
            for j, m in enumerate(best_motif):
                if j != i:
                    new_motif += m

            new_motif_id = Qmotifs()
            for j, m in enumerate(best_motif_id):
                if j != i:
                    new_motif_id += (m,)

            out = self.evaluate_genotype(
                motif=new_motif,
                motif_id=motif_id,
                parent_ids=(desc,) if id is None else (desc, id),
                repetitions=repetitions,
                repetition_range=repetition_range,
                symbols=None,
                symbols_0=None,
            )

            if out is not None and getattr(out, attribute) >= attribute_threshold:
                best_motif = new_motif
                best_motif_id = new_motif_id
                attribute_threshold = getattr(out, attribute)
                best_out = out

        return best_out

    def copy_eval(self, genotype, parent_ids=None):
        genotype_id = uuid.uuid4().hex

        return self.GenotypeInformation(
            fitness=genotype.fitness,
            mean_fidelity=genotype.mean_fidelity,
            percentage_correct=genotype.percentage_correct,
            length=genotype.length,
            n_edges=genotype.n_edges,
            oracle_calls=genotype.oracle_calls,
            repetitions=genotype.repetitions,
            id=genotype_id,
            parent_ids="--".join(parent_ids) if parent_ids is not None else None,
            genotype=genotype.genotype,
        )
