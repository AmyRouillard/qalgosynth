# %%
import sympy as sp
import numpy as np

from hierarqcal import (
    Qinit,
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

from qalgosynth.utils import get_unitary_from_penny_gate

H = get_unitary_from_penny_gate("Hadamard")
X = get_unitary_from_penny_gate("X")

Cp = get_unitary_from_penny_gate("CPhase", [-np.pi / 2], "CP-", 2)
Cpd = get_unitary_from_penny_gate("CPhase", [np.pi / 2], "CP+", 2)

Cnot = get_unitary_from_penny_gate("CNOT")
Cz = get_unitary_from_penny_gate("CZ")

U_toffoli = get_unitary_from_penny_gate("Toffoli")

multiCZ = (
    Qpivot(mapping=X, global_pattern="1*")
    + Qcycle(mapping=U_toffoli, step=2, boundary="open")
    + Qpivot(mapping=Cz, global_pattern="*1", merge_within="11")
    + Qcycle(mapping=U_toffoli, step=2, boundary="open", edge_order=[-1])
    + Qpivot(mapping=X, global_pattern="1*")
)


U_d = (
    Qpivot(mapping=H, global_pattern="01")
    + Qpivot(mapping=X, global_pattern="01")
    + multiCZ
    + Qpivot(mapping=X, global_pattern="01")
    + Qpivot(mapping=H, global_pattern="01")
)

# U_d = (
#     Qmask("10")
#     + Qcycle(mapping=H)
#     + Qcycle(mapping=X)
#     + Qunmask("previous")
#     + multiCZ
#     + Qmask("10")
#     + Qcycle(mapping=X)
#     + Qcycle(mapping=H)
#     + Qunmask("previous")
# )

U_oracle_naked = lambda target_string: (
    Qmask("10")  # mask ancilla
    + Qpivot(
        mapping=X,
        global_pattern="".join(
            ["0" if x == "1" else "1" for x in target_string]
        ),  # [::-1],
    )  # |target_string> -> |11...1>
    + Qunmask("previous")
    + multiCZ  # |11...1> -> -|11...1>
    + Qmask("10")
    + Qpivot(
        mapping=X,
        global_pattern="".join(
            ["0" if x == "1" else "1" for x in target_string]
        ),  # [::-1],
    )  # |11...1> -> |target_string>
    + Qunmask("previous")
)


def add_motifs(motifs_list):
    motif = Qmotifs()
    for m in motifs_list:
        motif += m
    return motif


U_oracle_naked_mutliT = lambda target_string: (
    U_oracle_naked(target_string)
    if isinstance(target_string, str)
    else add_motifs([U_oracle_naked(t) for t in target_string])
)

U_oracle = lambda target_string: (
    Qinit(2 * len(target_string)) + U_oracle_naked(target_string)
)


###

# %%
# # View the circuit of the oracle
# import matplotlib.pyplot as plt

# from hierarqcal import (
#     Qinit,
#     plot_circuit,
# )

# target = "011"
# base_layout = lambda n=1: ["az", "z"] * n
# bit_layout = (
#     lambda n=1: [f"{b}{i}" for i in range(n - 1, -1, -1) for b in base_layout()] + []
# )

# circ = Qinit(bit_layout(len(target))) + U_oracle_naked(target) + U_d
# # circ = Qinit(bit_layout(len(target))) + U_oracle_naked_mutliT([target]) + U_d

# fig, ax = plot_circuit(
#     circ,
#     plot_width=40,
# )
# plt.show()
# print()

# #%%
# from train_sets import get_train_set_grover
# from qalgosynth.problemsetup import ProblemSetupBase

# results = []
# for nq in range(2,6):
#     print(f"nq: {nq}")
#     params = {
#         "penalty_fn": None,
#         "reward_fn": lambda mean_fidelity: mean_fidelity,
#         "train_set": get_train_set_grover(nq, enforce_ancilla=True),
#         }

#     p = ProblemSetupBase(**params)
#     genotypes = []
#     motif = Qmotifs() + (U_oracle_naked,)  + U_d
#     for _ in range(10):
#         out = p.evaluate_genotype(motif)
#         motif += (U_oracle_naked,) + U_d
#         genotypes.append(out)
#         print(out.mean_fidelity)

#     results.append(genotypes)

# print()

####

# from train_sets import get_train_set_grover_v2
# from qalgosynth.problemsetup import ProblemSetupBase

# results = []
# for nq in range(1, 4):
#     print(f"nq: {nq}")
#     params = {
#         "length_penalty_fn": None,
#         "oracle_penalty_fn": None,  # lambda x: 0.0001 * x,
#         "reward_fn": lambda mean_fidelity, percentage_correct: mean_fidelity,
#         "train_set": get_train_set_grover_v2(nq, enforce_ancilla=True),
#     }

#     p = ProblemSetupBase(**params)
#     genotypes = []
#     motif = Qmotifs() + (U_oracle_naked_mutliT,) + U_d
#     for _ in range(5):
#         out = p.evaluate_genotype(motif)
#         motif += (U_oracle_naked_mutliT,) + U_d
#         genotypes.append(out)
#         print(out.mean_fidelity)

#     # results.append(genotypes)

# print()

# %%
