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
Z = get_unitary_from_penny_gate("Z")
# Cp = get_unitary_from_penny_gate("CPhase", [-np.pi / 2], "CP-", 2)
# Cpd = get_unitary_from_penny_gate("CPhase", [np.pi / 2], "CP+", 2)

# Cnot = get_unitary_from_penny_gate("CNOT")
# Cz = get_unitary_from_penny_gate("CZ")

# U_toffoli = get_unitary_from_penny_gate("Toffoli")

multiCX = get_unitary_from_penny_gate("MultiControlledX", None, "MCX", -1)

multiCZ = (
    Qpivot(mapping=H, global_pattern="*1")
    + Qpivot(mapping=multiCX, global_pattern="1*", merge_within="!")
    + Qpivot(mapping=H, global_pattern="*1")
)


U_d = (
    Qcycle(mapping=H)
    + Qcycle(mapping=X)
    + Qpivot(mapping=H, global_pattern="*1")
    + Qpivot(mapping=multiCX, global_pattern="1*", merge_within="!")
    + Qpivot(mapping=H, global_pattern="*1")
    + Qcycle(mapping=X)
    + Qcycle(mapping=H)
)

U_oracle_naked = lambda target_string: (
    Qpivot(
        mapping=X,
        global_pattern="".join(
            ["0" if x == "1" else "1" for x in target_string]
        ),  # [::-1],
    )  # |target_string> -> |11...1>
    + multiCZ  # |11...1> -> -|11...1>
    + Qpivot(
        mapping=X,
        global_pattern="".join(
            ["0" if x == "1" else "1" for x in target_string]
        ),  # [::-1],
    )  # |11...1> -> |target_string>
)


U_oracle = lambda target_string: (
    Qinit(len(target_string)) + U_oracle_naked(target_string)
)

# %%
# # View the circuit of the oracle
# import matplotlib.pyplot as plt

# from hierarqcal import (
#     Qinit,
#     plot_circuit,
# )

# target = "011"
# base_layout = lambda n=1: ["z"] * n

# circ = Qinit(len(target)) +  U_oracle_naked(target) + U_d

# # Identity = get_unitary_from_penny_gate("Identity", name="O", arity=1)
# # circ = Qinit(len(target)) + Qcycle(mapping=Identity)+ U_d

# # circ = Qinit(bit_layout(len(target))) + U_oracle_naked_mutliT([target]) + U_d

# fig, ax = plot_circuit(
#     circ,
#     plot_width=20,
# )
# # plt.show()
# # print()

# #%%
# from qalgosynth.train_sets import get_train_set_grover
# from qalgosynth.problemsetup import ProblemSetupBase

# results = []
# for nq in range(2,6):
#     print(f"nq: {nq}")
#     params = {
#         "penalty_fn": None,
#         "reward_fn": lambda mean_fidelity: mean_fidelity,
#         "train_set": get_train_set_grover(nq),
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

# # %%

# %%
