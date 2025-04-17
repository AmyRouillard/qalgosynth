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
)

from qalgosynth.utils import get_unitary_from_penny_gate

X = get_unitary_from_penny_gate("X")
MultiCX = get_unitary_from_penny_gate("MultiControlledX")
Identity = get_unitary_from_penny_gate("Identity")


def U(nq, ftype, targets_idx=None):
    if ftype == "balanced":
        # binary strings for number up to 2**(nq-1)
        targets = [f"{i:0{nq}b}" for i in range(2 ** (nq))]
        # randomly choose 2**(nq-1) targets
        # targets = np.random.choice(targets, 2 ** (nq - 1), replace=False)
        if targets_idx is None:
            targets = targets[: 2 ** (nq - 1)]
        else:
            # select the targets with the given indices
            targets = [targets[i] for i in targets_idx]

        motif = Qmotifs()
        for t in targets:
            t = t + "0"
            Edges = [i for i in range(nq + 1)]
            motif += (
                Qpivot(mapping=X, global_pattern=t)
                + Qmotif(mapping=MultiCX, E=[tuple(Edges)])
                + Qpivot(mapping=X, global_pattern=t)
            )
        # return motif
        return Qmotifs() + Qmotif(
            mapping=Qinit(nq + 1) + motif,
            E=[tuple(Edges)],
        )
    elif ftype == "constant":
        return Qmotifs() + Qpivot(mapping=Identity, global_pattern="1*")
    else:
        raise ValueError(f"Unknown type: {ftype}")


DJ_oracle_naked = lambda nq, ftype, targets_idx: U(nq, ftype, targets_idx)


###

# H = get_unitary_from_penny_gate("Hadamard")

# DJ_motif = (
#         Qpivot(mapping=X, global_pattern="*1")
#         + Qcycle(mapping=H)
#         + (DJ_oracle_naked,)
#         + Qcycle(mapping=H)
#         + Qpivot(mapping=X, global_pattern="*1")
#     )

# # %%

# # View the circuit of the oracle
# import matplotlib.pyplot as plt

# from hierarqcal import (
#     Qinit,
#     plot_circuit,
# )

# nq = 4
# ftype = "balanced"
# # ftype = "constant"
# circ = Qinit(nq) + DJ_oracle_naked(nq - 1, ftype, None)
# fig, ax = plot_circuit(
#     circ,
#     plot_width=40,
# )
# # save the figure
# # plt.savefig(f"test/deutsch_jozsa_oracle_{ftype}.png")
# plt.show()

# # %%

# from qalgosynth.train_sets import get_train_set_deutsch_jozsa
# from qalgosynth.problemsetup import ProblemSetupBase

# # %%
# get_train_set_deutsch_jozsa(4)
# # %%
# results = []
# for nq in range(2, 5):
#     print(f"nq: {nq}")
#     params = {
#         "penalty_fn": None,
#         "reward_fn": lambda mean_fidelity: mean_fidelity,
#         "train_set": get_train_set_deutsch_jozsa(nq),
#     }

#     p = ProblemSetupBase(**params)

#     results.append(p.evaluate_genotype(DJ_motif, repetitions=1))


#     circ = Qinit(nq)
#     for m in DJ_motif:
#         if isinstance(m, type(lambda x: x)):
#             circ += m(nq - 1, "constant", None)
#         else:
#             circ += m

#     fig, ax = plot_circuit(
#         circ,
#         plot_width=20,
#     )


# # print()

# # %%

# results

# %%
