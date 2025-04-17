import pennylane as qml
import inspect
from itertools import product

from hierarqcal import (
    Qunitary,
    Qcycle,
    Qmask,
    Qpivot,
    Qunmask,
)


def penny_gate_to_function(gate):
    return lambda bits, symbols: gate(*symbols, wires=[*bits])


def get_unitary_from_penny_gate(gate_name, symbols=None, name=None, arity=None):
    penny_gate = getattr(qml, gate_name)
    return Qunitary(
        penny_gate_to_function(penny_gate),  #
        n_symbols=penny_gate.num_params,
        arity=(
            penny_gate.num_wires if arity is None else arity
        ),  # TODO: for the Multi-controlled gates can we make the default arity the number of available qubits?
        symbols=symbols,
        name=gate_name if name is None else name,
    )


def __add_nodes_edges__(genotype, history, all_parents, population_dict):
    parent_ids = genotype.parent_ids.split("--")
    parents = [population_dict[x] for x in parent_ids if x in population_dict.keys()]
    history += [
        (
            genotype.id,
            genotype.parent_ids,
        )
    ]
    all_parents += parents
    if len(parents) == 0:
        pass
        # print('len = 0:',[p.parent_ids for p in parents])
    else:
        # print('len > 0:',[p.parent_ids for p in parents])
        for p in parents:
            __add_nodes_edges__(p, history, all_parents, population_dict)

    return history, all_parents


def describe_motif(motif):
    Qcp = [
        x
        for x in inspect.signature(Qcycle.__init__).parameters
        if x not in ["self", "kwargs"]
    ] + ["mapping", "share_weights", "edge_order"]
    Qpp = [
        x
        for x in inspect.signature(Qpivot.__init__).parameters
        if x not in ["self", "kwargs"]
    ] + ["mapping", "share_weights", "edge_order"]
    Qmp = [
        x
        for x in inspect.signature(Qmask.__init__).parameters
        if x not in ["self", "kwargs"]
    ] + ["mapping", "share_weights", "edge_order"]

    text = ""
    for m in motif:
        if isinstance(m, Qcycle):
            text += "Qcycle: "
            att = {k: getattr(m, k) for k in Qcp}
            att["mapping"] = getattr(m, "mapping").name
            text += ", ".join(
                [f"{k}:{str(att[k]).replace(' ','')}" for k in att.keys()]
            )  # f"{att}<br/> "
            text += "<br/> "
        elif isinstance(m, Qmask):
            text += "Qmask: "
            att = {k: getattr(m, k) for k in Qmp}
            try:
                att["mapping"] = getattr(m, "mapping").name
            except:
                att["mapping"] = None
            text += ", ".join(
                [f"{k}:{str(att[k]).replace(' ','')}" for k in att.keys()]
            )
            text += "<br/> "
        elif isinstance(m, Qpivot):
            text += "Qpivot: "
            att = {k: getattr(m, k) for k in Qpp}
            att["mapping"] = getattr(m, "mapping").name
            text += ", ".join(
                [f"{k}:{str(att[k]).replace(' ','')}" for k in att.keys()]
            )
            text += "<br/> "
        elif isinstance(m, type(lambda x: x)):
            text += "Oracle<br/> "
        elif isinstance(m, Qunmask):
            text += "Qunmask<br/> "
        else:
            text += "??<br/> "

    if "<hierarqcal.core.Qunitaryobjectat" in text:
        raise ValueError("Text error, Qunitary object in genotype")

    return text


import numpy as np
import torch


def JSD_uniform_np(p):
    """
    Jensen-Shannon divergence from uniform distribution.
    return between 0 and 1, 0 best 1 worst.
    """

    # add a very small number to avoid log(0)
    p = np.array(p) + 1e-10
    p = p / np.sum(p)
    y = 0.5 * (
        2
        - np.log2(len(p))
        + np.sum(p * np.log2(p) - (p + 1 / len(p)) * np.log2(p + 1 / len(p)))
    )
    if np.isnan(y):
        print("JSD_uniform_np: ", p)
    return y


def JSD_uniform_t(p):
    """
    Jensen-Shannon divergence from uniform distribution.
    return between 0 and 1, 0 best 1 worst.
    """
    # add a very small number to avoid log(0)
    p = torch.add(p, 1e-10)
    p = torch.div(p, torch.sum(p))
    return 0.5 * (
        2
        - np.log2(len(p))
        + torch.sum(p * torch.log2(p) - (p + 1 / len(p)) * torch.log2(p + 1 / len(p)))
    )


def get_search_space_size(discrete_variables, mask=True, oracle=True):

    PCycle = [
        "mapping_names",
        "stride",
        "step",
        "offset",
        "boundary",
        "share_weights",
        "edge_order",
    ]
    PPivot = [
        "mapping_names",
        "strides",
        "steps",
        "offsets",
        "boundaries",
        "global_pattern",
        "merge_within",
        "share_weights",
        "edge_order",
    ]

    PMask = [
        "global_pattern",
    ]

    if not mask:
        Plist = [PCycle, PPivot]
    else:
        Plist = [PMask, PCycle, PPivot]

    N = 0
    for properties in Plist:
        n = 1
        for k, v in discrete_variables.items():
            if k in properties:
                n *= len([x for x in product(*v)])
        N += n
    if oracle:
        N += 1
    if mask:
        N += 1

    return N
