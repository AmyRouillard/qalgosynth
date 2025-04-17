import numpy as np
import itertools


def get_train_set_grover_ancillas(nq, enforce_ancilla=False):

    ket = [np.array([1, 0]), np.array([0, 1])]
    state_in = [
        (ket[0] + ket[1]) / np.sqrt(2) if elem % 2 == 1 else ket[0]
        for elem in range(nq * 2)
    ]
    state_vector_in = np.kron(state_in[0], state_in[1])
    for i in range(2, nq * 2):
        state_vector_in = np.kron(state_vector_in, state_in[i])

    states_out = []
    targets = [np.binary_repr(i, width=nq) for i in range(2**nq)]
    for t in targets:
        tmp = []
        if enforce_ancilla:
            for tt in [targets[0]]:  # ancillas must end in initial states
                tt_and_t = "".join([k + kk for k, kk in zip(tt, t)])
                tmp.append([ket[int(k)] for k in tt_and_t])
        else:
            raise NotImplementedError
            # the current output_state_vectors are assumed to be a single state
            for tt in targets:  # trace out ancillas
                tt_and_t = "".join([k + kk for k, kk in zip(tt, t)])
                tmp.append([ket[int(k)] for k in tt_and_t])

        states_out.append(tmp)

    state_vectors_out = []
    for ss in states_out:
        vv = []
        for s in ss:
            v = np.kron(s[0], s[1])
            for i in range(2, nq * 2):
                v = np.kron(v, s[i])
            vv.append(v)
        state_vectors_out.append(vv)

    train_set = {}
    for i, t in enumerate(targets):
        train_set[i] = {
            "input": state_vector_in,
            "output": state_vectors_out[i][0],
            "kwargs": {"target_string": t},
        }

    return train_set


def get_train_set_grover(
    nq,
):

    ket = [np.array([1, 0]), np.array([0, 1])]
    state_in = [(ket[0] + ket[1]) / np.sqrt(2) for _ in range(nq)]
    state_vector_in = np.kron(state_in[0], state_in[1])
    for i in range(2, nq):
        state_vector_in = np.kron(state_vector_in, state_in[i])

    states_out = []
    targets = [np.binary_repr(i, width=nq) for i in range(2**nq)]
    for t in targets:
        tmp = [ket[int(k)] for k in t]
        states_out.append([tmp])

    state_vectors_out = []
    for ss in states_out:
        vv = []
        for s in ss:
            v = np.kron(s[0], s[1])
            for i in range(2, nq):
                v = np.kron(v, s[i])
            vv.append(v)
        state_vectors_out.append(vv)

    train_set = {}
    for i, t in enumerate(targets):
        train_set[i] = {
            "input": state_vector_in,
            "output": state_vectors_out[i][0],
            "kwargs": {"target_string": t},
        }

    return train_set


# test = get_train_set_grover(2)
# print(test)

def get_train_set_grover_v2(nq, enforce_ancilla=False):

    ket = [np.array([1, 0]), np.array([0, 1])]
    state_in = [
        (ket[0] + ket[1]) / np.sqrt(2) if elem % 2 == 1 else ket[0]
        for elem in range(nq * 2)
    ]
    state_vector_in = np.kron(state_in[0], state_in[1])
    for i in range(2, nq * 2):
        state_vector_in = np.kron(state_vector_in, state_in[i])

    # single target
    states_out = []
    targets = [np.binary_repr(i, width=nq) for i in range(2**nq)]
    for t in targets:
        tmp = []
        if enforce_ancilla:
            for tt in [targets[0]]:  # ancillas must end in initial states
                tt_and_t = "".join([k + kk for k, kk in zip(tt, t)])
                tmp.append([ket[int(k)] for k in tt_and_t])
        else:
            for tt in targets:  # trace out ancillas
                tt_and_t = "".join([k + kk for k, kk in zip(tt, t)])
                tmp.append([ket[int(k)] for k in tt_and_t])

        states_out.append(tmp)

    state_vectors_out = []
    for ss in states_out:
        vv = []
        for s in ss:
            v = np.kron(s[0], s[1])
            for i in range(2, nq * 2):
                v = np.kron(v, s[i])
            vv.append(v)
        state_vectors_out.append(vv)

    train_set = {}
    # for i, t in enumerate(targets):
    #     train_set[i] = {
    #         "input": state_vector_in,
    #         "output": state_vectors_out[i][0],
    #         "kwargs": {"target_string": t},
    #     }
    for n in range(1, 2**nq + 1):
        for combination in itertools.combinations(range(len(targets)), n):
            targets_combination = [targets[i] for i in combination]
            state_vectors_combination = [state_vectors_out[i][0] for i in combination]
            output_vector = state_vectors_combination[0]
            for i in range(1, len(state_vectors_combination)):
                output_vector += state_vectors_combination[i]
            output_vector = output_vector / np.linalg.norm(output_vector)

            train_set[len(train_set)] = {
                "input": state_vector_in,
                "output": output_vector,
                "kwargs": {"target_string": targets_combination},
            }

    return train_set


# test = get_train_set_grover_v2(2)
# print(test)


def get_train_set_qft(nq):

    ket = [np.array([1, 0]), np.array([0, 1])]

    targets = [np.binary_repr(i, width=nq)[::-1] for i in range(2**nq)]

    states_in = []
    for target in targets:
        tmp = []
        for bit_string in target:
            tmp.append([ket[int(b)] for b in bit_string])
        states_in.append(tmp)

    state_vectors_in = []
    for state in states_in:
        state_vector_in = np.kron(state[0], state[1])
        for i in range(2, len(state)):
            state_vector_in = np.kron(state_vector_in, state[i])
        state_vectors_in.append(state_vector_in)

    state_vectors_out = []
    for target in targets:
        y_k = []
        for k in range(2**nq):
            y_k.append(np.exp(2j * np.pi * k * int(target, 2) / 2**nq) / np.sqrt(2**nq))
        tmp = y_k[0] * state_vectors_in[0]
        for i in range(1, len(y_k)):
            tmp += y_k[i] * state_vectors_in[i]
        state_vectors_out.append(tmp)

    train_set = {}
    for i in range(2**nq):
        train_set[i] = {
            "input": state_vectors_in[i][0],
            "output": state_vectors_out[i][0],
        }

    superposition = state_vectors_in[0].copy()
    for i in range(1, len(state_vectors_in)):
        superposition += state_vectors_in[i].copy()
    superposition = superposition / np.linalg.norm(superposition)

    superposition_out = state_vectors_out[0].copy()
    for i in range(1, len(state_vectors_out)):
        superposition_out += state_vectors_out[i].copy()
    superposition_out = superposition_out / np.linalg.norm(superposition_out)

    train_set[len(train_set)] = {
        "input": superposition[0],
        "output": superposition_out[0],
    }

    return train_set


# test = get_train_set_qft(4)
# print(test)


def get_train_set_3qubitgate(gate_name):

    gate_types = ["x", "toffoli", "or", "implies", "iff", "nxandy"]

    ket = [np.array([1, 0]), np.array([0, 1])]
    nq = 1
    targets = [np.binary_repr(i, width=nq) for i in range(2**nq)]

    outputs = []
    if gate_name == "x":
        # identity: x
        for x, y in itertools.product(targets, repeat=2):
            outputs.append(x)
    elif gate_name == "toffoli":
        # toffoli: x*y
        for x, y in itertools.product(targets, repeat=2):
            outputs.append(np.binary_repr((int(x, 2) * int(y, 2)) % 2, width=nq))
    elif gate_name == "or":
        # or: int(x or y)
        for x, y in itertools.product(targets, repeat=2):
            x = int(x)
            y = int(y)
            outputs.append(str(int((x or y))))
    elif gate_name == "implies":
        # implies: int((x and y) or (1 - x))
        for x, y in itertools.product(targets, repeat=2):
            x = int(x)
            y = int(y)
            outputs.append(str((x and y) or (1 - x)))
    elif gate_name == "iff":
        # iff: int((x and y) or (1 - x)) * int((x and y) or (1 - y))
        for x, y in itertools.product(targets, repeat=2):
            x = int(x)
            y = int(y)
            outputs.append(str(((x and y) or (1 - x)) * ((x and y) or (1 - y))))
    elif gate_name == "nxandy":
        # nxandy: int((1-x) and y)
        for x, y in itertools.product(targets, repeat=2):
            x = int(x)
            y = int(y)
            outputs.append(str((1 - x) and y))
    else:
        raise ValueError(f"gate_name must be one of {gate_types}")

    # print(outputs)

    states_in = []
    states_out = []
    for i, target in enumerate(itertools.product(targets, repeat=2)):
        bitstring_in = "".join(target) + "0"
        bitstring_out = "".join(target) + outputs[i]
        tmp_in = []
        tmp_out = []
        for bit in bitstring_in:
            tmp_in.append([ket[int(bit)]])
        for bit in bitstring_out:
            tmp_out.append([ket[int(bit)]])
        states_in.append(tmp_in)
        states_out.append(tmp_out)

    state_vectors_in = []
    for state in states_in:
        state_vector_in = np.kron(state[0], state[1])
        for i in range(2, len(state)):
            state_vector_in = np.kron(state_vector_in, state[i])
        state_vectors_in.append(state_vector_in)

    state_vectors_out = []
    for state in states_out:
        state_vector_out = np.kron(state[0], state[1])
        for i in range(2, len(state)):
            state_vector_out = np.kron(state_vector_out, state[i])
        state_vectors_out.append(state_vector_out)

    train_set = {}
    for i in range(len(state_vectors_out)):
        train_set[i] = {
            "input": state_vectors_in[i][0],
            "output": state_vectors_out[i][0],
        }

    superposition = state_vectors_in[0].copy()
    for i in range(1, len(state_vectors_in)):
        superposition += state_vectors_in[i].copy()
    superposition = superposition / np.linalg.norm(superposition)

    superposition_out = state_vectors_out[0].copy()
    for i in range(1, len(state_vectors_out)):
        superposition_out += state_vectors_out[i].copy()
    superposition_out = superposition_out / np.linalg.norm(superposition_out)

    train_set[len(train_set)] = {
        "input": superposition[0],
        "output": superposition_out[0],
    }

    return train_set


# test = get_train_set_3qubitgate("x")
# print(test)
# test = get_train_set_3qubitgate("toffoli")
# print(test)
# test = get_train_set_3qubitgate("or")
# print(test)
# test = get_train_set_3qubitgate("implies")
# print(test)
# test = get_train_set_3qubitgate("iff")
# print(test)
# test = get_train_set_3qubitgate("nxandy")
# print(test)


def get_train_set_adder(nq):

    # adder |0*(nq+1)>|x,0>|y,0> -> |0*(nq+1)>|x0>|x+y>

    ket = [np.array([1, 0]), np.array([0, 1])]
    targets = [np.binary_repr(i, width=nq) for i in range(2**nq)]

    inputs = []
    outputs = []
    for x, y in itertools.product(targets, repeat=2):
        x = np.binary_repr(int(x, 2), width=nq)
        y = np.binary_repr(int(y, 2), width=nq)
        inputs.append(["0" * nq + "0", x[::-1] + "0", y[::-1] + "0"])
        outputs.append(
            [
                "0" * nq + "0",
                x[::-1] + "0",
                np.binary_repr((int(x, 2) + int(y, 2)), width=nq + 1)[::-1],
            ]
        )  # enforce ancilla to be 0 in output

    states_in = []
    states_out = []
    input_strings = []
    output_strings = []
    for input, output in zip(inputs, outputs):
        a, x, y = input
        bitstring_in = "".join(["".join([i, j, k]) for i, j, k in zip(a, x, y)])
        input_strings.append(bitstring_in)

        a, x, y = output
        bitstring_out = "".join(["".join([i, j, k]) for i, j, k in zip(a, x, y)])
        output_strings.append(bitstring_out)

        tmp_in = []
        tmp_out = []
        for bit in bitstring_in:
            tmp_in.append([ket[int(bit)]])
        for bit in bitstring_out:
            tmp_out.append([ket[int(bit)]])
        states_in.append(tmp_in)
        states_out.append(tmp_out)

    state_vectors_in = []
    for state in states_in:
        state_vector_in = np.kron(state[0], state[1])
        for i in range(2, len(state)):
            state_vector_in = np.kron(state_vector_in, state[i])
        state_vectors_in.append(state_vector_in)

    state_vectors_out = []
    for state in states_out:
        state_vector_out = np.kron(state[0], state[1])
        for i in range(2, len(state)):
            state_vector_out = np.kron(state_vector_out, state[i])
        state_vectors_out.append(state_vector_out)

    train_set = {}
    for i in range(len(state_vectors_out)):
        train_set[i] = {
            "input": state_vectors_in[i][0],
            "output": state_vectors_out[i][0],
            "input_string": input_strings[i],
            "output_string": output_strings[i],
        }

    # superposition = state_vectors_in[0].copy()
    # for i in range(1, len(state_vectors_in)):
    #     superposition += state_vectors_in[i].copy()
    # superposition = superposition / np.linalg.norm(superposition)

    # superposition_out = state_vectors_out[0].copy()
    # for i in range(1, len(state_vectors_out)):
    #     superposition_out += state_vectors_out[i].copy()
    # superposition_out = superposition_out / np.linalg.norm(superposition_out)

    # train_set[len(train_set)] = {
    #     "input": superposition[0],
    #     "output": superposition_out[0],
    # }

    return train_set


# test = get_train_set_adder(2)
# print(test)


def get_train_set_deutsch_jozsa(nq):

    ket = [np.array([1, 0]), np.array([0, 1])]
    state_in = [ket[0] for _ in range(nq)]

    state_vector_in = np.kron(state_in[0], state_in[1])
    for i in range(2, nq):
        state_vector_in = np.kron(state_vector_in, state_in[i])

    # train_set = {}
    # for _ in range(n_balanced):
    #     target_idx = np.random.choice(
    #         range(2 ** (nq - 1)), 2 ** (nq - 2), replace=False
    #     )
    #     for t in ["balanced", "constant"]:  #
    #         train_set[len(train_set)] = {
    #             "input": state_vector_in,
    #             "output": state_vector_in,
    #             "kwargs": {
    #                 "nq": nq - 1,
    #                 "ftype": t,
    #                 "targets_idx": target_idx,
    #             },
    #             "expected_fidelity": 1.0 if t == "constant" else 0.0,
    #         }

    # train_set = {}
    # for start in range(2 ** (nq - 1)):
    # target_idx = [i % (2 ** (nq - 1)) for i in range(start, start + 2 ** (nq - 2))]

    train_set = {}
    for target_idx in itertools.combinations(range(2 ** (nq - 1)), 2 ** (nq - 2)):
        for t in ["balanced", "constant"]:  #
            train_set[len(train_set)] = {
                "input": state_vector_in,
                "output": state_vector_in,
                "kwargs": {
                    "nq": nq - 1,
                    "ftype": t,
                    "targets_idx": target_idx,
                },
                "expected_fidelity": 1.0 if t == "constant" else 0.0,
            }

    # train_set = {}
    # for target_idx in itertools.combinations(range(2 ** (nq - 1)), 2 ** (nq - 2)):
    #     train_set[len(train_set)] = {
    #         "input": state_vector_in,
    #         "output": state_vector_in,
    #         "kwargs": {
    #             "nq": nq - 1,
    #             "ftype": "balanced",
    #             "targets_idx": target_idx,
    #         },
    #         "expected_fidelity": 0.0,
    #     }

    # train_set[len(train_set)] = {
    #     "input": state_vector_in,
    #     "output": state_vector_in,
    #     "kwargs": {
    #         "nq": nq - 1,
    #         "ftype": "constant",
    #         "targets_idx": target_idx,
    #     },
    #     "expected_fidelity": 1.0,
    # }

    return train_set


# tmp = get_train_set_deutsch_jozsa(4)
# print()
