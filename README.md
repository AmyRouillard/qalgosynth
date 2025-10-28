# Automated Quantum Algorithm Synthesis

A repository for the research and implementation described in the paper **"Automated Quantum Algorithm Synthesis"**.

## Overview

This repository contains the code and tools used in our work on **Automated Quantum Algorithm Synthesis**. We introduce a computational method for automatically designing $n$-qubit realizations of quantum algorithms using an domain-specific language (DSL). This DSL abstracts quantum circuits into modular, scalable building blocks—moving beyond standard gate-sequence representations and enabling algorithm structures to be learned rather than specific unitary implementations. The DSL is implemented as a python package, [heirarqcal](https://github.com/matt-lourens/hierarqcal).

Our approach is validated by successfully rediscovering the Quantum Fourier Transform, Deutsch-Jozsa algorithm, and Grover’s search algorithm. Notably, general solutions for these algorithms were learned using training examples with at most 5 qubits. The code in this repository provides the code and information necessary to reproduce the results presented in the paper.


## Repository Structure
```
. ├── qalgosynth/ # Main source code 
  ├── examples/ # Example scripts
  └── README.md # This file
```

## Getting Started

### Requirements

- pennylane, torch, dill, pathos
- heirarqcal - please used this forked version installed locally: https://github.com/AmyRouillard/hierarqcal

### Installation

```bash
git clone https://github.com/AmyRouillard/qalgosynth.git
cd your-repo-name
pip install -r requirements.txt
```

Tested on Python 3.11.5.

### Running an Example

The file `run_search.py` contains an example of how to run the search algorithm and can be used to reproduce the results in the paper.

## Citation

If you use this work in your research, please cite:
```bibtex
@misc{rouillard2025automatedquantumalgorithmsynthesis,
      title={Automated Quantum Algorithm Synthesis}, 
      author={Amy Rouillard and Matt Lourens and Francesco Petruccione},
      year={2025},
      eprint={2503.08449},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2503.08449}, 
}
```

## License

Apache License. See LICENSE file for more information.