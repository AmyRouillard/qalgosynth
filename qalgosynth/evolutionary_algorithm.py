import numpy as np
from hierarqcal import Qmotifs
from qalgosynth.motif_population import MotifPopulation
from itertools import product


class EvolutionaryCircuitOptimizer(MotifPopulation):
    """
    Class to perform evolutionary optimization of quantum circuits.

    Parameters
    ----------
    population : list of numedtuple
        Initial population of namedtuples with fields 'motif','id','fitness'...
    pressure : float
        Selection pressure.
    base_layout : function
        Function to generate base layout.
    additional_ancilla : list
        Additional ancilla qubits.
    motif_probabilities : list
        List of motif probabilities.
    oracle : function
        Oracle function.
    nq : int
        Number of computational qubits, excluding ancillas. Lets the program assume a minimum size of the circuit.

    Public methods
    -------
    tournament_selection(memory_table, pressure, max_pool)
        Using tournament selection to select two genotypes from the memory table.
    get_genotypes_tournament(max_pool, **kwargs)
        Get a list of genotypes by performing tournament selection and applying crossovers and mutations.
    get_genotypes_random(lengths, **kwargs)
        Get a list of random genotypes of given lengths.

    Attributes
    ----------
    population : list
        List of Qmotifs.
    pressure : float
        Selection pressure.
    crossovers : dict
        Dictionary of crossover functions.
    mutations : dict
        Dictionary of mutation functions.

    """

    def __init__(
        self,
        population=[],
        pressure=0.05,
        exp_exp_tradeoff=0.1,
        base_layout=lambda n=1: ["az", "z"] * n,
        additional_ancilla=[],
        motif_probabilities=None,
        oracle=None,
        discrete_variables=None,
        motif_size_limit=None,
        random_number_generator=None,
        nq=None,
    ):
        super().__init__(
            base_layout=base_layout,
            additional_ancilla=additional_ancilla,
            motif_probabilities=motif_probabilities,
            oracle=oracle,
            discrete_variables=discrete_variables,
            random_number_generator=random_number_generator,
            nq=nq,
        )

        self.population = population
        self.population_dict = self.__get_population_dict__()

        self.pressure = pressure
        self.exp_exp_tradeoff = exp_exp_tradeoff
        self.motif_size_limit = motif_size_limit

        self.crossovers = {
            # "append": self.__crossover_join__,
            # "symmetrize": self.__crossover_symmetrize__,
            # "interleave": self.__crossover_interleave__,
            "chop_and_join": self.__crossover_chop_and_join__,
        }

        self.mutations = {
            "motif": self.__mutate__motif__,
            # "replace": self.__mutate__replace__,
            # "symmetrize": self.__mutate__symmetrize__,
            # "shuffle": self.__mutate__shuffle__,
            "dropout": self.__mutate__dropout__,
            # "chop": self.__mutate__chop__,
            # "append": self.__mutate__append__,
            "insert": self.__mutate__insert__,
            # "insertmid": self.__mutate__insert__mid__,
        }

    def __get_population_dict__(self):
        """
        Get the population as a dictionary.

        Returns
        -------
        dict
            Dictionary of the population.
        """
        population_dict = {}
        for i, genotype in enumerate(self.population):
            if "".join(genotype.genotype) not in population_dict:
                population_dict["".join(genotype.genotype)] = i

        return population_dict

    def update_population(self, new_population):
        """
        Update the population dictionary.

        Parameters
        ----------
        new_population : list
            List of Qmotifs.
        """

        for i, genotype in enumerate(new_population):
            self.population_dict["".join(genotype.genotype)] = i + len(self.population)

        self.population.extend(new_population)

    def tournament_selection(
        self, pressure, max_pool, exp_exp_tradeoff, max_selected=10
    ):
        """
        Using tournament selection to select two genotypes from the memory table.
        With 10% probability the pressure is ignored and two random genotypes are selected.

        Parameters
        ----------
        memory_table : list
            List of named tuples with field 'fitness'.
        pressure : float
            Selection pressure.
        max_pool : int
            Maximum pool size.

        Returns
        -------
        tuple
            Tuple of two selected genotypes.

        """

        if len(self.population) < 2:
            raise ValueError("Memory table must have at least two elements.")

        if max_pool is None:
            max_pool = len(self.population)

        if self.rng.random() < exp_exp_tradeoff:
            selected_genotypes_ind = self.rng.choice(
                len(self.population), 2, replace=False
            )
            selected_genotypes = (
                self.population[selected_genotypes_ind[0]],
                self.population[selected_genotypes_ind[1]],
            )
        else:

            num_elements = np.min([int(len(self.population) * pressure), max_pool])
            if num_elements < 2:
                num_elements = 2

            selected_genotypes_ind = self.rng.choice(
                len(self.population), num_elements, replace=False
            )
            # shuffle selected genotypes
            self.rng.shuffle(selected_genotypes_ind)

            selected_genotypes_ind = sorted(
                [ind for ind in selected_genotypes_ind],
                key=lambda x: self.population[x].fitness,
                reverse=True,
            )[:max_selected]

            # TODO: negative fitness values??!!
            fitness = [
                (
                    self.population[ind].fitness
                    if self.population[ind].fitness > 0
                    else 1e-8
                )
                for ind in selected_genotypes_ind
            ]
            fitness = fitness / np.sum(fitness)
            # print(selected_genotypes_ind)
            selected_genotypes_ind = self.rng.choice(
                selected_genotypes_ind, 2, replace=False, p=fitness
            )
            selected_genotypes = (
                self.population[selected_genotypes_ind[0]],
                self.population[selected_genotypes_ind[1]],
            )
            # print([round(x, 2) for x in fitness])
            # print(selected_genotypes_ind)

            # selected_genotypes = sorted(
            #     [self.population[ind] for ind in selected_genotypes_ind],
            #     key=lambda x: x.fitness,
            #     reverse=True,
            # )
        return selected_genotypes

    def __resize_motif__(self, motif):
        """
        If motif_size_limit is set then resize the given motif by taking a random slice.

        Parameters
        ----------
        motif : Qmotifs
            Motif to resize.
        """

        if self.motif_size_limit is None:
            return motif
        else:
            if len(motif) > self.motif_size_limit:
                # take a random start and end point
                start = self.rng.integers(0, len(motif))
                if 1 < len(motif) - start:
                    end = start + min(
                        self.rng.integers(1, len(motif) - start), self.motif_size_limit
                    )
                else:
                    end = start + 1
                return motif[start:end]
            else:
                return motif

    def __symmetrize_motif__(self, motif):
        property = "edge_order"
        symmetric_motif = Qmotifs()
        for motif_id in motif[::-1]:
            motif_name = self.available_motifs[int(motif_id[0])]
            motif_properties = motif_id[1:]

            if property in self.motif_params_dict[motif_name]:
                property_ind = self.motif_params_dict[motif_name].index(property)
                id_iterator = [range(n) for n in self.N_opts[motif_name]]
                offset = (
                    self.id_pad
                    - len(self.motif_params_dict[motif_name]) * self.param_pad
                )
                old_value = motif_properties[
                    property_ind + offset : property_ind + offset + self.param_pad
                ]
                new_value = self.rng.choice(
                    [x for x in id_iterator[property_ind] if x != old_value]
                )

                symmetric_motif += (
                    motif_id[0]
                    + motif_properties[: property_ind + offset]
                    + "0" * (self.param_pad - len(str(new_value)))
                    + str(new_value)
                    + motif_properties[property_ind + offset + self.param_pad :],
                )
            else:
                symmetric_motif += (motif_id,)

        return symmetric_motif

    def __crossover_join__(self, parent1, parent2):
        """
        Mutate motif by joining the motif head to tail.

        Parameters
        ----------
        parent1 : Qmotifs
            First parent.
        parent2 : Qmotifs
            Second parent.
        """

        child1 = Qmotifs() + parent1 + parent2
        child2 = Qmotifs() + parent2 + parent1

        # child1 = self.__resize_motif__(child1)
        # child2 = self.__resize_motif__(child2)

        return child1, child2

    def __crossover_symmetrize__(self, parent1, parent2):

        symmetric_motif1 = self.__symmetrize_motif__(parent1)
        symmetric_motif2 = self.__symmetrize_motif__(parent2)

        child1, child2 = parent1 + symmetric_motif2, parent2 + symmetric_motif1

        # child1 = self.__resize_motif__(child1)
        # child2 = self.__resize_motif__(child2)

        return (child1, child2)

    def __crossover_interleave__(self, parent1, parent2):
        """
        Mutate motif by interleaving the motifs.
        The second child is the reverse of the first child.

        Parameters
        ----------
        parent1 : Qmotifs
            First parent.
        parent2 : Qmotifs
            Second parent.

        """

        if len(parent1) > len(parent2):
            len_1 = len(parent2)
            len_2 = len(parent1)
            parent1, parent2 = parent2, parent1
        else:
            len_1 = len(parent1)
            len_2 = len(parent2)

        if len_1 == 1 and len_2 == 1:
            # child1, child2 = (
            #     Qmotifs() + parent1 + parent2,
            #     Qmotifs() + parent2 + parent1,
            # )

            # # child1 = self.__resize_motif__(child1)
            # # child2 = self.__resize_motif__(child2)
            # return child1, child2
            # cross over join already does this
            return None

        offset = self.rng.integers(0, len_2 - len_1) if len_2 > len_1 else 0

        child = Qmotifs() + parent1[:offset]
        for i in range(0, len_1):
            child += (
                parent2[i + offset],
                parent1[i],
            )
        child += parent2[offset + len_1 :]
        # child = self.__resize_motif__(child)

        child_reverse = Qmotifs() + child[::-1]

        return child, child_reverse

    def __crossover_chop_and_join__(self, parent1, parent2):
        """
        Mutate mofits by chopping and joining the motifs.

        Parameters
        ----------
        parent1 : Qmotifs
            First parent.
        parent2 : Qmotifs
            Second parent.
        """
        len_1 = len(parent1)
        len_2 = len(parent2)

        if (len_1 == 1 or len_2 == 1) or (len_1 < 6 and len_2 < 6):
            # cross over join already does this
            child1, child2 = (
                Qmotifs() + parent1 + parent2,
                Qmotifs() + parent2 + parent1,
            )
        else:
            i = self.rng.integers(1, len_1)
            j = self.rng.integers(1, len_2)
            child1 = parent1[0:i] + parent2[j:]
            child2 = parent2[0:j] + parent1[i:]

        # child1 = self.__resize_motif__(child1)
        # child2 = self.__resize_motif__(child2)

        return child1, child2

    def __mutate__motif__(self, parent):
        """
        Choose a random motif from the parent and mutate it.
        Qunmask and Oracles are replaced by random motifs.

        Parameters
        ----------
        parent : Qmotifs
            Parent to mutate.
        """

        if len(parent) > 1:
            tmp = self.rng.choice(range(len(parent)))
        else:
            tmp = 0
            # return (Qmotifs() + self.get_random_motif(),)

        new_genotype = Qmotifs()
        for i, motif in enumerate(parent):
            if i == tmp:
                new_genotype += self.mutate_motif(motif)
            else:
                new_genotype += (motif,)

        return (new_genotype,)

    def __mutate__shuffle__(self, parent):
        """
        Shuffle two randomly selected motifs from parent.

        Parameters
        ----------
        parent : Qmotifs
            Parent to shuffle.
        """

        if len(parent) < 2:
            # return (Qmotifs() + self.get_random_motif(),)
            # mutate_motif already does this
            return None
        else:
            # select two motifs to swap, not the same
            tmp1, tmp2 = self.rng.choice(range(len(parent)), 2, replace=False)

            new_genotype = Qmotifs()
            for i, motif in enumerate(parent):
                if i == tmp1:
                    new_genotype += (parent[tmp2],)
                elif i == tmp2:
                    new_genotype += (parent[tmp1],)
                else:
                    new_genotype += (motif,)

            return (new_genotype,)

    def __mutate__replace__(self, parent):
        """ """

        if len(parent) > 1:
            new_genotype = Qmotifs()
            tmp = self.rng.choice(range(len(parent)))
            # add a new random motif to the genotype
            for i, motif in enumerate(parent):
                if i == tmp:
                    new_genotype += self.get_random_motif()
                else:
                    new_genotype += (motif,)

            # new_genotype = self.__resize_motif__(new_genotype)
            return (new_genotype,)
        else:
            return (Qmotifs() + self.get_random_motif(),)

    def __mutate__dropout__(self, parent):

        if len(parent) > 1:
            new_genotype = Qmotifs()
            tmp = self.rng.choice(range(len(parent)))
            # remove a random motif from the genotype
            for i, motif in enumerate(parent):
                if i == tmp:
                    pass
                else:
                    new_genotype += (motif,)

            # new_genotype = self.__resize_motif__(new_genotype)
            return (new_genotype,)
        else:
            return None

    def __mutate__append__(self, parent):
        """
        Append a random motif to the parent.

        Parameters
        ----------
        parent : Qmotifs
            Parent to append.

        """

        if self.rng.random() < 0.5:
            new_genotype = Qmotifs() + parent + self.get_random_motif()
        else:
            new_genotype = Qmotifs() + self.get_random_motif() + parent

        # new_genotype = self.__resize_motif__(new_genotype)
        return (new_genotype,)

    def __mutate__insert__(self, parent):
        """
        Insert a random motif to the parent.

        Parameters
        ----------
        parent : Qmotifs
            Parent to insert.

        Returns
        -------
        tuple
            Tuple of genotypes.
        """

        new_genotype = Qmotifs()
        tmp = self.rng.choice(range(len(parent) + 1))
        # add a new random motif to the genotype
        for i, motif in enumerate(parent):
            if i == tmp:
                new_genotype += self.get_random_motif()
                new_genotype += (motif,)
            else:
                new_genotype += (motif,)
        if tmp == len(parent):
            new_genotype += self.get_random_motif()

        # new_genotype = self.__resize_motif__(new_genotype)
        return (new_genotype,)

    def __mutate__insert__mid__(self, parent):
        """
        Insert a random motif to the parent.

        Parameters
        ----------
        parent : Qmotifs
            Parent to insert.

        Returns
        -------
        tuple
            Tuple of genotypes.
        """

        new_genotype = Qmotifs()
        tmp = len(parent) // 2
        # add a new random motif to the genotype
        for i, motif in enumerate(parent):
            if i == tmp:
                new_genotype += self.get_random_motif()
                new_genotype += (motif,)
            else:
                new_genotype += (motif,)
        if tmp == len(parent):
            new_genotype += self.get_random_motif()

        # new_genotype = self.__resize_motif__(new_genotype)
        return (new_genotype,)

    def __mutate__chop__(self, parent):
        """
        Chop the parent into two parts return the parts.

        Parameters
        ----------
        parent : Qmotifs
            Parent to chop.
        """

        if len(parent) > 1:
            tmp = self.rng.choice(range(1, len(parent)))
            return parent[:tmp], parent[tmp:]
        else:
            return None

    def __mutate__symmetrize__(self, parent):

        symmetric_motif = self.__symmetrize_motif__(parent)

        # symmetric_motif = self.__resize_motif__(symmetric_motif)

        return (parent + symmetric_motif,)

    def get_genotypes_tournament(
        self, generation, exp_exp_tradeoff=None, pressure=None, max_pool=None, **kwargs
    ):
        """
        Get a list og genotypes by performing tournament selection and applying crossovers and mutations.

        Parameters
        ----------
        max_pool : int
            Maximum pool size.
        kwargs : dict
            Additional keyword arguments for the evaluation function.
        """

        if exp_exp_tradeoff is None:
            exp_exp_tradeoff = self.exp_exp_tradeoff

        genotype1, genotype2 = self.tournament_selection(
            self.pressure if pressure is None else pressure,
            max_pool,
            exp_exp_tradeoff,
        )

        new_genotypes = []
        for k in self.crossovers.keys():
            children = self.crossovers[k](genotype1.genotype, genotype2.genotype)
            if children is not None:
                for child in children:
                    tmp = {
                        "motif_id": child,
                        "motif": self.get_motif_from_id(child),
                        "parent_ids": (
                            str(generation) + ":" + k,
                            genotype1.id,
                            genotype2.id,
                        ),
                    }

                    tmp.update(kwargs)

                    new_genotypes.append(tmp)

        for k in self.mutations.keys():
            for g in [genotype1, genotype2]:
                children = self.mutations[k](g.genotype)
                if children is not None:
                    for child in children:
                        tmp = {
                            "motif_id": child,
                            "motif": self.get_motif_from_id(child),
                            "parent_ids": (str(generation) + ":" + k, g.id),
                        }

                        tmp.update(kwargs)

                        new_genotypes.append(tmp)

        # exclude None values
        new_genotypes = [x for x in new_genotypes if x["motif"] is not None]

        return new_genotypes

    def get_genotypes_random(self, lengths, parent_id=("seed",), **kwargs):
        """
        Get a list of random genotypes of given lengths.

        Parameters
        ----------
        lengths : list
            List of lengths of the genotypes.
        kwargs : dict
            Additional keyword arguments for the evaluation function.
        """

        new_genotypes = []
        for length in lengths:
            motif = Qmotifs()
            for _ in range(length):
                motif += self.get_random_motif()
            tmp = {
                "motif_id": motif,
                "motif": self.get_motif_from_id(motif),
                "parent_ids": parent_id,
            }
            tmp.update(kwargs)

            new_genotypes.append(tmp)

        new_genotypes = [x for x in new_genotypes if x["motif"] is not None]

        return new_genotypes

    def get_genotypes_unit_length(self, n_motifs, parent_id=("seed",), **kwargs):
        """ """

        new_genotypes = []
        for motif in self.__get_base_motifs__(n_motifs):
            tmp = {
                "motif_id": motif,
                "motif": self.get_motif_from_id(motif),
                "parent_ids": parent_id,
            }
            tmp.update(kwargs)

            new_genotypes.append(tmp)

        new_genotypes = [x for x in new_genotypes if x["motif"] is not None]

        return new_genotypes

    def get_genotypes_length_n(self, n_motifs, n=2, parent_id=("seed",), **kwargs):
        """ """

        new_genotypes = []
        base_motifs = [m for m in self.id_dict.keys()]

        for m in product(base_motifs, repeat=n):
            motif = Qmotifs() + m
            tmp = {
                "motif_id": motif,
                "motif": self.get_motif_from_id(motif),
                "parent_ids": parent_id,
            }
            tmp.update(kwargs)

            new_genotypes.append(tmp)

        new_genotypes = [x for x in new_genotypes if x["motif"] is not None]

        # take a random sample of n_motifs
        new_genotypes = self.rng.choice(
            new_genotypes,
            n_motifs,
            replace=False if n_motifs < len(new_genotypes) else True,
        )

        return new_genotypes
