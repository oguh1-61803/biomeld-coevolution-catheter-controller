# All libraries necessary.
from neat import CompleteExtinctionException
from neat.reporting import ReporterSet
from neat.math_util import mean
import random
import neat


# This class controls the evolution of each population. It receives the configuration file of the population that is
# encoding.
class CoevolutionaryPopulation:

    def __init__(self, conf, function_bank):

        self.reporters = ReporterSet()
        reporter = neat.StdOutReporter(True)
        statistics = neat.StatisticsReporter()
        self.reporters.add(reporter)
        self.reporters.add(statistics)
        self.configuration = self.__generate_configuration(conf, function_bank)
        stagnation = self.configuration.stagnation_type(self.configuration.stagnation_config, self.reporters)
        self.reproduction = self.configuration.reproduction_type(self.configuration.reproduction_config, self.reporters,
                                                                 stagnation)

        self.fittest_individual_data = []
        self.population_average_fitness_data = []

        if self.configuration.fitness_criterion == 'max':

            self.fitness_criterion = max

        elif self.configuration.fitness_criterion == 'min':

            self.fitness_criterion = min

        elif self.configuration.fitness_criterion == 'mean':

            self.fitness_criterion = mean

        elif not self.configuration.no_fitness_termination:

            raise RuntimeError(

                "Unexpected fitness_criterion: {0!r}".format(conf.fitness_criterion))

        self.population = self.reproduction.create_new(self.configuration.genome_type, self.configuration.genome_config,
                                                       self.configuration.pop_size)
        self.species = self.configuration.species_set_type(self.configuration.species_set_config, self.reporters)
        self.generation = 0
        self.species.speciate(self.configuration, self.population, self.generation)
        self.best_genome = None
        self.reporters.start_generation(self.generation)

    # It returns the individuals.
    def get_population(self):

        return list(self.population.items())

    # It returns the configuration required by the NEAT library.
    def get_configuration(self):

        return self.configuration

    # It finds the best individual genome of the population.
    def set_best_genome(self):

        genome_fitness_list = []

        for genome_id, genome in self.population.items():
            data = (genome_id, genome.fitness)
            genome_fitness_list.append(data)

        genome_fitness_list.sort(key=lambda x: x[1], reverse=True)
        best_genome_data = genome_fitness_list[0]
        self.best_genome = self.population.get(best_genome_data[0])

    # It returns the best individual genome of the population.
    def get_best_genome(self):

        return self.best_genome

    # It returns the n fittest individuals of the population under a specific collaboration method.
    def get_top_n_genomes(self, n_individuals, collaboration_method):

        genome_aux_list = []
        n_genomes = []

        for genome_id, genome in self.population.items():

            data = (genome_id, genome.fitness)
            genome_aux_list.append(data)

        if collaboration_method == 'fittest_vs_all':

            genome_aux_list.sort(key=lambda x: x[1], reverse=True)

            for index in range(0, n_individuals):

                genome_data = genome_aux_list[index]
                genome = (genome_data[0], self.population.get(genome_data[0]))
                n_genomes.append(genome)

            return n_genomes

        elif collaboration_method == 'worst_vs_all':

            genome_aux_list.sort(key=lambda x: x[1], reverse=False)

            for index in range(0, n_individuals):

                genome_data = genome_aux_list[index]
                genome = (genome_data[0], self.population.get(genome_data[0]))
                n_genomes.append(genome)

            return n_genomes

        elif collaboration_method == 'best_and_worst_vs_all':

            genome_aux_list.sort(key=lambda x: x[1], reverse=True)
            fittest_index = 0
            worst_index = len(genome_aux_list) - 1

            for _ in range(0, n_individuals):

                fittest_genome_data = genome_aux_list[fittest_index]
                fittest_genome = (fittest_genome_data[0], self.population.get(fittest_genome_data[0]))
                n_genomes.append(fittest_genome)
                worst_genome_data = genome_aux_list[worst_index]
                worst_genome = (worst_genome_data[0], self.population.get(worst_genome_data[0]))
                n_genomes.append(worst_genome)
                fittest_index += 1
                worst_index -= 1

            return n_genomes

        elif collaboration_method == 'random_vs_all':

            random_genome_data_list = random.sample(genome_aux_list, n_individuals)

            for genome_data in random_genome_data_list:

                genome = (genome_data[0], self.population.get(genome_data[0]))
                n_genomes.append(genome)

            return n_genomes

        else:

            raise ValueError('Invalid approach: {0!r}'.format(collaboration_method))

    # It returns the data associated to the fittest individual and the population.
    def get_fitness_data(self):

        return self.fittest_individual_data, self.population_average_fitness_data

    # This method evolves the population employing the mechanism of the NEAT library.
    def evolve_population(self):

        self.reporters.start_generation(self.generation)
        average_fitness = 0.0

        best = None

        for g in self.population.values():

            if g.fitness is None:
                raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

            if best is None or g.fitness > best.fitness:
                best = g

            average_fitness += g.fitness

        self.reporters.post_evaluate(self.configuration, self.population, self.species, best)

        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best

        self.fittest_individual_data.append(self.best_genome.fitness)
        average_fitness = average_fitness / len(self.population)
        self.population_average_fitness_data.append(average_fitness)

        if not self.configuration.no_fitness_termination:

            fv = self.fitness_criterion(g.fitness for g in self.population.values())

            if fv >= self.configuration.fitness_threshold:
                self.reporters.found_solution(self.configuration, self.generation, best)

                return

        self.population = self.reproduction.reproduce(self.configuration, self.species, self.configuration.pop_size,
                                                      self.generation)

        if not self.species.species:

            self.reporters.complete_extinction()

            if self.configuration.reset_on_extinction:

                self.population = self.reproduction.create_new(self.configuration.genome_type,
                                                               self.configuration.genome_config,
                                                               self.configuration.pop_size)

            else:

                raise CompleteExtinctionException()

        self.species.speciate(self.configuration, self.population, self.generation)
        self.reporters.end_generation(self.configuration, self.population, self.species)

        self.generation += 1

        if self.configuration.no_fitness_termination:
            self.reporters.found_solution(self.configuration, self.generation, self.best_genome)

    # This method helps to build the configuration object required by the NEAT library.
    def __generate_configuration(self, conf, function_bank):

        configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, conf)
        configuration.genome_config.add_activation("neg_abs", function_bank.negative_abs)
        configuration.genome_config.add_activation("neg_square", function_bank.negative_square)
        configuration.genome_config.add_activation("sqrt_abs", function_bank.square_abs)
        configuration.genome_config.add_activation("neg_sqrt_abs", function_bank.negative_square_abs)
        configuration.genome_config.add_activation("neg_sin", function_bank.negative_sin)

        return configuration
