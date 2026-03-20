# All necessary libraries.
from CoevolutionaryPopulation import CoevolutionaryPopulation
from CoevolutionaryEvaluator import CoevolutionEvaluator
from ActivationFunctionBank import ActivationFunctionBank
from configupdater import ConfigUpdater
import pickle
import neat


# This is the main class. It orchestrates the execution of NEAT under a cooperative coevolutionary approach. It
# receives the configuration of CPPNs representing SAMs and controllers. Furthermore, it receives the data necessary to
# configure all the elements involved in the execution of the algorithm.
class CoevolutionaryAlgorithm:

    def __init__(self, conf_morph_path, conf_contr_path, eval_data):

        self.number_of_generations = None
        self.collaboration_method = eval_data.get("collaboration_method")
        self.number_of_collaborators = eval_data.get("number_of_collaborators")
        self.evaluation_approach = eval_data.get("evaluation_approach")
        self.function_bank = ActivationFunctionBank()
        self.morphology_population = CoevolutionaryPopulation(conf_morph_path, self.function_bank)
        self.controller_population = CoevolutionaryPopulation(conf_contr_path, self.function_bank)
        self.fitness_function = CoevolutionEvaluator(eval_data, self.evaluation_approach)

        self.__configure_files_and_coevolution_parameters(conf_morph_path, conf_contr_path, eval_data)

    # This algorithm controls the coevolutonary process between SAMs population and controllers population. This method
    # also saves the fittest SAM and the fittest controller.
    def execute_coevolution(self):

        generation_counter = 0
        evaluation_flag = "M"

        morphologies = self.morphology_population.get_population()
        morph_conf = self.morphology_population.get_configuration()
        controllers = self.controller_population.get_population()
        contr_conf = self.controller_population.get_configuration()
        self.fitness_function.evaluate_populations_in_full(morph_conf, morphologies, contr_conf, controllers)
        self.controller_population.set_best_genome()

        while generation_counter <= self.number_of_generations:

            if evaluation_flag == "M":

                self.morphology_population.evolve_population()
                fittest_controllers = self.controller_population.get_top_n_genomes(self.number_of_collaborators,
                                                                                   self.collaboration_method)
                morphologies = self.morphology_population.get_population()
                self.fitness_function.evaluate_n_fittest_controllers_all_morphologies(contr_conf, fittest_controllers,
                                                                                      morph_conf, morphologies)
                evaluation_flag = "C"

            elif evaluation_flag == "C":

                self.controller_population.evolve_population()
                fittest_morphologies = self.morphology_population.get_top_n_genomes(self.number_of_collaborators,
                                                                                    self.collaboration_method)
                controllers = self.controller_population.get_population()
                self.fitness_function.evaluate_n_fittest_morphologies_all_controllers(morph_conf, fittest_morphologies,
                                                                                      contr_conf, controllers)
                evaluation_flag = "M"

            else:

                raise ValueError("Invalid value for evaluation_flag")

            generation_counter += 1

        fittest_morphology_genome = self.morphology_population.get_best_genome()

        if self.fitness_function.morphology_recurrent_topology:

            fittest_morphology_cppn = neat.nn.RecurrentNetwork.create(fittest_morphology_genome, morph_conf)

        else:

            fittest_morphology_cppn = neat.nn.FeedForwardNetwork.create(fittest_morphology_genome, morph_conf)

        fittest_morphology = self.fitness_function.build_catheter_morphology(fittest_morphology_cppn)
        fittest_material_data = self.fitness_function.get_catheter_material_data(fittest_morphology)

        fittest_controller_genome = self.controller_population.get_best_genome()

        if self.fitness_function.controller_recurrent_topology:

            fittest_controller_cppn = neat.nn.RecurrentNetwork.create(fittest_controller_genome, contr_conf)

        else:

            fittest_controller_cppn = neat.nn.FeedForwardNetwork.create(fittest_controller_genome, contr_conf)

        fittest_offset = self.fitness_function.build_offset(fittest_material_data, fittest_controller_cppn)

        self.fitness_function.get_fittest_individual_file(fittest_morphology, fittest_offset, self.collaboration_method,
                                                          self.number_of_collaborators, self.evaluation_approach)
        self.fitness_function.get_fittest_individual_data(self.morphology_population.get_fitness_data(), self.collaboration_method,
                                                          self.number_of_collaborators, self.evaluation_approach, "morphology")
        self.fitness_function.get_fittest_individual_data(self.controller_population.get_fitness_data(), self.collaboration_method,
                                                          self.number_of_collaborators, self.evaluation_approach, "controller")


        path = (self.collaboration_method + "_n=" + str(self.number_of_collaborators) + "_eval_approach_" + self.evaluation_approach +
                "_fittest_controller_cppn_" +".pickle")

        with open(path, "wb") as file:

            pickle.dump(fittest_controller_cppn, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

    # This method helps the set values of the CPPN files associated to SAMs and controllers.
    def __configure_files_and_coevolution_parameters(self, morphology_path, controller_path, eval_data):

        self.number_of_generations = eval_data.get("ga_parameters").get("number_of_generations") * 2
        updater = ConfigUpdater()
        updater.read(morphology_path)
        updater["NEAT"]["pop_size"] = eval_data.get("ga_parameters").get("morphology_population_size")
        updater.update_file()
        updater.read(controller_path)
        updater["NEAT"]["pop_size"] = eval_data.get("ga_parameters").get("controller_population_size")
        updater.update_file()


