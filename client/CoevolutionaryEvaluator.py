# All the libraries required.
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import gmean
from scipy.stats import hmean
import requests
import random
import base64
import numpy
import json
import math
import neat


# This class evaluates individuals of SAMs and controllers populations. It receives all the data required to initialise
# the server. It also provides the reference data to generate SAMs simulations.
class CoevolutionEvaluator:

    MAPPING_REFERENCE = 0.5
    POSITIVE_MAPPING_REFERENCE = math.pi * 2
    NEGATIVE_MAPPING_REFERENCE = POSITIVE_MAPPING_REFERENCE * -1.0
    TWO_INTERACTIONS_WEIGHTS = [0.6, 0.4]
    THREE_INTERACTIONS_WEIGHTS = [0.5, 0.3, 0.2]
    FIVE_INTERACTIONS_WEIGHTS = [0.4, 0.3, 0.15, 0.1, 0.05]
    SEVEN_INTERACTIONS_WEIGHTS = [0.35, 0.25, 0.15, 0.12, 0.07, 0.04, 0.02]
    TEN_INTERACTIONS_WEIGHTS = [0.3, 0.2, 0.15, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01]

    FITTEST_INDIVIDUAL_DATA = "data_ fittest_individual"
    POPULATION_DATA = "population_average_fitness"

    def __init__(self, eval_data, eval_approach):

        self.target_url = None
        self.number_of_workers = None
        self.morphology_recurrent_topology = None
        self.controller_recurrent_topology = None
        self.evaluation_approach = eval_approach
        self.x_vector = None
        self.y_vector = None
        self.z_vector = None

        self.__configure_evaluator_and_server(eval_data)

    # This method evaluate both populations. It is used to initialise the coevolutionary run.
    def evaluate_populations_in_full(self, morphology_configuration, morphology_population, controller_configuration,
                                     controller_population):

        list_of_morphology_cppns = self.__build_cppns_from_genomes(morphology_configuration, morphology_population)

        list_of_catheter_morphologies = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for morphology in executor.map(self.build_catheter_morphology, list_of_morphology_cppns, chunksize=2):

                list_of_catheter_morphologies.append(morphology)

        list_of_controller_cppns = self.__build_cppns_from_genomes(controller_configuration, controller_population)

        matrix_of_fitness_values = []

        for catheter_morphology in list_of_catheter_morphologies:

            catheter_material_data = self.get_catheter_material_data(catheter_morphology)
            catheter_material_data_copies = []
            catheter_morphology_copies = []

            for _ in range(0, len(list_of_controller_cppns)):

                catheter_material_data_copies.append(catheter_material_data)
                catheter_morphology_copies.append(catheter_morphology)

            list_of_offsets = []

            with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

                for offset in executor.map(self.build_offset, catheter_material_data_copies,
                                           list_of_controller_cppns, chunksize=2):

                    list_of_offsets.append(offset)

            fitness_values = []

            with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

                for fitness_value in executor.map(self.evaluate_morphology_and_controller, catheter_morphology_copies,
                                                  list_of_offsets, chunksize=2):

                    fitness_values.append(fitness_value)

            matrix_of_fitness_values.append(fitness_values)

        morphology_fitness_values, controller_fitness_values = self.__get_morphology_and_controller_fitness_values(
                                                               matrix_of_fitness_values)

        self.__set_fitness_values_to_population(morphology_fitness_values, morphology_population)
        self.__set_fitness_values_to_population(controller_fitness_values, controller_population)

    # This method evaluates SAMs (or morphologies) with the n fittest controllers.
    def evaluate_n_fittest_controllers_all_morphologies(self, controller_configuration, fittest_controllers,
                                                        morphology_configuration, morphology_population):

        list_of_morphology_cppns = self.__build_cppns_from_genomes(morphology_configuration, morphology_population)

        list_of_catheter_morphologies = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for morphology in executor.map(self.build_catheter_morphology, list_of_morphology_cppns, chunksize=2):

                list_of_catheter_morphologies.append(morphology)

        list_of_catheters_material_data = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for morphology in executor.map(self.get_catheter_material_data, list_of_catheter_morphologies, chunksize=2):

                list_of_catheters_material_data.append(morphology)

        list_of_fittest_controller_cppns = self.__build_cppns_from_genomes(controller_configuration, fittest_controllers)

        matrix_of_fitness_values = []

        for fittest_controller_cppn in list_of_fittest_controller_cppns:

            copies_of_fittest_controller = []

            for _ in range(0, len(list_of_catheter_morphologies)):

                copies_of_fittest_controller.append(fittest_controller_cppn)

            list_of_offsets = []

            with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

                for offset in executor.map(self.build_offset, list_of_catheters_material_data,
                                           copies_of_fittest_controller, chunksize=2):

                    list_of_offsets.append(offset)

            morph_fitness_values = []

            with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

                for fitness_value in executor.map(self.evaluate_morphology_and_controller, list_of_catheter_morphologies,
                                                  list_of_offsets, chunksize=2):

                    morph_fitness_values.append(fitness_value)

            matrix_of_fitness_values.append(morph_fitness_values)

        morphology_fitness_values = self.__get_morphology_and_controller_fitness_values(matrix_of_fitness_values, "")
        self.__set_fitness_values_to_population(morphology_fitness_values, morphology_population)

    # This method evaluates controllers with the n fittest SAMs (or morphologies).
    def evaluate_n_fittest_morphologies_all_controllers(self, morphology_configuration, fittest_morphs,
                                                        controller_configuration, controller_population):

        list_of_controller_cppns = self.__build_cppns_from_genomes(controller_configuration, controller_population)

        fittest_morphology_cppns = self.__build_cppns_from_genomes(morphology_configuration, fittest_morphs)

        list_of_fittest_catheter_morphologies = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for morphology in executor.map(self.build_catheter_morphology, fittest_morphology_cppns, chunksize=2):

                list_of_fittest_catheter_morphologies.append(morphology)

        list_of_fittest_catheters_material_data = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for morphology in executor.map(self.get_catheter_material_data, list_of_fittest_catheter_morphologies,
                                           chunksize=2):

                list_of_fittest_catheters_material_data.append(morphology)

        matrix_of_fitness_values = []

        for fittest_material, fittest_morphology in zip(list_of_fittest_catheters_material_data,
                                                        list_of_fittest_catheter_morphologies):

            copies_of_fittest_material_data = []
            copies_of_fittest_morphology = []

            for _ in range(0, len(list_of_controller_cppns)):

                copies_of_fittest_material_data.append(fittest_material)
                copies_of_fittest_morphology.append(fittest_morphology)

            list_of_offsets = []

            with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

                for offset in executor.map(self.build_offset, copies_of_fittest_material_data, list_of_controller_cppns,
                                           chunksize=2):

                    list_of_offsets.append(offset)

            cont_fitness_values = []

            with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

                for fitness_value in executor.map(self.evaluate_morphology_and_controller, copies_of_fittest_morphology,
                                                  list_of_offsets, chunksize=2):

                    cont_fitness_values.append(fitness_value)

            matrix_of_fitness_values.append(cont_fitness_values)

        controller_fitness_values = self.__get_morphology_and_controller_fitness_values(matrix_of_fitness_values, "")

        self.__set_fitness_values_to_population(controller_fitness_values, controller_population)

    # This method received a CPPN as a paremeter, which is queried to generate the SAM.
    def build_catheter_morphology(self, cppn):

        morphology = []

        for z_coordinate in self.z_vector:

            layer = ""

            if z_coordinate == 0 or z_coordinate == (len(self.z_vector) - 1):

                x = len(self.x_vector)
                y = len(self.y_vector)
                layer += ("1" * (x * y))
                morphology.append(layer)

                continue

            for y_coordinate in self.y_vector:

                for x_coordinate in self.x_vector:

                    if x_coordinate == 0 or x_coordinate == len(self.x_vector) - 1:

                        layer += "1"

                        continue

                    if y_coordinate == 0 or y_coordinate == len(self.y_vector) - 1:

                        layer += "1"

                        continue

                    cpp_input = [x_coordinate, y_coordinate, z_coordinate]
                    cppn_output = cppn.activate(cpp_input)

                    vp = numpy.fabs(cppn_output[0])

                    if vp < self.MAPPING_REFERENCE:

                        layer += "0"

                    else:

                        m = numpy.fabs(cppn_output[1])

                        if m < self.MAPPING_REFERENCE:

                            layer += "1"

                        else:

                            layer += "3"

            morphology.append(layer)

        return morphology

    # This method receives the type of material a SAM is made of, and a CPPN that generates the phase offsets for the SAM.
    def build_offset(self, material_data, controller_cppn):

        offset = []

        for layer in material_data.keys():

            layer_data = material_data.get(layer)
            layer_offset = ""

            for coordinate in layer_data.keys():

                material_id = layer_data.get(coordinate)
                cppn_input = [coordinate[0], coordinate[1], coordinate[2], material_id]
                offset_value = controller_cppn.activate(cppn_input)

                if offset_value[0] >= self.POSITIVE_MAPPING_REFERENCE:

                    layer_offset += str(self.POSITIVE_MAPPING_REFERENCE) + ", "

                elif offset_value[0] <= self.NEGATIVE_MAPPING_REFERENCE:

                    layer_offset += str(self.NEGATIVE_MAPPING_REFERENCE) + ", "

                else:

                    layer_offset += str(offset_value[0]) + ", "

            layer_offset = layer_offset[:-2]
            offset.append(layer_offset)

        controller_cppn.reset()

        return offset

    # This method sends a SAM y a phase offset scenario to be evaluated in the server.
    def evaluate_morphology_and_controller(self, catheter_morphology, offset):

        catheter = {

            "evaluation": True,
            "layers": catheter_morphology,
            "offsets": offset

        }

        r = requests.get(url=self.target_url + "/biomeld-hn", json=catheter).json()
        z_trace = r.get("trace").get("z")
        displacement = float(z_trace.get("final")) - float(z_trace.get("initial"))

        if math.isnan(displacement):

            r = requests.get(url=self.target_url + "/biomeld-hn", json=catheter).json()
            z_trace = r.get("trace").get("z")
            second_displacement = float(z_trace.get("final")) - float(z_trace.get("initial"))

            if math.isnan(second_displacement):

                value = random.uniform(0.0025, 0.0045)

                return value

            else:

                return second_displacement

        else:

            return displacement

    # This method gathers the material typy of a SAM.
    def get_catheter_material_data(self, catheter_morphology):

        coordinates_and_material_id = {}
        z_counter = 0
        x_reference = len(self.x_vector)

        for layer in catheter_morphology:

            layer_dictionary = {}
            y_counter = 0
            x_counter = 0

            for material_id in layer:

                if material_id == "0":

                    layer_dictionary[(float(x_counter), float(y_counter), float(z_counter))] = 0.0

                elif material_id == "1":

                    layer_dictionary[(float(x_counter), float(y_counter), float(z_counter))] = 1.0

                else:

                    layer_dictionary[(float(x_counter), float(y_counter), float(z_counter))] = 3.0

                x_counter += 1

                if x_counter == x_reference:

                    x_counter = 0
                    y_counter += 1

            coordinates_and_material_id["layer_" + str(z_counter)] = layer_dictionary
            z_counter += 1

        return coordinates_and_material_id

    # This method generates the .vxa file of the fittest SAM.
    def get_fittest_individual_file(self, fittest_morphology, fittest_offset, collaboration_method, individuals, evaluation_approach):

        catheter = {

            "evaluation": False,
            "layers": fittest_morphology,
            "offsets": fittest_offset
        }

        r = requests.get(url=self.target_url + "/biomeld-hn", json=catheter).json()
        print(r)
        raw_file = r.get("individual_file")
        bytes_file = raw_file.encode()
        final_file = base64.b64decode(bytes_file)
        print(final_file)

        path = collaboration_method + "_n=" + str(individuals) + "_eval_approach_" + evaluation_approach + "_fittest_SAM.vxa"
        fitness_values = r.get("fitness_values")
        print("Number of voxels: ", fitness_values.get("voxels"))
        print("Displacement: ", fitness_values.get("displacement"))
        print("Instance used:", r.get("instance"))

        with open(path, "wb") as file:

            file.write(final_file)
            file.close()

    # This method generates a JSON file containing the aptitude value of the fittest individual throughout the evolutionary process.
    def get_fittest_individual_data(self, fittest_data, collaboration_method, individuals, evaluation_approach, population_name):

        path = (collaboration_method + "_n=" + str(individuals) + "_eval_approach_" + evaluation_approach + "_" +
                population_name + "_fittest_data_"  + ".json")

        data = {

            self.FITTEST_INDIVIDUAL_DATA: fittest_data[0],
            self.POPULATION_DATA: fittest_data[1]
        }

        with open(path, "w", encoding="utf-8") as file:

            json.dump(data, file, indent=4)
            file.close()

    # This method configure several elements of the evaluator. For instance the x,y,z reference vectors.
    def __configure_evaluator_and_server(self, eval_data):

        self.morphology_recurrent_topology = eval_data.get("recurrent_topology").get("morphology_cppns")
        self.controller_recurrent_topology = eval_data.get("recurrent_topology").get("controller_cppns")

        self.x_vector = [float(x) for x in range(0, eval_data.get("catheter_layout").get("x"))]
        self.y_vector = [float(y) for y in range(0, eval_data.get("catheter_layout").get("y"))]
        self.z_vector = [float(z) for z in range(0, eval_data.get("catheter_layout").get("z"))]

        self.target_url = eval_data.get("server_configuration").get("target_url") + "8000"
        self.number_of_workers = eval_data.get("server_configuration").get("simulator_instances")
        initial_port = eval_data.get("server_configuration").get("initial_port")

        for _ in range(0, self.number_of_workers):

            initialisation = {

                "voxels": (float(eval_data.get("catheter_layout").get("x")),
                           float(eval_data.get("catheter_layout").get("y")),
                           float(eval_data.get("catheter_layout").get("z")))
            }

            print(initialisation)
            r = requests.post(url=eval_data.get("server_configuration").get("target_url") + str(initial_port) +
                              "/biomeld-hn", json=initialisation)
            print(r.json())
            initial_port += 1

    # This method help to generate CPPNs from genomes, an object of NEAT library.
    def __build_cppns_from_genomes(self, configuration, genomes):

        list_of_cppns = []

        for genome_id, genome in genomes:

            if self.morphology_recurrent_topology:

                cppn = neat.nn.RecurrentNetwork.create(genome, configuration)
                list_of_cppns.append(cppn)

            else:

                cppn = neat.nn.FeedForwardNetwork.create(genome, configuration)
                list_of_cppns.append(cppn)

        return list_of_cppns

    # This method retrieves the fitness values (or aptitude) of individuals by using a specific evaluation approach.
    def __get_morphology_and_controller_fitness_values(self, matrix_of_fitness_values, fitness_flag="MC"):

        if fitness_flag == "MC":

            morphologies_fitness_values = self.__get_morphologies_fitness_values(matrix_of_fitness_values)
            controllers_fitness_values = self.__get_controllers_fitness_values(matrix_of_fitness_values)

            return morphologies_fitness_values, controllers_fitness_values

        else:

            if self.evaluation_approach == "arithmetic":

                fitness_values = self.__get_controllers_fitness_values(matrix_of_fitness_values)

            elif self.evaluation_approach == "weighted":

                fitness_values = self.__get_weighted_fitness_values(matrix_of_fitness_values)

            elif self.evaluation_approach == "geometric":

                fitness_values = self.__get_geometric_fitness_values(matrix_of_fitness_values)

            elif self.evaluation_approach == "harmonic":

                fitness_values = self.__get_harmonic_fitness_values(matrix_of_fitness_values)

            else:

                raise ValueError('Invalid approach: {0!r}'.format(self.evaluation_approach))

            return fitness_values

    # This method retrieves the fitness values of SAMs.
    def __get_morphologies_fitness_values(self, matrix_of_fitness_values):

        morphology_fitness_values = []

        for fitness_values in matrix_of_fitness_values:

            fitness_value = 0.0

            for value in fitness_values:

                fitness_value += value

            fitness_value = fitness_value / len(fitness_values)
            morphology_fitness_values.append(fitness_value)

        return morphology_fitness_values

    # This method calculates the fitness values of controllers using the arithmetic mean.
    def __get_controllers_fitness_values(self, matrix_of_fitness_values):

        controller_fitness_values = []

        for c in range(len(matrix_of_fitness_values[0])):

            fitness_value = 0.0

            for m in range(len(matrix_of_fitness_values)):

                fitness_value += matrix_of_fitness_values[m][c]

            fitness_value = fitness_value / len(matrix_of_fitness_values)
            controller_fitness_values.append(fitness_value)

        return controller_fitness_values

    # This method calculates the fitness values of individual using the weighted mean.
    def __get_weighted_fitness_values(self, matrix_of_fitness_values):

        fitness_values = []

        if len(matrix_of_fitness_values) == 1:

            for c in range(len(matrix_of_fitness_values[0])):

                fitness_value = 0.0

                for m in range(len(matrix_of_fitness_values)):

                    fitness_value += matrix_of_fitness_values[m][c]

                fitness_value = fitness_value / len(matrix_of_fitness_values)
                fitness_values.append(fitness_value)

        elif len(matrix_of_fitness_values) == 2:

            for c in range(len(matrix_of_fitness_values[0])):

                values_aux = []

                for m in range(len(matrix_of_fitness_values)):

                    values_aux.append(matrix_of_fitness_values[m][c])

                values_aux.sort(reverse=True)
                fitness_value = ((values_aux[0] * self.TWO_INTERACTIONS_WEIGHTS[0]) +
                                 (values_aux[1] * self.TWO_INTERACTIONS_WEIGHTS[1]))
                fitness_values.append(fitness_value)

        elif len(matrix_of_fitness_values) == 3:

            for c in range(len(matrix_of_fitness_values[0])):

                values_aux = []

                for m in range(len(matrix_of_fitness_values)):
                    values_aux.append(matrix_of_fitness_values[m][c])

                values_aux.sort(reverse=True)
                fitness_value = ((values_aux[0] * self.THREE_INTERACTIONS_WEIGHTS[0]) +
                                 (values_aux[1] * self.THREE_INTERACTIONS_WEIGHTS[1]) +
                                 (values_aux[2] * self.THREE_INTERACTIONS_WEIGHTS[2]))
                fitness_values.append(fitness_value)

        elif len(matrix_of_fitness_values) == 5:

            for c in range(len(matrix_of_fitness_values[0])):

                values_aux = []

                for m in range(len(matrix_of_fitness_values)):

                    values_aux.append(matrix_of_fitness_values[m][c])

                values_aux.sort(reverse=True)
                fitness_value = ((values_aux[0] * self.FIVE_INTERACTIONS_WEIGHTS[0]) +
                                 (values_aux[1] * self.FIVE_INTERACTIONS_WEIGHTS[1]) +
                                 (values_aux[2] * self.FIVE_INTERACTIONS_WEIGHTS[2]) +
                                 (values_aux[3] * self.FIVE_INTERACTIONS_WEIGHTS[3]) +
                                 (values_aux[4] * self.FIVE_INTERACTIONS_WEIGHTS[4]))

                fitness_values.append(fitness_value)

        elif len(matrix_of_fitness_values) == 7:

            for c in range(len(matrix_of_fitness_values[0])):

                values_aux = []

                for m in range(len(matrix_of_fitness_values)):

                    values_aux.append(matrix_of_fitness_values[m][c])

                values_aux.sort(reverse=True)
                fitness_value = ((values_aux[0] * self.SEVEN_INTERACTIONS_WEIGHTS[0]) +
                                 (values_aux[1] * self.SEVEN_INTERACTIONS_WEIGHTS[1]) +
                                 (values_aux[2] * self.SEVEN_INTERACTIONS_WEIGHTS[2]) +
                                 (values_aux[3] * self.SEVEN_INTERACTIONS_WEIGHTS[3]) +
                                 (values_aux[4] * self.SEVEN_INTERACTIONS_WEIGHTS[4]) +
                                 (values_aux[5] * self.SEVEN_INTERACTIONS_WEIGHTS[5]) +
                                 (values_aux[6] * self.SEVEN_INTERACTIONS_WEIGHTS[6]))

                fitness_values.append(fitness_value)

        elif len(matrix_of_fitness_values) == 10:

            for c in range(len(matrix_of_fitness_values[0])):

                values_aux = []

                for m in range(len(matrix_of_fitness_values)):
                    values_aux.append(matrix_of_fitness_values[m][c])

                values_aux.sort(reverse=True)
                fitness_value = ((values_aux[0] * self.TEN_INTERACTIONS_WEIGHTS[0]) +
                                 (values_aux[1] * self.TEN_INTERACTIONS_WEIGHTS[1]) +
                                 (values_aux[2] * self.TEN_INTERACTIONS_WEIGHTS[2]) +
                                 (values_aux[3] * self.TEN_INTERACTIONS_WEIGHTS[3]) +
                                 (values_aux[4] * self.TEN_INTERACTIONS_WEIGHTS[4]) +
                                 (values_aux[5] * self.TEN_INTERACTIONS_WEIGHTS[5]) +
                                 (values_aux[6] * self.TEN_INTERACTIONS_WEIGHTS[6]) +
                                 (values_aux[7] * self.TEN_INTERACTIONS_WEIGHTS[7]) +
                                 (values_aux[8] * self.TEN_INTERACTIONS_WEIGHTS[8]) +
                                 (values_aux[9] * self.TEN_INTERACTIONS_WEIGHTS[9]))
                fitness_values.append(fitness_value)

        # When any of these cases occurs, the weighted mean is calculated in a different manner. Arguably, this behaviour
        # is triggered when the NEAT library adds some individuals to the population.
        elif len(matrix_of_fitness_values) == 4 or len(matrix_of_fitness_values) == 6 or len(matrix_of_fitness_values) == 20:

            for c in range(len(matrix_of_fitness_values[0])):

                values_aux = []

                for m in range(len(matrix_of_fitness_values)):

                    values_aux.append(matrix_of_fitness_values[m][c])

                values_aux.sort(reverse=True)

                if len(values_aux) == 4:

                    first_value = (values_aux[0] + values_aux[1]) / 2
                    second_value = (values_aux[2] + values_aux[3]) / 2
                    fitness_value = ((first_value * self.TWO_INTERACTIONS_WEIGHTS[0]) +
                                     (second_value * self.TWO_INTERACTIONS_WEIGHTS[1]))
                    fitness_values.append(fitness_value)

                elif len(values_aux) == 6:

                    first_value = (values_aux[0] + values_aux[1]) / 2
                    second_value = (values_aux[2] + values_aux[3]) / 2
                    third_value = (values_aux[4] + values_aux[5]) / 2
                    fitness_value = ((first_value * self.THREE_INTERACTIONS_WEIGHTS[0]) +
                                     (second_value * self.THREE_INTERACTIONS_WEIGHTS[1]) +
                                     (third_value * self.THREE_INTERACTIONS_WEIGHTS[2]))
                    fitness_values.append(fitness_value)

                elif len(values_aux) == 20:

                    first_value = (values_aux[0] + values_aux[1]) / 2
                    second_value = (values_aux[2] + values_aux[3]) / 2
                    third_value = (values_aux[4] + values_aux[5]) / 2
                    fourth_value = (values_aux[6] + values_aux[7]) / 2
                    fifth_value = (values_aux[8] + values_aux[9]) / 2
                    sixth_value = (values_aux[10] + values_aux[11]) / 2
                    seventh_value = (values_aux[12] + values_aux[13]) / 2
                    eighth_value = (values_aux[14] + values_aux[15]) / 2
                    ninth_value = (values_aux[16] + values_aux[17]) / 2
                    tenth_value = (values_aux[18] + values_aux[19]) / 2
                    fitness_value = ((first_value * self.TEN_INTERACTIONS_WEIGHTS[0]) +
                                     (second_value * self.TEN_INTERACTIONS_WEIGHTS[1]) +
                                     (third_value * self.TEN_INTERACTIONS_WEIGHTS[2]) +
                                     (fourth_value * self.TEN_INTERACTIONS_WEIGHTS[3]) +
                                     (fifth_value * self.TEN_INTERACTIONS_WEIGHTS[4]) +
                                     (sixth_value * self.TEN_INTERACTIONS_WEIGHTS[5]) +
                                     (seventh_value * self.TEN_INTERACTIONS_WEIGHTS[6]) +
                                     (eighth_value * self.TEN_INTERACTIONS_WEIGHTS[7]) +
                                     (ninth_value * self.TEN_INTERACTIONS_WEIGHTS[8]) +
                                     (tenth_value * self.TEN_INTERACTIONS_WEIGHTS[9]))
                    fitness_values.append(fitness_value)

        else:

            print("Error")

            return

        return fitness_values

    # This method calculates the fitness values of controllers using the geometric mean.
    def __get_geometric_fitness_values(self, matrix_of_fitness_values):

        fitness_values = []

        for c in range(len(matrix_of_fitness_values[0])):

            aux_fitness_values = []

            for m in range(len(matrix_of_fitness_values)):

                if matrix_of_fitness_values[m][c] <= 0.0:

                    aux_fitness_values.append(0.00001)

                else:

                    aux_fitness_values.append(matrix_of_fitness_values[m][c])

            fitness_value = gmean(aux_fitness_values)
            fitness_values.append(fitness_value.item())

        return fitness_values

    # This method calculates the fitness values of controllers using the harmonic mean.
    def __get_harmonic_fitness_values(self, matrix_of_fitness_values):

        fitness_values = []

        for c in range(len(matrix_of_fitness_values[0])):

            aux_fitness_values = []

            for m in range(len(matrix_of_fitness_values)):

                if matrix_of_fitness_values[m][c] <= 0.0:

                    aux_fitness_values.append(0.00001)

                else:

                    aux_fitness_values.append(matrix_of_fitness_values[m][c])

            fitness_value = hmean(aux_fitness_values)
            fitness_values.append(fitness_value.item())

        return fitness_values

    # This method assigns the fitness values to individuals of the population.
    def __set_fitness_values_to_population(self, fitness_values, population):

        index = 0

        for genome_id, genome in population:

            genome.fitness = fitness_values[index]
            index += 1

