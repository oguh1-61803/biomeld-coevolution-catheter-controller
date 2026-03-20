# All necessary libraries.
from CoevolutionaryAlgorithm import CoevolutionaryAlgorithm


if __name__ == '__main__':

    # Paths of the configuration files for CPPNs of each population.
    morphologies_path = "configuration_files/catheter_NEAT.cfg"
    controllers_path = "configuration_files/controller_NEAT.cfg"

    # All the data necessary to execute coevolution.
    evaluator_data = {

        # Evolutionary parameters.
        "ga_parameters": {
            #200
            "number_of_generations": 5,
            "morphology_population_size": 25,
            "controller_population_size": 25,
        },

        # Recurrent topologies for CPPNs of each population.
        "recurrent_topology": {

            "morphology_cppns": True,
            "controller_cppns": True

        },

        # Server configuration parameters.
        "server_configuration": {

            "simulator_instances": 14,
            "initial_port": 8081,
            "target_url": "http://10.211.55.5:"
        },

        # Catheter dimensions in terms of voxels.
        "catheter_layout": {

            "x": 20,
            "y": 8,
            "z": 8
        },

        # Four different collaboration methods are implemented:
        # (i)   Best and worst versus all: best_and_worst_vs_all
        # (ii)  Fittest versus all: fittest_vs_all
        # (iii) Random versus all: random_vs_all
        # (iv)  Worst versis all: worst_vs_all
        "collaboration_method": "fittest_vs_all",

        # The set of values for the number of collaborators is: 2, 3, 5, 7, 10.
        "number_of_collaborators": 5,

        # Four evaluation approaches are implemented:
        # (i)   Based on the arithmetic mean: arithmetic
        # (ii)  Based on the weighted mean: weighted
        # (iii) Based on the geometric mean: geometric
        # (iv)  Based on the harmonic mean: harmonic
        "evaluation_approach": "weighted"
    }

    # IMPORTANT NOTES:
    #
    # The experimental set-up described in the paper called "Designing morphologies of soft medical devices using
    # cooperative neuro coevolution", whose link is https://dl.acm.org/doi/10.1145/3712255.3726671,
    # the configuration for the evaluation approach (evaluation_approach) is fixed with the value "arithmetic".
    #
    # The experimental set-up described in the paper called "Evaluating Fitness Averaging Strategies in Cooperative
    # NeuroCoEvolution for Automated Soft Actuator Design", whose link is https://doi.org/10.1007/978-3-032-15635-8_8,
    # the configuration for the collaboration method (collaboration_method) is fixed with the value "fittest_vs_all".

    coa = CoevolutionaryAlgorithm(morphologies_path, controllers_path, evaluator_data)
    coa.execute_coevolution()
