# All necessary libraries and imports from other files.
from server.biomeld.ContextManager import ContextManager
from lxml import etree
import subprocess
import re


# This class is focused on the evaluation of phase offsets by using catheter morphologies.
class Evaluator:

    # Necessary constants to evaluate phase offsets.
    X_VOXELS = "X_Voxels"
    Y_VOXELS = "Y_Voxels"
    Z_VOXELS = "Z_Voxels"
    INITIAL_INDEX = 0
    FINAL_INDEX = 5

    # If needed, don't forget to update the path of the files!
    PREFIX_PATH = "/home/BioMeld_Catheter/src/biomeld/"
    VOXELYZE_PATH = PREFIX_PATH + "voxelyze/voxelyze_catheter"
    VOXELYZE_COMMAND = "./voxelyze_catheter -f "
    CATHETER_PATH = PREFIX_PATH + "voxelyze/catheter/base_controller.vxa"
    EVALUATOR_PATH = PREFIX_PATH + "evaluators/"
    EVALUATOR_WORD = "evaluator_"
    FITNESS_FILE = "fitness.xml"

    INVALID_PATTERN = re.compile(r"\b(0)\1+\b")

    # Constructor
    def __init__(self, evaluator):

        self.parser = etree.XMLParser(remove_blank_text=True)
        self.raw_tree = etree.parse(self.CATHETER_PATH, self.parser)
        self.root = self.raw_tree.getroot()
        self.number_of_hard_voxels = 0
        self.number_of_soft_voxels = 0
        self.number_of_no_voxels = 0
        self.evaluator_id = str(evaluator)
        print(self.evaluator_id)

    # This method initialises the file that will be used to evaluate phase offsets and generates a directory where one
    # Voxelyze is deployed.
    def initialise_xml_tree(self, conf):

        self.root.find("VXC").find("Structure").find(self.X_VOXELS).text = str(conf.get("voxels")[0])
        self.root.find("VXC").find("Structure").find(self.Y_VOXELS).text = str(conf.get("voxels")[1])
        self.root.find("VXC").find("Structure").find(self.Z_VOXELS).text = str(conf.get("voxels")[2])
        self.__clean_data()
        self.root.find("Simulator").find("GA").find("FitnessFileName").text = "fitness.xml"

        with ContextManager(self.EVALUATOR_PATH):

            command = "mkdir " + self.EVALUATOR_WORD + self.evaluator_id + "/"
            subprocess.run(command, shell=True)

        command = "cp " + self.VOXELYZE_PATH + " " + self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/"
        subprocess.run(command, shell=True)

    # This method receives the data necessary to populate the .vxa file, which interacts with the Voxelyze instance.
    def create_catheter_file(self, individual_data):

        self.__set_layers(individual_data.get("layers"))
        self.__set_offsets(individual_data.get("offsets"))

        with open(self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/individual_file.vxa", 'wb') as f:

             self.raw_tree.write(f, encoding="ISO-8859-1", pretty_print=True)

    # This method evaluates the phase offset using morphologies through a Voxelyze instance. It returns the maximum
    # displacement in the yz plane and the number of voxels, including the type of voxels.
    def evaluate_individual(self):

        aptitude = {"voxels": {"total": -1}, "displacement": None, "trace": {}}

        if self.__check_morphology_consistency():

            with ContextManager(self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/"):

                command = self.VOXELYZE_COMMAND + "individual_file.vxa"
                subprocess.run(command, shell=True)

            with ContextManager(self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/"):

                aptitude_tree = etree.parse(self.FITNESS_FILE, self.parser)
                root = aptitude_tree.getroot()
                aptitude["voxels"] = {"total": root.find("Fitness").find("VoxelNumber").text}
                aptitude["displacement"] = root.find("Fitness").find("normAbsoluteDisplacement").text
                aptitude["trace"]["y"] = {"initial": root.find("CMTrace")[self.INITIAL_INDEX].find("TraceY").text, "final": root.find("CMTrace")[self.FINAL_INDEX].find("TraceY").text}
                aptitude["trace"]["z"] = {"initial": root.find("CMTrace")[self.INITIAL_INDEX].find("TraceZ").text, "final": root.find("CMTrace")[self.FINAL_INDEX].find("TraceZ").text}

            self.__count_number_of_voxels()

            aptitude["voxels"]["hard"] = self.number_of_hard_voxels
            aptitude["voxels"]["soft"] = self.number_of_soft_voxels
            aptitude["voxels"]["absence"] = self.number_of_no_voxels

        self.__clean_data()

        return aptitude

    # This function returns the path of the morphology file (a .vxa file). The path includes the number of instance that
    # evaluated the morphology
    def get_individual_file_path(self):

        return self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/individual_file.vxa"

    # This method adds the data of the morphology that will be simulated.
    def __set_layers(self, individual_layers):

        for layer in individual_layers:

            l = etree.SubElement(self.root.find("VXC").find("Structure").find("Data"), "Layer")
            l.text = etree.CDATA(layer)

    # This method adds the phase offsets that will be evaluated.
    def __set_offsets(self, individual_offsets):

        for offsets in individual_offsets:

            o = etree.SubElement(self.root.find("VXC").find("Structure").find("PhaseOffset"), "Layer")
            o.text = etree.CDATA(offsets)

    # This method counts the number of voxels that compose a SAM.
    def __count_number_of_voxels(self):

        for layer in self.root.find("VXC").find("Structure").find("Data"):

            for number in layer.text:

                if number == "0":

                    self.number_of_no_voxels += 1
                    continue

                if number == "1":

                    self.number_of_hard_voxels += 1
                    continue

                if number == "3":

                    self.number_of_soft_voxels += 1
                    continue

    # This validates that the structure of the morphology has at least 1 voxel.
    def __check_morphology_consistency(self):

        inconsistency_counter = 0
        consistency_reference = len(self.root.find("VXC").find("Structure").find("Data"))

        for layer in self.root.find("VXC").find("Structure").find("Data"):

            if re.search(self.INVALID_PATTERN, layer.text):

                inconsistency_counter += 1

        if inconsistency_counter == consistency_reference:

            return False

        else:

            return True

    # This method sets the number of voxels to zero and delete the data related to the SAM.
    def __clean_data(self):

        for child in self.root.find("VXC").find("Structure").find("Data"):

            self.root.find("VXC").find("Structure").find("Data").remove(child)

        for child in self.root.find("VXC").find("Structure").find("PhaseOffset"):

            self.root.find("VXC").find("Structure").find("PhaseOffset").remove(child)

        self.number_of_hard_voxels = 0
        self.number_of_soft_voxels = 0
        self.number_of_no_voxels = 0

