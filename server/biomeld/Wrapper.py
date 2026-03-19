# All necessary libraries and imports from other files.
import tornado.escape
import tornado.web
import base64


# This class initialises the Voxelyze instance and receives the phase offset (and morphologies) and returns the
# displacement reached by the morphology and the number of voxels of the morphology, which are used for evaluating
# the phase offset.
class Wrapper(tornado.web.RequestHandler):

    # This method sets the number of evaluator.
    def initialize(self, evaluator):

        self.evaluator = evaluator

    # This method inialises the evaluator.
    def post(self):

        data = tornado.escape.json_decode(self.request.body)
        # print(data)
        self.evaluator.initialise_xml_tree(data)
        self.write({"status": "Evaluator (HyperNEAT - Catheter) %s initialised." % self.evaluator.evaluator_id})

    # This method received the data related to the phase offset to either evaluate it or to generate the .vxa file
    # containing the morphology.
    def get(self):

        data = tornado.escape.json_decode(self.request.body)
        self.evaluator.create_catheter_file(data)
        fitness_values = self.evaluator.evaluate_individual()

        if data.get("evaluation"):

            self.write(fitness_values)

        else:

            with open(self.evaluator.get_individual_file_path(), "rb") as file:

                f = base64.b64encode(file.read())
                self.write({"individual_file": f.decode(), "fitness_values": fitness_values, "instance": self.evaluator.evaluator_id})
