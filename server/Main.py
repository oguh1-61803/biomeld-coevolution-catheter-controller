# All necessary libraries and imports from other files.
from server.biomeld.Evaluator import Evaluator
from server.biomeld.Wrapper import Wrapper
import tornado.ioloop
import tornado.web
import sys


# This function creates the evaluator and wraps it into a web application.
def build_evaluator(evaluator_id):

    eva = Evaluator(evaluator_id)
    application = tornado.web.Application([(r"/biomeld-hn", Wrapper, {"evaluator": eva})])

    return application


# Voxelyze instances are behind a reverse proxy to balance the workload. Nginx is suggested to deploy a reverse proxy.
# Furthermore, each Voxelyze instance is managed by a process control system. The recommended software to manage the
# instances is Supervisor, which is available for Linux-based operative systems and Windows.
# IMPORTANT: The number of instances deployed should be the same number as the value set in the variable called
# "simulator_instances" in the main file of the client-side code.
if __name__ == "__main__":

    evaluator = build_evaluator(sys.argv[1])
    evaluator.listen(sys.argv[2])
    tornado.ioloop.IOLoop.instance().start()
