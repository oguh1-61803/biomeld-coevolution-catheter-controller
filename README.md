> # Coevolutionary-bsed desing of soft actuator morphologies for catheters and their controllers

This implementation utilises Neuroevolution of Augmenting Topologies (NEAT) to design controllers of soft actuator morphologies (SAMs) focused on catheters and their controllers. The implementation operates under a cooperative coevolutionary scheme. SAMs and the effect induced by controllers are simulated in a physics engine called Voxelyze, which can be found in the following GitHub repository:

https://github.com/skriegman/reconfigurable_organisms

To adapt the physics engine to the dynamics of SAMs, two modifications to the source code were performed. The modifications can be found in:

* https://github.com/Antisthenis/reconfigurable_organisms/commit/80ae5d9af6f381d565fa4885ba1672f2813a8a28

* https://github.com/Antisthenis/reconfigurable_organisms/commit/dfcf21dcd0670a0f77d94e826563d09ed82a3786#diff-eed874d9aea4ad5f133265b296188a268bcc85f6f610b9c03ca3ada556cf88f5

> **Architecture**

Since the evolutionary process implies a simulation task, the runtime takes significant time. This software has been designed to reduce the time spent finding suitable SAMs and their controllers. It uses concurrency and was designed under a client-server architecture. Generally, the cooperative coevolutionary genetic algorithm (CCGA) is executed on the client side, whereas the core of the fitness function (Voxelyze) is executed on the server side.

The software of the client side was written in Python 3.11, and Python 3.10 was employed for the code of the server side.

> **Repository Structure**

The source code of this repository is split into two:

* _client_: code related to client side. It contains the implementation related to NEAT under cooperative coevolution.
* _server_: code related to the server side. It contains the implementation related to Voxelyze.

The packages used for the client and server sides are listed in the file called "requirements.md".

> **Important Notice**

The code provided in this repository was used as part of an academic research documented in:

* https://doi.org/10.1145/3712255.3726671
* https://doi.org/10.1007/978-3-032-15635-8_8

Furthermore, this project has received funding from the European Union’s Horizon Europe Research and Innovation programme under grant agreement No. 101070328.UWE researchers were funded by the UK Researchand Innovation grant No. 10044516. 
 
