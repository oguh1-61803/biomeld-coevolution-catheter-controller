> # Coevolutionary-bsed desing of soft actuator morphologies for catheters and their controllers

This implementation utilises Neuroevolution of Augmenting Topologies (NEAT) to design controllers of soft actuator morphologies (SAMs) focused on catheters and their controllers. The implementation operates under a cooperative coevolutionary scheme. SAMs and the effect induced by controllers are simulated in a physics engine called Voxelyze, which can be found in the following GitHub repository:

https://github.com/skriegman/reconfigurable_organisms

To adapt the physics engine to the dynamics of SAMs, two modifications to the source code were performed. The modifications can be found in:

* https://github.com/Antisthenis/reconfigurable_organisms/commit/80ae5d9af6f381d565fa4885ba1672f2813a8a28
* https://github.com/Antisthenis/reconfigurable_organisms/commit/dfcf21dcd0670a0f77d94e826563d09ed82a3786#diff-eed874d9aea4ad5f133265b296188a268bcc85f6f610b9c03ca3ada556cf88f5
