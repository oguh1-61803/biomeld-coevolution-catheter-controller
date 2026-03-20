> # Requirements for the client side

The code related to the client side was written in **Python 3.11**.

> Libraries
* scipy == 1.13.0
* requests == 2.31.0
* neat-python-2023 == 0.93
* ConfigUpdater == 3.2

> # Requirements for the server side

The code related to the server side was written in **Python 3.10**.

> Libraries
* tornado == 6.3.3 
* lxml == 4.9.3

> # Other important considerations

* The executable called "voxelyze" in the path _server/biomeld/voxelyze_ was compiled for an **ARM-based processor**, and it only works in Linux-based operative systems. If your server has an x86-based or amd64-based processor and it has a different operative system, you need to compile your own executable using the code contained in the following repository:
  
  https://github.com/skriegman/reconfigurable_organisms
  
* Don't forget to consider the two modifications to the source code of the physics engine mentioned in the README file!

* Lines 18-22 of the "Main.py" file of the server-side code provide a suggestion regarding the server deployment.
