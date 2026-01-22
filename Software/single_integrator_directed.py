#!/usr/bin/env python3

##########################################################
#                                                        #
# Single Integrator Formation Control - Directed Graph   #
# Code Author: João Zenário                              #
#                                                        #     
##########################################################

from formation_controller import BROV_Formation 
from parameters import Parameters
from plot_animator import Animation

class Simulation:
    #* Function to setup all classes in order to run the simulation.
    def __init__(self):
        # 1. Initialize all parameters and variables.
        self.params = Parameters() 
        
        # 2. Setup the controller.
        self.controller = BROV_Formation()

        # 3. Setup the animation.
        self.animator = Animation(self.params)

    #* Function to run the animation.
    def run(self):
        # Start the Matplotlib event loop.
        self.animator.plot_animation(self.params, self.controller)

#* Main.
def main():
    sim = Simulation() # Create the simulation object.
    sim.run() # Run it.

if __name__ == "__main__":
    main()