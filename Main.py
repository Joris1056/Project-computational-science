
# Main

import numpy as np
from pyics import Model

class ParkinsonSim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.config = None

        # 1. Define your parameters here
        # These will appear as sliders/input fields in the GUI
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('k' = ....)
        self.make_param('r' = ....)
        self.make_param('infection_p_stage1', 0.1)
        self.make_param('infection_p_stage2', 0.2)
        self.make_param('infection_p_stage3', 0.3)
        self.make_param('infection_p_stage4', 0.4)
        self.make_param('infection_p_stage5', 0.5)
        # SUGGESTION: Add parameters like 'p_spontaneous', 'p_infection', etc.

    def reset(self):
        """Initializes or resets the simulation state."""
        self.t = 0
        
        # 2. Initialize your grid here (e.g., all zeros for healthy)
        # You could also 'infect' one cell in the center to start the process
        self.config = np.zeros((self.height,self.width))
        middle_x = self.width//2
        middle_y = self.height//2
        self.config[middle_y,middle_x] = 1

    def draw(self):
        """Handles the visualization of the grid."""
        import matplotlib.pyplot as plt
        
        plt.cla()
        # 3. Use plt.imshow() to render self.config
        # Note: Set vmin and vmax to keep the color scale consistent
        ...
        plt.title(f'Time step: {self.t}')

    def get_neighbors(self, y, x):
        """
        Helper function to retrieve the states of the 8 neighbors (Moore neighborhood).
        Tip: Use the modulo operator (%) to handle periodic boundary conditions (wrapping).
        """
        pass

    def step(self):
        """
        Contains the transition rules for one time step.
        Each step represents a progression in time (e.g., one day or week).
        """
        self.t += 1
        
        # Create a copy to store the updates for the next state
        # We must use the 'old' state to calculate all 'new' states simultaneously
        new_config = np.copy(self.config)

        # 4. Iterate through every cell in the grid (nested for-loop)
        # 5. Apply your logic based on the current state:
        #    - Healthy cells: Check if they get infected by neighbors or age
        #    - Sick cells: Advance their degradation level
        #    - Dead cells: Remain dead (or perhaps leave a gap)
        
        # 6. Use stochastic logic (probability):
        #    if np.random.random() < calculated_probability:
        #        ... update state ...

        self.config = new_config
        
        # Return True to stop the simulation, False to keep running
        return False

if __name__ == '__main__':
    from pyics import GUI
    sim = ParkinsonSim()
    cx = GUI(sim)
    cx.start()

