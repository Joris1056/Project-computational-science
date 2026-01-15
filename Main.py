
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
        self.make_param('k', 7)  # states
        self.make_param('infection_p_stage1', 0.1)
        self.make_param('infection_p_stage2', 0.2)
        self.make_param('infection_p_stage3', 0.3)
        self.make_param('infection_p_stage4', 0.4)
        self.make_param('infection_p_stage5', 0.5)
        self.make_param('degeneration_p_stage1', 0.1)
        self.make_param('degeneration_p_stage2', 0.2)
        self.make_param('degeneration_p_stage3', 0.3)
        self.make_param('degeneration_p_stage4', 0.4)
        self.make_param('degeneration_p_stage5', 0.5)
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
        import matplotlib
        import matplotlib.pyplot as plt
        
        plt.cla()
        # 3. Use plt.imshow() to render self.config
        # Note: Set vmin and vmax to keep the color scale consistent
        cmap = plt.get_cmap('YlOrRd')
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=cmap)
        plt.axis('image')
        plt.title(f'Time step: {self.t}')

    def get_neighbours(self, y, x):
        """
        Helper function to retrieve the states of the 8 neighbors (Moore neighborhood).
        Tip: Use the modulo operator (%) to handle periodic boundary conditions (wrapping).
        """
        set_neighbours = set()
        for dx in range(-1, 2):
            for dy in range(-1,2):
                tuple_neighbour = ((x+dx)%self.width, (y+dy)%self.height)
                if not (dx == 0 and dy == 0):
                    set_neighbours.add(tuple_neighbour)
    
        return list(set_neighbours)
    
    def calculate_new_cell_value(self, current_value, neighbour_values):
        
        # dead cell remains dead
        if current_value == 6:
            new_value = 6
            return new_value
        
        # if current cell is healthy chance of getting starting degeneration
        elif current_value == 0:
            sick_neighbours = []
            for value in neighbour_values:
                if value >= 1:
                    sick_neighbours.append(value)
            
            p_no_infection_one_cell = []
            for value in sick_neighbours:
                if value == 1:
                    p_no_infection_one_cell.append(1-self.infection_p_stage1)
                elif value == 2:
                    p_no_infection_one_cell.append(1-self.infection_p_stage2)
                elif value == 3:
                    p_no_infection_one_cell.append(1-self.infection_p_stage3)
                elif value == 4:
                    p_no_infection_one_cell.append(1-self.infection_p_stage4)
                elif value == 5:
                    p_no_infection_one_cell.append(1-self.infection_p_stage5)
            
            p_no_infection = 1
            for i in range(0,len(p_no_infection_one_cell)):
                p_no_infection *= p_no_infection_one_cell[i]

            # chance of healthy cell getting infected
            p_infection = 1 - p_no_infection

            if np.random.random() < p_infection:
                new_value = 1
                return new_value
            else:
                new_value = 0
                return new_value
        
        # internal progression
        p_degeneration = 0
        if current_value != 0 and current_value != 6:
            if current_value == 1:
                    p_degeneration = self.degeneration_p_stage1
            elif current_value == 2:
                    p_degeneration = self.degeneration_p_stage2
            elif current_value == 3:
                    p_degeneration = self.degeneration_p_stage3
            elif current_value == 4:
                    p_degeneration = self.degeneration_p_stage4
            elif current_value == 5:
                    p_degeneration = self.degeneration_p_stage5

            if np.random.random() < p_degeneration:
                    new_value = current_value + 1
            else:
                new_value = current_value
            
        return new_value



    def step(self):
        """
        Contains the transition rules for one time step.
        Each step represents a progression in time (e.g., one day or week).
        """
        self.t += 1
        new_config = np.copy(self.config)
        for x in range(self.width):
            for y in range(self.height):
                cell_value = self.config[y,x]
                list_neighbour_value = []
                for neighbour_x, neighbour_y in self.get_neighbours(y,x):
                    value = self.config[neighbour_y,neighbour_x]
                    list_neighbour_value.append(value)
                new_config[y,x] = self.calculate_new_cell_value(cell_value,list_neighbour_value)
        
        self.config = new_config
        
        return False

if __name__ == '__main__':
    from pyics import GUI
    sim = ParkinsonSim()
    cx = GUI(sim)
    cx.start()

