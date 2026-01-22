
# Main

import numpy as np
from pyics import Model

class ParkinsonSim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.config = None
        self.sensitivity_matrix = None
        self.neuron_death = []
        self.time = []
        self.time_years =  []
        self.year_per_step = None
        self.t_70 = None
        self.t_30 =None
        self.t_0 = None

        #Here we are defining all the parameters

        #size of the figure
        self.make_param('width', 150)
        self.make_param('height', 150)

        #amount of states
        self.make_param('k', 7)  

        #the probability of infection per stage
        self.make_param('infection_p_stage1', 0.02)
        self.make_param('infection_p_stage2', 0.04)
        self.make_param('infection_p_stage3', 0.08)
        self.make_param('infection_p_stage4', 0.16)
        self.make_param('infection_p_stage5', 0.32)

        #the probability of degeneration per stage
        self.make_param('degeneration_p_stage1', 0.02)
        self.make_param('degeneration_p_stage2', 0.04)
        self.make_param('degeneration_p_stage3', 0.08)
        self.make_param('degeneration_p_stage4', 0.16)
        self.make_param('degeneration_p_stage5', 0.32)

        #maybe if there is spontaneous degeneration then we have p spon deg.
        self.make_param('p_spontaneous_degeneration', 0)
        self.make_param('lateral_base_multiplier', 1)
        self.make_param('lateral_ratio_multiplication', 4)
        self.make_param('ventral_base_multiplier', 1)
        self.make_param('ventral_ratio_multiplication', 5)
        self.make_param('dead_neighbour_multiplier', 0.05)
        

    def reset(self):
        """Initializes or resets the simulation state."""
        import matplotlib.pyplot as plt
        self.t = 0
        self.time = []
        self.neuron_death = []
        self.time_years = []
        self.year_per_step = None
        self.t_70 = None
        self.t_30 = None
        self.t_0 = None
        
        # old way:
        # 2. Initialize your grid here (e.g., all zeros for healthy)
        # You could also 'infect' one cell in the center to start the process
        #self.config = np.zeros((self.height,self.width))
        #middle_x = self.width//2
        #middle_y = self.height//2
        #self.config[middle_y,middle_x] = 1


        self.config = np.full((self.height, self.width), -1.0)
        
        center_x = self.width * 0.35
        for y in range(self.height):
            for x in range(self.width):
                nx = x / self.width
                curve = (self.height * 0.2) + ((x - center_x)**2 / (self.width * 0.8)) - (x * 0.15)
                min_dikte = self.height * 0.1
                max_dikte = self.height * 0.25
                dikte = min_dikte + (nx * (max_dikte - min_dikte))
                
                if y > curve and y < curve + dikte:
                    self.config[y, x] = 0
        
        self.sn_bounds = {}

        for x in range(self.width):
            ys = np.where(self.config[:,x] != -1)[0]
            if len(ys) > 0:
                y_min = ys.min()
                y_max = ys.max()
                self.sn_bounds[x] = (y_min,y_max)
        
        self.sensitivity_matrix = np.full((self.height, self.width), 0)
        
        
        for y in range(self.height):
            for x in range(self.width):
                if self.config[y,x] != -1:
                    ratio_x = x/self.width
                    y_min,y_max = self.sn_bounds[x]
                    relative_y = (y-y_min)/(y_max-y_min)
                    x_multiplier = self.lateral_base_multiplier + (ratio_x**2 * self.lateral_ratio_multiplication)
                    y_multiplier = self.ventral_base_multiplier + ((1 - relative_y)**2 * self.ventral_ratio_multiplication)
                    self.sensitivity_matrix[y,x] = x_multiplier * y_multiplier

        found = False
        for x in range(self.width - 1, 0, -1):
            for y in range(self.height):
                # Check of dit punt in onze SN-vorm ligt
                if self.config[y, x] == 0:
                    # Maak deze cel stadium 1 (beginnende degeneratie)
                    self.config[y, x] = 1
                    neighbours = self.get_neighbours(y,x)
                    for x,y in neighbours:
                        self.config[y,x] = 1
                    found = True
                    break
            if found: 
                break
        
        self.fig_neuron_alive, self.ax_neuron_alive = plt.subplots()
        self.line_neuron_alive, = self.ax_neuron_alive.plot([], [])
        self.ax_neuron_alive.set_ylabel('% live Neurons')
        self.ax_neuron_alive.set_title('Neurons alive Over Time')
        self.ax_neuron_alive.legend()



    def draw(self):
        """Handles the visualization of the grid."""
        import matplotlib
        import matplotlib.pyplot as plt

        mask = (self.config != -1)
        number_neurons = np.sum(mask)

        number_dead_neurons = np.sum(self.config == 6)

        total_neuron = 4.5*10**5
        neuron_per_cel = total_neuron/number_neurons
        perc_dead_neurons = round((number_dead_neurons/number_neurons) * 100,2)
        neuron_representation = f'One cell = {int(neuron_per_cel)} dopaminergic neurons'
        percentage_dead_neurons = f'{perc_dead_neurons}% neurons dead'        
        plt.cla()
        # 3. Use plt.imshow() to render self.config
        # Note: Set vmin and vmax to keep the color scale consistent
        cmap = plt.get_cmap('YlOrRd')
        cmap.set_under('lightgrey')
        plt.imshow(self.config, origin = 'lower', vmin=0, vmax=self.k - 1,
                cmap=cmap)
        plt.axis('image')
        plt.title(f'Substantia Nigra (coronal view) Time step: {self.t}')
        plt.xlabel('medial --> lateral')
        plt.ylabel('ventral <--> dorsal')
        plt.text(0.02, 0.95, neuron_representation, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.text(0.02, 0.85, percentage_dead_neurons, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


        if self.year_per_step is not None:
            self.line_neuron_alive.set_data(self.time_years, 100 - np.array(self.neuron_death))
        else:
            self.line_neuron_alive.set_data(self.time, 100 - np.array(self.neuron_death))
        
        for line in self.ax_neuron_alive.lines[1:]:
            line.remove()

        if self.t_30 != None and self.year_per_step == None:
            self.ax_neuron_alive.axvline(x = self.t_30, color = 'red', label = f't2 = {self.t_30}, 30% left')
        
        if self.t_30 != None and self.year_per_step != None:
            self.ax_neuron_alive.axvline(x = self.t_30 * self.year_per_step, color = 'black', label = f'{round(self.t_30*self.year_per_step,1)}th year, 30% left')
        
        if self.t_70 != None and self.year_per_step == None:
            self.ax_neuron_alive.axvline(x = self.t_70, color = 'green', label = f't1 = {self.t_70}, 70% left')
        
        if self.t_70 != None and self.year_per_step != None:
            self.ax_neuron_alive.axvline(x = self.t_70 * self.year_per_step, color = 'black', label = f'{round(self.t_70*self.year_per_step,1)}th year, 70% left')
        
        if self.t_0 != None and self.year_per_step == None:
            self.ax_neuron_alive.axvline(x = self.t_0, color = 'lightblue', label = f't3 = {self.t_0}, 0% left')
        
        if self.t_0 != None and self.year_per_step != None:
            self.ax_neuron_alive.axvline(x = self.t_0*self.year_per_step, color = 'black', label = f'{round(self.t_0*self.year_per_step,1)}th year, 1% left')
        
        if self.year_per_step == None:
            self.ax_neuron_alive.set_xlabel('Time step')
        if self.year_per_step != None:
            self.ax_neuron_alive.set_xlabel('Time in years')
        self.ax_neuron_alive.relim()
        self.ax_neuron_alive.autoscale_view()
        self.ax_neuron_alive.legend()
        plt.pause(0.01)


    def get_neighbours(self, y, x):
        """
        Helper function to retrieve the states of the 8 neighbors (Moore neighborhood).
        """
        set_neighbours = set()
        for dx in range(-1, 2):
            for dy in range(-1,2):
                tuple_neighbour = ((x+dx)%self.width, (y+dy)%self.height)
                if not (dx == 0 and dy == 0):
                    set_neighbours.add(tuple_neighbour)
    
        return list(set_neighbours)
    
    def calculate_new_cell_value(self, current_value, neighbour_values, y, x):
        if current_value == -1:
            return -1
        
        local_sensitivity = self.sensitivity_matrix[y,x]

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
                    raw_p = self.infection_p_stage1*local_sensitivity
                    schaled_p = np.exp(-raw_p)
                    p_no_infection_one_cell.append(schaled_p)
                elif value == 2:
                    raw_p = self.infection_p_stage2*local_sensitivity
                    schaled_p = np.exp(-raw_p)
                    p_no_infection_one_cell.append(schaled_p)
                elif value == 3:
                    raw_p = self.infection_p_stage3*local_sensitivity
                    schaled_p = np.exp(-raw_p)
                    p_no_infection_one_cell.append(schaled_p)
                elif value == 4:
                    raw_p = self.infection_p_stage4*local_sensitivity
                    schaled_p = np.exp(-raw_p)
                    p_no_infection_one_cell.append(schaled_p)
                elif value == 5:
                    raw_p = self.infection_p_stage5*local_sensitivity
                    schaled_p = np.exp(-raw_p)
                    p_no_infection_one_cell.append(schaled_p)
            
            p_no_infection = 1 - (self.p_spontaneous_degeneration * local_sensitivity)
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

        dead_neighbours = 0
        for value in neighbour_values:
            if value == 6:
                dead_neighbours += 1
        
        dead_neighbours_multiplier = 1 + self.dead_neighbour_multiplier * dead_neighbours

        if current_value != 0 and current_value != 6:
            if current_value == 1:
                    p_degeneration = self.degeneration_p_stage1*local_sensitivity * dead_neighbours_multiplier
                    p_degeneration_scaled = 1 - np.exp(-p_degeneration)
            elif current_value == 2:
                    p_degeneration = self.degeneration_p_stage2*local_sensitivity * dead_neighbours_multiplier
                    p_degeneration_scaled = 1 - np.exp(-p_degeneration)
            elif current_value == 3:
                    p_degeneration = self.degeneration_p_stage3*local_sensitivity * dead_neighbours_multiplier
                    p_degeneration_scaled = 1 - np.exp(-p_degeneration)
            elif current_value == 4:
                    p_degeneration = self.degeneration_p_stage4*local_sensitivity * dead_neighbours_multiplier
                    p_degeneration_scaled = 1 - np.exp(-p_degeneration)
            elif current_value == 5:
                    p_degeneration = self.degeneration_p_stage5*local_sensitivity * dead_neighbours_multiplier
                    p_degeneration_scaled = 1 - np.exp(-p_degeneration)

            if np.random.random() < p_degeneration_scaled:
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
        self.time.append(self.t)
        new_config = np.copy(self.config)

        mask = (self.config != -1)
        number_neurons = np.sum(mask)
        number_dead_neurons = np.sum(self.config == 6)
        perc_dead_neurons = round((number_dead_neurons/number_neurons) * 100,2)
        perc_alive_neurons = 100 - perc_dead_neurons

        if perc_alive_neurons <= 70.0 and self.t_70 == None:
            self.t_70 = self.t
        if perc_alive_neurons <= 30.0 and self.t_30 == None:
            self.t_30 = self.t
        if perc_dead_neurons >= 99.0 and self.t_0 == None:
            self.t_0 = self.t
        
        if self.t_70 != None and self.t_30 != None and self.year_per_step == None:
            delta_step = self.t_30 - self.t_70
            self.year_per_step = 10/delta_step
        
        if self.year_per_step is not None:
            self.time_years = [i * self.year_per_step for i in self.time]


        self.neuron_death.append(perc_dead_neurons)



        for x in range(self.width):
            for y in range(self.height):
                cell_value = self.config[y,x]
                list_neighbour_value = []
                for neighbour_x, neighbour_y in self.get_neighbours(y,x):
                    value = self.config[neighbour_y,neighbour_x]
                    list_neighbour_value.append(value)
                new_config[y,x] = self.calculate_new_cell_value(cell_value,list_neighbour_value, y, x)
        
        self.config = new_config

        if perc_dead_neurons >= 99.0:
            return True
        
        return False

if __name__ == '__main__':
    from pyics import GUI
    sim = ParkinsonSim()
    cx = GUI(sim)
    cx.start()


