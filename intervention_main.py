# Intervention main

# Here we run the same simulation as Main.py but now we have also added the parameter 
# treatment_alpha_syn, the lower this value (between 0.0 and 1.0) the stronger the treatment.

# fort the imports
import numpy as np
from pyics import Model

# The class is overall the same as the class in Main.py
# Changes: added parameter treatment_alpha_syn
#          added parameter year_per_step

#
class ParkinsonSim_intervention(Model):
    def __init__(self, visualize = True):
        Model.__init__(self)
        self.visualize = visualize

        self.t = 0
        self.config = None
        self.sensitivity_matrix = None
        self.neuron_death = []
        self.time = []
        self.time_years =  []
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
        self.make_param('infection_p_stage1', 0.05)
        self.make_param('infection_p_stage2', 0.10)
        self.make_param('infection_p_stage3', 0.20)
        self.make_param('infection_p_stage4', 0.30)
        self.make_param('infection_p_stage5', 0.40)

        #the probability of degeneration per stage
        self.make_param('degeneration_p_stage1', 0.02)
        self.make_param('degeneration_p_stage2', 0.05)
        self.make_param('degeneration_p_stage3', 0.10)
        self.make_param('degeneration_p_stage4', 0.15)
        self.make_param('degeneration_p_stage5', 0.25)
        #maybe if there is spontaneous degeneration then we have p spon deg.
        self.make_param('p_spontaneous_degeneration', 0)
        self.make_param('lateral_base_multiplier', 1)
        self.make_param('lateral_ratio_multiplication', 0.3)
        self.make_param('ventral_base_multiplier', 1)
        self.make_param('ventral_ratio_multiplication', 0.7)
        self.make_param('dead_neighbour_multiplier', 0)

        # added parameters for treatment:
        
        # first of all lets add a parameter that give the treatment strength value.
        # this will be a value that is between 0.0 and 1.0. The lower the value the
        # stronger the treatment will be (0.30 is 70% decreased probability of degeneration of the group neurons).
        self.make_param('treatment_alpha_syn', 0.3)

        # In the Main.py code we calculate ta year_per_step. this lines up the 30% loss
        # and the 70% loss of neurons in the substantia nigra to our found literature.
        # In this intervention codewe compare it to a non intervention simulation. 
        # So we use the year_per_step and add it here.
        self.make_param('year_per_step',0.07194244604316546)
        

    def reset(self):
        """Initializes or resets the simulation state."""
        if self.visualize:
            import matplotlib.pyplot as plt
            self.fig_neuron_alive, self.ax_neuron_alive = plt.subplots()
            self.line_neuron_alive, = self.ax_neuron_alive.plot([], [])
            self.ax_neuron_alive.set_ylabel('% live Neurons')
            self.ax_neuron_alive.set_title('Neurons alive Over Time')
            self.ax_neuron_alive.legend()

        # initialise values to be tracked in simulation
        self.t = 0
        self.time = []
        self.neuron_death = []
        self.time_years = []
        self.t_70 = None
        self.t_30 = None
        self.t_0 = None
        self.t_70_years = None
        self.t_30_years = None  
        self.t_0_years = None   
        

        # create the substantia nigra shape in the CA grid
        self.config = np.full((self.height, self.width), -1.0)
        
        center_x = self.width * 0.35
        for y in range(self.height):
            for x in range(self.width):
                nx = x / self.width
                curve = (self.height * 0.2) + ((x - center_x)**2 / (self.width * 0.8)) - (x * 0.15)
                min_width = self.height * 0.1
                max_width = self.height * 0.25
                width = min_width + (nx * (max_width - min_width))
                
                if y > curve and y < curve + width:
                    self.config[y, x] = 0

        # find the bounds of the SN grid to be used in the sensitivity matrix
        self.sn_bounds = {}

        for x in range(self.width):
            ys = np.where(self.config[:,x] != -1)[0]
            if len(ys) > 0:
                y_min = ys.min()
                y_max = ys.max()
                self.sn_bounds[x] = (y_min,y_max)

        # create sensitivity matrix
        self.sensitivity_matrix = np.full((self.height, self.width), 0)
        
        
        for y in range(self.height):
            for x in range(self.width):
                if self.config[y,x] != -1:
                    ratio_x = x/self.width
                    y_min,y_max = self.sn_bounds[x]
                    relative_y = (y-y_min)/(y_max-y_min)

                    # the formula for the x and y multipliers is multiplier = base + (ratio * multiplication)
                    # the x and y multipliers are then multiplied together and squared to get the sensitivity value
                    # the multiplication and squaring makes the sensitivity more extreme towards the ventral-lateral area 
                    # it also keeps the multipliers more reasonable given the probabilities
                    x_multiplier = self.lateral_base_multiplier + (ratio_x * self.lateral_ratio_multiplication)
                    y_multiplier = self.ventral_base_multiplier + ((1 - relative_y) * self.ventral_ratio_multiplication)
                    self.sensitivity_matrix[y,x] = (x_multiplier * y_multiplier)**2

        # infect one cell and Moore neighbourhood on the lateral-verntral side to start the infection
        found = False
        for x in range(self.width - 1, 0, -1):
            for y in range(self.height):
                if self.config[y, x] == 0:
                    self.config[y, x] = 1
                    neighbours = self.get_neighbours(y,x)
                    for x,y in neighbours:
                        self.config[y,x] = 1
                    found = True
                    break
            if found:
                break 
    



    def draw(self):
        """Handles the visualization of the grid."""
        if not self.visualize:
            return 
        import matplotlib
        import matplotlib.pyplot as plt

        # calculate percentage of alive and dead neurons
        mask = (self.config != -1)
        number_neurons = np.sum(mask)

        number_dead_neurons = np.sum(self.config == 6)

        total_neuron = 5.5*10**5
        neuron_per_cel = total_neuron/number_neurons
        perc_dead_neurons = round((number_dead_neurons/number_neurons) * 100,2)
        neuron_representation = f'One cell = {int(neuron_per_cel)} dopaminergic neurons'
        percentage_dead_neurons = f'{perc_dead_neurons}% neurons dead'   

        # generate interface of the simulation
        plt.cla()
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

        
        # code that changes the live plot of neurons alive over time based on simulation progression
        if self.t_70 is not None:
            self.line_neuron_alive.set_data(np.array(self.time_years) - self.t_70_years, 100 - np.array(self.neuron_death))
        else:
            self.line_neuron_alive.set_data(self.time_years, 100 - np.array(self.neuron_death))
        
        for line in self.ax_neuron_alive.lines[1:]:
            line.remove()
        
        if self.t_70_years != None:
            self.ax_neuron_alive.axvline(x = self.t_70_years-self.t_70_years, color = 'black', linestyle = '--', label = f'{round(self.t_70_years-self.t_70_years,2)}th year, 70% left')

        if self.t_30_years != None:
            self.ax_neuron_alive.axvline(x = self.t_30_years-self.t_70_years, color = 'green', linestyle = '--',label = f'{round(self.t_30_years-self.t_70_years,2)}th year, 30% left)')
        
        if self.t_0_years != None:
            self.ax_neuron_alive.axvline(x = self.t_0_years-self.t_70_years, linestyle = '--', color = 'black', label = f'{round(self.t_0_years-self.t_70_years,1)}th year, 1% left')
        
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

        #Now lets add the intervention step at t_70
        if self.t_70 != None:
            treatment = self.treatment_alpha_syn
        else:
            treatment = 1
            

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

            # raw prob of no infection = infection prob per stage * local sensitivity
            # schaled prob of no infection = exp(- raw prob of infection)
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
            # total prob of no infection is the product of all no infection probabilities from each sick neighbour
            for i in range(0,len(p_no_infection_one_cell)):
                p_no_infection *= p_no_infection_one_cell[i]

            # chance of healthy cell getting infected = 1 - chance of no infection
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

        # dead neighbour multiplier calculation --> more dead neighbours increases degeneration prob
        dead_neighbours_multiplier = 1 + self.dead_neighbour_multiplier * dead_neighbours

        # raw prob degeneration = degeneration prob per stage * local sensitivity * dead neighbour multiplier
        # scaled prob degeneration = 1 - exp(- raw prob degeneration)
        if current_value != 0 and current_value != 6:
            if current_value == 1:
                    p_degeneration = self.degeneration_p_stage1*local_sensitivity * dead_neighbours_multiplier
                    p_degeneration_scaled = (1 - np.exp(-p_degeneration)) * treatment
            elif current_value == 2:
                    p_degeneration = self.degeneration_p_stage2*local_sensitivity * dead_neighbours_multiplier
                    p_degeneration_scaled = (1 - np.exp(-p_degeneration)) * treatment
            elif current_value == 3:
                    p_degeneration = self.degeneration_p_stage3*local_sensitivity * dead_neighbours_multiplier 
                    p_degeneration_scaled = (1 - np.exp(-p_degeneration)) * treatment
            elif current_value == 4:
                    p_degeneration = self.degeneration_p_stage4*local_sensitivity * dead_neighbours_multiplier 
                    p_degeneration_scaled = (1 - np.exp(-p_degeneration)) * treatment
            elif current_value == 5:
                    p_degeneration = self.degeneration_p_stage5*local_sensitivity * dead_neighbours_multiplier 
                    p_degeneration_scaled = (1 - np.exp(-p_degeneration)) * treatment

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

        if self.year_per_step is not None:
            self.time_years.append(self.t * self.year_per_step)
        new_config = np.copy(self.config)


        mask = (self.config != -1)
        number_neurons = np.sum(mask)
        number_dead_neurons = np.sum(self.config == 6)
        perc_dead_neurons = round((number_dead_neurons/number_neurons) * 100,2)
        perc_alive_neurons = 100 - perc_dead_neurons

        if perc_alive_neurons <= 70.0 and self.t_70 == None:
            self.t_70 = self.t
            self.t_70_years = self.t * self.year_per_step
        if perc_alive_neurons <= 30.0 and self.t_30 == None:
            self.t_30 = self.t
            self.t_30_years = self.t * self.year_per_step
        if perc_dead_neurons >= 99.0 and self.t_0 == None:
            self.t_0 = self.t
            self.t_0_years = self.t * self.year_per_step


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

        # stop simulation when 99% of neurons are dead
        if perc_dead_neurons >= 99.0:
            return True
        
        return False

if __name__ == '__main__':
    from pyics import GUI
    sim = ParkinsonSim_intervention()
    cx = GUI(sim)
    cx.start()

