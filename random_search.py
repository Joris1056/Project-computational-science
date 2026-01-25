# data match

from Main import ParkinsonSim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



target_data = {
    5:50,
    10:30,
    15:20,
    20:15,
    25:10,
    30:7.5,
    35:5,
    40:1,
}

def calculate_RMSE(shifted_years, sim_percentages, target_data):

    error = []

    for years, percentage in target_data.items():
        index = (np.abs(np.array(shifted_years) - years)).argmin()
        sim_value = sim_percentages[index]

        squared_error = (sim_value - percentage)**2
        error.append(squared_error)

    return np.sqrt(np.mean(error))

def random_search(iterations, n_runs_per_set):
    best_RMSE = float('inf')
    best_params = {}
    results = []

    for i in range(iterations):
        print(i)
        
        params = {
            'infection_p_stage1':np.random.uniform(0.005, 0.02),
            'infection_p_stage2':np.random.uniform(0.05, 0.12),
            'infection_p_stage3':np.random.uniform(0.03, 0.15),
            'infection_p_stage4':np.random.uniform(0.04, 0.2),
            'infection_p_stage5':np.random.uniform(0.05, 0.25),
            'degeneration_p_stage1':np.random.uniform(0.01, 0.05),
            'degeneration_p_stage2':np.random.uniform(0.2, 0.3),
            'degeneration_p_stage3':np.random.uniform(0.03, 0.15),
            'degeneration_p_stage4':np.random.uniform(0.01, 0.05),
            'degeneration_p_stage5':np.random.uniform(0.35, 0.50),
            'lateral_base_multiplier':np.random.uniform(0.2, 0.6),
            'lateral_ratio_multiplication':np.random.uniform(4, 6),
            'ventral_base_multiplier':np.random.uniform(0.8, 1.5),
            'ventral_ratio_multiplication':np.random.uniform(1, 6),
            'dead_neighbour_multiplier':np.random.uniform(0.05, 0.1),
        }
        
        sim_RMSE = []
        for run in range(n_runs_per_set):
            sim = ParkinsonSim(visualize=False)
            sim.reset()
            for name, value in params.items():
                setattr(sim,name,value)

            done = False
            while not done:
                done = sim.step()
                if sim.t > 2000:
                    break
            
            neuron_alive = 100 - np.array(sim.neuron_death)
            shifted_years = np.array(sim.time_years) - (sim.t_70 * sim.year_per_step)
            current_RMSE = calculate_RMSE(shifted_years, neuron_alive, target_data)
            sim_RMSE.append(current_RMSE)
        
        mean_RMSE_set = np.mean(sim_RMSE)
        print(mean_RMSE_set)
        results.append({'iteratie': i, 'rmse': mean_RMSE_set,  **params})

        if mean_RMSE_set < best_RMSE:
            best_RMSE = mean_RMSE_set
            best_params = params
            best_time = sim.time_years
            best_neuron_alive = neuron_alive
            best_t_70 = sim.t_70*sim.year_per_step

        

    
    return best_params, best_RMSE, results, best_time, best_neuron_alive, best_t_70

if __name__ == "__main__":
    best_params, best_RMSE, results, best_time, best_neuron_alive, best_t_70= random_search(3, 3)

    print(f'best RMSE = {best_RMSE}')
    for name, value in best_params.items():
        print(f'best {name} = {round(value,2)}')
     
    df = pd.DataFrame(results)

    correlations = df.corr()['rmse'].sort_values()
    print(f'correlations are {correlations}')
    df.to_csv("random_search_results.csv", index=False)
    df = pd.read_csv("random_search_results.csv")

    plt.plot(np.array(best_time) - best_t_70, best_neuron_alive, label = 'best simulation')
    plt.show()