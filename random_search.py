# data match

from Main import ParkinsonSim
import numpy as np
import pandas as pd



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

def calculate_RMSE(sim_y70, sim_years, sim_percentages, target_data):

    error = []

    for years, percentage in target_data.items():
        index = (np.abs((np.array(sim_years) - sim_y70) - years)).argmin()
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
            'infection_p_stage1':np.random.uniform(0.01, 0.05),
            'infection_p_stage2':np.random.uniform(0.02, 0.1),
            'infection_p_stage3':np.random.uniform(0.03, 0.15),
            'infection_p_stage4':np.random.uniform(0.04, 0.2),
            'infection_p_stage5':np.random.uniform(0.05, 0.25),
            'degeneration_p_stage1':np.random.uniform(0.01, 0.05),
            'degeneration_p_stage2':np.random.uniform(0.02, 0.1),
            'degeneration_p_stage3':np.random.uniform(0.03, 0.15),
            'degeneration_p_stage4':np.random.uniform(0.04, 0.20),
            'degeneration_p_stage5':np.random.uniform(0.05, 0.25),
            'lateral_base_multiplier':np.random.uniform(0.8, 1.5),
            'lateral_ratio_multiplication':np.random.uniform(1, 6),
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
            current_RMSE = calculate_RMSE(sim.t_70*sim.year_per_step,sim.time_years, neuron_alive, target_data)
            sim_RMSE.append(current_RMSE)
        
        mean_RMSE_set = np.mean(sim_RMSE)
        print(mean_RMSE_set)
        results.append({'iteratie': i, 'rmse': mean_RMSE_set,  **params})

        if mean_RMSE_set < best_RMSE:
            best_RMSE = mean_RMSE_set
            best_params = params
        

    
    return best_params, best_RMSE, results

if __name__ == "__main__":
    best_params, best_RMSE, results = random_search(3, 2)

    print(f'best RMSE = {best_RMSE}')
    for name, value in best_params.items():
        print(f'best {name} = {value}')
     
    df = pd.DataFrame(results)

    correlations = df.corr()['rmse'].sort_values()
    print(f'correlations are {correlations}')
    df.to_csv("random_search_results.csv", index=False)
    df = pd.read_csv("random_search_results.csv")