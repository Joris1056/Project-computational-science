from Main import ParkinsonSim
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from intervention_main import ParkinsonSim_intervention



def sim_parkinsons_no_intervention(number_runs, params):
    neurons_alive_total = []
    years_total = []
    year_step_runs = []
    for run in range(number_runs):
        print(run)
        sim = ParkinsonSim(visualize=False)
        sim.reset()
        for name, value in params.items():
                setattr(sim,name,value)

        done = False
        while not done:
            done = sim.step()
            if sim.t >= 2000:
                break
        neuron_alive = [100- p for p in sim.neuron_death]
        year_step_runs.append(sim.year_per_step)

        neurons_alive_total.append(neuron_alive)
        years_total.append(sim.time_years)
    
    # mean year/step for intervention simulations
    mean_year_per_step = np.mean(year_step_runs)

    t_min = 0
    t_max = min(max(y) for y in years_total)
    n_points = 200

    common_time = np.linspace(t_min, t_max, n_points)

    neurons_interp = []
    for years, neurons in zip(years_total,neurons_alive_total):
        f = interp1d(
            years,
            neurons,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        neurons_interp.append(f(common_time))

    neurons_interp = np.array(neurons_interp)

    mean_neurons_alive = np.nanmean(neurons_interp, axis=0)
    std_neurons_alive = np.nanstd(neurons_interp, axis=0)
    CI = [mean_neurons_alive - (1.96 * (std_neurons_alive/np.sqrt(number_runs))), mean_neurons_alive + (1.96 * (std_neurons_alive/np.sqrt(number_runs)))]

    index_70 = np.where(mean_neurons_alive <= 70)[0][0]

    return common_time, mean_neurons_alive, CI, index_70, mean_year_per_step


def sim_parkinsons_intervention(number_runs, params):
    neurons_alive_total = []
    years_total = []
    for run in range(number_runs):
        print(run)
        sim = ParkinsonSim_intervention(visualize=False)
        sim.reset()
        for name, value in params.items():
                setattr(sim,name,value)

        done = False
        while not done:
            done = sim.step()
            if sim.t >= 2000:
                break
        neuron_alive = [100- p for p in sim.neuron_death]

        neurons_alive_total.append(neuron_alive)
        years_total.append(sim.time_years)

    t_min = 0
    t_max = min(max(y) for y in years_total)
    n_points = 200

    common_time = np.linspace(t_min, t_max, n_points)

    neurons_interp = []
    for years, neurons in zip(years_total,neurons_alive_total):
        f = interp1d(
            years,
            neurons,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        neurons_interp.append(f(common_time))

    neurons_interp = np.array(neurons_interp)

    mean_neurons_alive = np.nanmean(neurons_interp, axis=0)
    std_neurons_alive = np.nanstd(neurons_interp, axis=0)
    CI = [mean_neurons_alive - (1.96 * (std_neurons_alive/np.sqrt(number_runs))), mean_neurons_alive + (1.96 * (std_neurons_alive/np.sqrt(number_runs)))]
         
    idx_30_arr = np.where(mean_neurons_alive <= 30)[0]
    if idx_30_arr.size == 0:
        return 0, [0, 0] # Of een andere 'fail' waarde
    
    index_30 = idx_30_arr[0]
    index_70 = np.where(mean_neurons_alive <= 70)[0][0]
    mean_diff = common_time[index_30] - common_time[index_70]

    # Bereken CI per individuele run
    diff_runs = []
    for run in neurons_interp: # Gebruik de geïnterpoleerde data voor gelijke tijdstappen
        i70 = np.where(run <= 70)[0]
        i30 = np.where(run <= 30)[0]
        
        if i70.size > 0 and i30.size > 0:
            diff_runs.append(common_time[i30[0]] - common_time[i70[0]])
    
    if len(diff_runs) > 0:
        std_val = np.std(diff_runs)
        mean_val = np.mean(diff_runs)
        err = 1.96 * (std_val / np.sqrt(len(diff_runs)))
        ci = [mean_val - err, mean_val + err]
    else:
        ci = [0, 0]

    return mean_diff, ci



def plot_neuron_degen_over_time(common_time, mean_neurons_alive, CI, index_70, title, runs):
    plt.plot(common_time-common_time[index_70], mean_neurons_alive, label = 'mean')
    plt.fill_between(np.array(common_time)-common_time[index_70], CI[0], CI[1], alpha = 0.3, color = 'red', label = '95% CI')
    plt.axvline(x = common_time[index_70]-common_time[index_70], color='red', linestyle='--', label='70% neurons alive')
    plt.xlabel('years')
    plt.ylabel('% neurons alive')
    plt.legend()
    plt.title(f'{title}, number of runs: {runs}')
    plt.show() 
    

if __name__ == "__main__":
    runs = 2

    params_no_intervention = {
        'infection_p_stage1': 0.05,
        'infection_p_stage2': 0.10,
        'infection_p_stage3': 0.20,
        'infection_p_stage4': 0.30,
        'infection_p_stage5': 0.40,
        'degeneration_p_stage1': 0.02,
        'degeneration_p_stage2': 0.05,
        'degeneration_p_stage3': 0.10,
        'degeneration_p_stage4': 0.15,
        'degeneration_p_stage5': 0.25,
        'p_spontaneous_degeneration': 0,
        'lateral_base_multiplier': 1,
        'lateral_ratio_multiplication': 0.3,
        'ventral_base_multiplier': 1,
        'ventral_ratio_multiplication': 0.7,
        'dead_neighbour_multiplier': 0.03
    }
    common_time_no_int, mean_neurons_alive_no_int, CI_no_int, index_70_no_int, mean_year_per_step_no_int = sim_parkinsons_no_intervention(runs, params_no_intervention)
    #plot_neuron_degen_over_time(common_time_no_int, mean_neurons_alive_no_int, CI_no_int, index_70_no_int, 'No Intervention', runs)

    difference_years_70_30_list = []
    CI_difference_years = []
    treatment_list = np.linspace(0.1,1, 4)
    for i in range(len(treatment_list)):
        params_intervention = {
            'infection_p_stage1': 0.05,
            'infection_p_stage2': 0.10,
            'infection_p_stage3': 0.20,
            'infection_p_stage4': 0.30,
            'infection_p_stage5': 0.40,
            'degeneration_p_stage1': 0.02,
            'degeneration_p_stage2': 0.05,
            'degeneration_p_stage3': 0.10,
            'degeneration_p_stage4': 0.15,
            'degeneration_p_stage5': 0.25,
            'p_spontaneous_degeneration': 0,
            'lateral_base_multiplier': 1,
            'lateral_ratio_multiplication': 1.3,
            'ventral_base_multiplier': 1,
            'ventral_ratio_multiplication': 1.7,
            'dead_neighbour_multiplier': 0,
            'treatment_alpha_syn': treatment_list[i],
            'year_per_step': mean_year_per_step_no_int
            }
        
        difference_years_70_30, CI_difference = sim_parkinsons_intervention(runs, params_intervention)
        difference_years_70_30_list.append(difference_years_70_30)
        CI_difference_years.append(CI_difference)
    
    CI_lower = [ci[0] for ci in CI_difference_years]
    CI_upper = [ci[1] for ci in CI_difference_years]


    reduction_percentages = (1 - np.array(treatment_list)) * 100


    sort_indices = np.argsort(reduction_percentages)
    x_coords = reduction_percentages[sort_indices]
    y_coords = np.array(difference_years_70_30_list)[sort_indices]
    ci_low = np.array(CI_lower)[sort_indices]
    ci_high = np.array(CI_upper)[sort_indices]
    plt.figure()
    plt.plot(x_coords, y_coords)
    plt.fill_between(x_coords, ci_low, ci_high, alpha = 0.3,color = 'red', label = '95% CI')
    plt.xlabel('strength of intervention')
    plt.ylabel('difference 70% and 30% neurons alive (years)')
    plt.title(f'Effect intervention on years between 70% and 30% neurons alive runs: {runs}')
    plt.show()