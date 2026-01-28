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
            if sim.t >= 1500:
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
        print(f"Intervention run {run}")
        sim = ParkinsonSim_intervention(visualize=False)
        sim.reset()
        for name, value in params.items():
            setattr(sim, name, value)

        done = False
        while not done:
            done = sim.step()
            if sim.t >= 1500:  
                break

        neuron_alive = [100 - p for p in sim.neuron_death]
        neurons_alive_total.append(neuron_alive)
        years_total.append(sim.time_years)

    diff_runs = []
    for years, neurons in zip(years_total, neurons_alive_total):
        neurons_arr = np.array(neurons)
        years_arr = np.array(years)

        t_70 = np.interp(70, neurons_arr[::-1], years_arr[::-1])

        if np.any(neurons_arr <= 30):
            t_30 = np.interp(30, neurons_arr[::-1], years_arr[::-1])
        else:
            t_30 = years_arr[-1] 

        diff_runs.append(t_30 - t_70)

    diff_runs = np.array(diff_runs)
    mean_diff_runs = np.mean(diff_runs)
    std_diff_runs = np.std(diff_runs)
    ci = [
        mean_diff_runs - 1.96 * std_diff_runs / np.sqrt(number_runs),
        mean_diff_runs + 1.96 * std_diff_runs / np.sqrt(number_runs)
    ]

    return mean_diff_runs, ci



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
        'dead_neighbour_multiplier': 0,
    }
    common_time_no_int, mean_neurons_alive_no_int, CI_no_int, index_70_no_int, mean_year_per_step_no_int = sim_parkinsons_no_intervention(runs, params_no_intervention)
    plot_neuron_degen_over_time(common_time_no_int, mean_neurons_alive_no_int, CI_no_int, index_70_no_int, 'No Intervention', runs)

    difference_years_70_30_list = []
    CI_difference_years = []
    treatment_list = np.linspace(0.1,1, 3)
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
            'lateral_ratio_multiplication': 0.3,
            'ventral_base_multiplier': 1,
            'ventral_ratio_multiplication': 0.7,
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
