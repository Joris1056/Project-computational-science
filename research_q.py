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

    index_70 = np.where(mean_neurons_alive <= 70)[0][0]

    return common_time, mean_neurons_alive, CI, index_70



def plot_neuron_degen_over_time(common_time, mean_neurons_alive, CI, index_70, title):
    plt.plot(common_time-common_time[index_70], mean_neurons_alive, label = 'mean')
    plt.fill_between(np.array(common_time)-common_time[index_70], CI[0], CI[1], alpha = 0.3, label = '95% CI')
    plt.axvline(x = common_time[index_70]-common_time[index_70], color='red', linestyle='--', label='70% neurons alive')
    plt.xlabel('years')
    plt.ylabel('% neurons alive')
    plt.legend()
    plt.title(title)
    plt.show(block = False)
    

if __name__ == "__main__":
    params_no_intervention = {
        'infection_p_stage1': 0.03,
        'infection_p_stage2': 0.06,
        'infection_p_stage3': 0.12,
        'infection_p_stage4': 0.24,
        'infection_p_stage5': 0.48,
        'degeneration_p_stage1': 0.03,
        'degeneration_p_stage2': 0.06,
        'degeneration_p_stage3': 0.12,
        'degeneration_p_stage4': 0.24,
        'degeneration_p_stage5': 0.48,
        'p_spontaneous_degeneration': 0,
        'lateral_base_multiplier': 1,
        'lateral_ratio_multiplication': 3,
        'ventral_base_multiplier': 1,
        'ventral_ratio_multiplication': 6,
        'dead_neighbour_multiplier': 0.03
    }
    common_time_no_int, mean_neurons_alive_no_int, CI_no_int, index_70_no_int, mean_year_per_step_no_int = sim_parkinsons_no_intervention(2, params_no_intervention)
    plt.figure()
    plot_neuron_degen_over_time(common_time_no_int, mean_neurons_alive_no_int, CI_no_int, index_70_no_int, 'No Intervention Simulation')


    params_intervention = {
        'infection_p_stage1': 0.03,
        'infection_p_stage2': 0.06,
        'infection_p_stage3': 0.12,
        'infection_p_stage4': 0.24,
        'infection_p_stage5': 0.48,
        'degeneration_p_stage1': 0.03,
        'degeneration_p_stage2': 0.06,
        'degeneration_p_stage3': 0.12,
        'degeneration_p_stage4': 0.24,
        'degeneration_p_stage5': 0.48,
        'p_spontaneous_degeneration': 0,
        'lateral_base_multiplier': 1,
        'lateral_ratio_multiplication': 3,
        'ventral_base_multiplier': 1,
        'ventral_ratio_multiplication': 6,
        'dead_neighbour_multiplier': 0.03,
        'treatment_alpha_syn': 0.5,
        'year_per_step': mean_year_per_step_no_int
        }
    common_time_int_70, mean_neurons_alive_int_70, CI_int_70, index_70_int_70 = sim_parkinsons_intervention(2, params_intervention)
    plt.figure()
    plot_neuron_degen_over_time(common_time_int_70, mean_neurons_alive_int_70, CI_int_70, index_70_int_70, 'Intervention Simulation (treatment at 70% neurons alive)')

