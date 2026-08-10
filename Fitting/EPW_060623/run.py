import os
import sys
import pymc as pm
import numpy as np
import glob
from multiprocessing import Process, Pool, Queue
## Add the Libraries folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Libraries'))
from normalisation import Data_normalisation
from OTS_function import initialize, OTS, calculations
from mcmc import Fit_OTS, LogLike_nograd
import matplotlib.pyplot as plt
import arviz as az
import cma
import pandas as pd
import time
import seaborn as sns
from scipy.stats import gaussian_kde

class Initialization:

    def __init__(self):
        self.model_options()
        self.scattering_params()
        self.material_params()
        self.OTS_init = self.input()
        self.OTS_model = OTS(self.OTS_init)

    def model_options(self):
        self.use_zbar_eos = True # Whether to use the zbar from the EOS file for the fitting process
        self.use_range = False # Whether to include the range of scattering vectors in the fitting process
        self.fit_type = 'bohm-gross' # or 'salpeter'

    def scattering_params(self):
        # Wavelength in nm, Angle in degrees

        self.wavelength_min = 399  # nm
        self.wavelength_max = 480  # nm
        self.wavelength_step = 0.1  # nm

        self.wavelength_fwhm = 3.208  # nm
        self.wavelength = 1053/2  # nm
        self.theta = 59.9  # degrees
        self.wavelengths = np.arange(self.wavelength_min, self.wavelength_max, self.wavelength_step)  # nm

    def material_params(self):
        # Material parameters
        self.elements = ['C', 'H', 'Cl']
        self.fractions = [0.499, 0.438, 0.063]  # Relative fractions of each element
        self.zs = [6, 1, 17]  # Ionisation states
        self.eos_file = '/Users/hpoole/Documents/Simulations/PROPACEOS/C49_9H43_8Cl6_3/grid_data.npz'

        ## Fixed parameters
        self.e_cur = 0 # km/s
        self.flow = 0 # km/s

    def input(self):
        OTS_init = initialize()
        OTS_init.model_options(self.use_zbar_eos, self.use_range, self.fit_type)
        OTS_init.scattering_params(self.wavelength, self.theta, self.wavelength_fwhm, self.wavelength_min, self.wavelength_max, self.wavelength_step)
        OTS_init.material_params(self.elements, self.fractions, eos_file=self.eos_file)
        OTS_init.plasma_params(ionizations=self.zs, e_cur=self.e_cur, flow=self.flow)
        return OTS_init

class Exploration:

    def __init__(self, use_IAW_data=False, parent_loc=None, shot_time=None, shot_num=None):
        ## SETTING THE PARAMETERS TO EXPLORE

        ## params -> User friendly naming of parameters
        ## units -> Unit of each parameter
        ## inits -> The initial guess for fitting parameter
        ## mins -> Lower bound on the parameter exploration
        ## maxs -> Upper bound on the parameter exploration
        ## logbool -> Bool function whether to explore the parameter in log space
        self.param_names = ['TE', 'TI', 'NE', 'NE_RANGE']
        self.param_units = ['eV', 'eV', '1/cc', '1/cc']
        self.param_logbool = [False, False, True, True]

        if use_IAW_data:
            self.get_IAW_ranges(shot_time, shot_num, parent_loc)
        else:
            self.param_inits = [266, 74, 1.2e20, 2e19]
            self.param_mins = [25, 5, 4e19, 1e18]
            self.param_maxs = [550, 550, 6e20, 8e19]

        ## Normalise the parameters, as MCMC explores parameters between 0 and 1
        self.norm_param_inits = [Data_normalisation().normalise_data(self.param_inits[i], self.param_mins[i], self.param_maxs[i], log=self.param_logbool[i]) for i in range(len(self.param_inits))]

        ## Sigma utilised in cost function of MCMC fitting
        self.sigma = 4

        df = {
            'names': self.param_names,
            'units': self.param_units,
            'inits': self.param_inits,
            'mins': self.param_mins,
            'maxs': self.param_maxs,
            'logbool': self.param_logbool,
            'sigma': self.sigma
        }
        pd.DataFrame(df).to_csv('./MCMC_fitting_info.csv', index=False)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # Properly call the parent class
        except AttributeError:
            # Handle missing attributes gracefully
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # return getattr(self, name)

    def get_IAW_ranges(self, shot_time, shot_num, parent_loc):
        IAW_loc = parent_loc.replace('EPW', 'IAW')
        IAW_file = os.path.join(IAW_loc, 'Results', 'MCMC', f'MCMC_fitting_info_s{shot_num}.csv')
        IAW_info = pd.read_csv(IAW_file, index_col=None)
        IAW_times = IAW_info['Time (ns)'].values * 1e3  # Convert to ps
        IAW_Mean_Te = IAW_info['Mean Te (eV)'].values
        IAW_Std_Te = IAW_info['Std Te (eV)'].values
        IAW_Mean_Ti = IAW_info['Mean Ti (eV)'].values
        IAW_Std_Ti = IAW_info['Std Ti (eV)'].values

        Splits = np.array([(IAW_times[p + 1] + IAW_times[p]) / 2 for p in range(0, len(IAW_times) - 1, 1)])
        Time_split = np.insert(Splits, 0, 0)
        Time_split = np.append(Time_split, 100000)

        for t in range(0, len(Time_split) - 1, 1):
            Max_time = Time_split[t + 1]
            Min_time = Time_split[t]
            tepw = int(shot_time)
            if tepw <= Max_time and tepw >= Min_time:
                TI = IAW_Mean_Ti[t]
                TI_STD = IAW_Std_Ti[t]
                TE = IAW_Mean_Te[t]
                TE_STD = IAW_Std_Te[t]

        self.param_inits = [TE, TI, 1.4e20, 1.7e19]
        self.param_mins = [TE - 2 * TE_STD, TI - 2 * TI_STD, 4e19, 1e18]
        self.param_maxs = [TE + 2 * TE_STD, TI + 2 * TI_STD, 6e20, 1e20]

class Fitting:

    def __init__(self):
        ## CLASS WHERE THE FITTING ALGORITHMS ARE DEFINED

        ## Defines the raw data to fit to
        ## Perform an initial test fit to compare to data
        ## Run optimisation (CMA-ES) fit to obtain best fit to data
        ## Run MCMC model to fit to data
        return

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # Properly call the parent class
        except AttributeError:
            # Handle missing attributes gracefully
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # return getattr(self, name)

    def get_model(self):
        return self.OTS_model

    def setup(self, shot_day, shot_number, shot_time, raw_file, exploration=None):
        self.shot_day = shot_day
        self.shot_number = shot_number
        self.shot_time = shot_time
        self.raw_file = raw_file
        self.experimental_data()
        if exploration is not None:
            self.EXPLORE = exploration
        else:
            self.EXPLORE = Exploration()
        self.MCMC_INIT = Initialization()
        self.OTS_model = self.MCMC_INIT.OTS_model

    def experimental_data(self):
        ## Defining the raw data
        Raw_data = np.genfromtxt(self.raw_file)
        self.raw_lambda, self.raw_I, Raw_per_err = Raw_data[:, 0], Raw_data[:, 1], Raw_data[:, 2]
        self.raw_err = self.raw_I * Raw_per_err

    def initial_test(self):
        ## Run an initial test of an example fit
        ## Can help debug any issues in the system
        ## NB: Doesn't perform any fitting

        t0 = time.time()
        scaled_f = self.generate_fit(self.EXPLORE.param_inits)
        t1 = time.time()
        print('Initial test fit took {:.2f} seconds'.format(t1-t0))
        cost = -Fit_OTS().likelihood(self.EXPLORE.sigma, scaled_f, self.raw_I, self.raw_err)
        t2 = time.time()
        print('Cost calculation took {:.2f} seconds'.format(t2-t1))

        fig, axs = plt.subplots()
        axs.errorbar(self.raw_lambda, self.raw_I, yerr=np.abs(self.raw_err), color='gray',
                     capsize=4, barsabove=True, linestyle='None', marker='D', markersize=2,
                     alpha=0.5)
        axs.plot(self.raw_lambda, self.raw_I, 'k-', alpha=0.7, label='Raw data')
        axs.plot(self.raw_lambda, scaled_f, 'r-', label='Starting fit')
        axs.plot(self.raw_lambda, cost / np.nanmax(cost), 'b-', alpha=0.5, label='Cost')
        # axs.plot(Info[0], Info[1], 'g--', alpha=1, label='Code')
        plt.legend(loc='best')
        plt.suptitle('Initial test\nCost={:.5g}'.format(np.sum(cost)))
        plt.show()

    def generate_fit(self, params):  
        # Unnormalized parameters to generate a fit
        model_x, model_y = self.OTS_model.run_fitting(params, self.EXPLORE.param_names)
        model = np.interp(self.raw_lambda, model_x * 1e9, model_y)  # Interpolate model_y to match x
        scaled_model = Fit_OTS().scalings(self.EXPLORE.sigma, self.raw_I, self.raw_err, model)
        return scaled_model
    
    def unnormalize(self, params):
        orig_params = []
        for i in range(0, len(params), 1):
            orig_params.append(Data_normalisation().unnormalise_data(params[i], self.EXPLORE.param_mins[i], self.EXPLORE.param_maxs[i], log=self.EXPLORE.param_logbool[i]))
        return np.asarray(orig_params)
    
    def normalize(self, params):
        norm_params = []
        for i in range(0, len(params), 1):
            norm_params.append(Data_normalisation().normalise_data(params[i], self.EXPLORE.param_mins[i], self.EXPLORE.param_maxs[i], log=self.EXPLORE.param_logbool[i]))
        return np.asarray(norm_params)

    def run_optimisation_model(self):
        ## Perform optimisation model fitting (CMA-ES)
        ## Produces an initial best fit to the data based on the defined cost function

        def function(params):
            ## Define the fitting function
            orig_params = self.unnormalize(params)
            scaled_model = self.generate_fit(orig_params)
            costs = -Fit_OTS().likelihood(self.EXPLORE.sigma*0.5, scaled_model, self.raw_I, self.raw_err)
            return np.sum(costs)  # Return chi-squared error or another error metric

        ## CMAES options with bounds
        ## By definition the bounds for each parameter are 0 and 1
        lower_bounds, upper_bounds = np.zeros(len(self.EXPLORE.norm_param_inits)), np.ones(len(self.EXPLORE.norm_param_inits))
        options = {
            'bounds': [lower_bounds, upper_bounds],  # Set the bounds
            'maxiter': 10000,  # Maximum number of iterations
        }
    
        es = cma.CMAEvolutionStrategy(x0=self.EXPLORE.norm_param_inits, sigma0=0.1, options=options)
        opt_result = es.optimize(function)

        # # Output the results
        best_params = opt_result.result.xbest  # Best parameter set found
        unnormalize_best_params = self.unnormalize(best_params)

        ## Compare the fit of the initial parameters to the optimised parameters
        Initial_y = self.generate_fit(self.EXPLORE.param_inits)
        Optimized_y = self.generate_fit(unnormalize_best_params)
        fig, axs = plt.subplots()
        axs.errorbar(self.raw_lambda, self.raw_I, yerr=np.abs(self.raw_err), color='gray',
                     capsize=4, barsabove=True, linestyle='None', marker='D', markersize=2,
                     alpha=0.5)
        axs.plot(self.raw_lambda, self.raw_I, 'k-', alpha=0.7, label='Raw data')
        axs.plot(self.raw_lambda, Initial_y, 'g-', label='Starting fit')
        axs.plot(self.raw_lambda, Optimized_y, 'r-', label='Optimised fit')
        plt.legend(loc='best')
        plt.show()

        ## Print the optimised parameters found
        df = np.array([self.EXPLORE.param_inits, unnormalize_best_params]).T
        dataframe = pd.DataFrame(df, index=self.EXPLORE.param_names, columns=['Initial', 'Optimised'])
        print('\nOptimal parameters')
        print(dataframe)

        return unnormalize_best_params, best_params

    def run_mcmc_model(self, norm_param_inits=None):
        ## Perform MCMC fitting function
        ## Runs MCMC fitting for one core

        ## Define the starting values for MCMC fitting process
        if norm_param_inits is None:
            norm_param_inits = self.EXPLORE.norm_param_inits
        norm_param_inits = np.round(norm_param_inits, decimals=5)

        with pm.Model():
            running_params = [pm.Uniform(f"{self.EXPLORE.param_names[i]}", lower=0, upper=1, initval=norm_param_inits[i]) for i in range(len(norm_param_inits))]

            ## Define the likelihood function used to assess appropriateness of each fit
            likelihood = pm.CustomDist(
                "Likelihood",
                self.raw_lambda,
                self.raw_err,
                *running_params,  # Unpack the list so each param is a separate argument
                observed=self.raw_I,
                logp=custom_dist_loglike
            )

            ## Inject a random seed into each chain
            rng = np.random.default_rng(666)

            ## Define the sampling step function
            step = pm.Metropolis()
            # step = pm.NUTS(step_scale=0.05)
            # step = pm.HamiltonianMC(step_scale=0.05)

            ## Run the MCMC sampling with certain number of iterations
            ## The number of chains run on each core can only be 1
            # self.trace = pm.sample(draws=25000, tune=5000, chains=12, step=step, cores=12, progressbar=True, random_seed=rng)
            self.trace = pm.sample(draws=100, tune=50, chains=12, step=step, cores=12, progressbar=True, random_seed=rng)

        return self.trace

    def process_idata(self, idata=None, percentages=[0, 100], save=False, plot=True):
        ## Process the idata output from MCMC fitting
        if idata is None:
            idata = self.trace

        posterior = idata.posterior
        n_params = len(Names)
        n_chains = posterior.sizes['chain']
        n_draws = posterior.sizes['draw']

        data_fraction = [int(d*0.01*n_draws) for d in percentages]

        ## Plot idata with arviz
        # az.plot_trace(idata, Names)
        # plt.show()

        ncols = int(n_params)
        nrows = 2
        if plot:
            fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        all_params = np.zeros((n_params, n_chains, int(data_fraction[1]-data_fraction[0])))
        for c in range(n_chains):
            ## Assign a unique color for each chain using a colormap
            cmap = plt.get_cmap('jet')  # pick any cmap you prefer, e.g. 'plasma', 'tab10', etc.
            if n_chains > 1:
                colors = [cmap(i / (n_chains-1)) for i in range(n_chains)]
            else:
                colors = [cmap(0.5)]
            chain_color = colors[c]
            
            ## Or use a single color for all chains
            # chain_color = 'C0'

            samples_list = [posterior[p].values[c][data_fraction[0]:data_fraction[1]].reshape(-1) for p in Names]
            values = np.vstack(samples_list).T
            unnorm_values = np.array([Run.unnormalize(values[i]) for i in range(len(values))])
            all_params[:, c, :] = unnorm_values.T

            samples = {Names[p]: unnorm_values[:, p] for p in range(n_params)}

            if plot:
                for i, p in enumerate(Names):
                    ax0, ax1 = axs[0, i], axs[1, i]
                    ## Plot a density line (KDE)
                    ## Can use az.plot_kde, but I prefer control with gaussian_kde (the outputs should be similar)
                    try:
                        data = np.asarray(samples[p])
                        kde = gaussian_kde(data, bw_method=0.1)
                        x_kde = np.linspace(data.min(), data.max(), 300)
                        y_kde = kde.evaluate(x_kde)
                        ax0.plot(x_kde, y_kde, color=chain_color, linewidth=2, alpha=0.9)
                        ax0.fill_between(x_kde, y_kde, color=chain_color, alpha=0.1)
                    except Exception:
                        az.plot_kde(samples[p], ax=ax0, bw='experimental')

                    ax1.plot(samples[p], np.arange(len(samples[p])), '-', color=chain_color, alpha=0.7)
        collapsed_params = np.reshape(all_params, (n_params, n_chains*int(data_fraction[1]-data_fraction[0])))
        samples = {Names[p]: collapsed_params[p, :] for p in range(n_params)}
        stats = {p: {
                    'median': np.median(samples[p]),
                    'p2.5': np.percentile(samples[p], 2.5),
                    'p97.5': np.percentile(samples[p], 97.5)
                    } for p in Names}
        if plot:
            for i, p in enumerate(Names):
                axs[1, i].set_xlabel(f'{p} [{Units[i]}]')
                axs[0, i].set_ylabel('Frequency')
                axs[1, i].set_ylabel('Sample')
                axs[0, i].set_title(p)
                for ax in axs[:, i].flat:
                    ax.axvline(Best_inits[i], color='forestgreen', linestyle=':', label='init' if i == 0 else None)
                    ax.axvline(stats[p]['median'], color='maroon', label='median')
                    ax.axvline(stats[p]['p2.5'], color='k', linestyle='--', label='95% CI' if i == 0 else None)
                    ax.axvline(stats[p]['p97.5'], color='k', linestyle='--')
                    if Logbool[i]:
                        ax.set_xscale('log')
                    ax.tick_params('both', which='major', direction='in', length=5)
                if i == 0:
                    axs[0, i].legend()
            fig.tight_layout()
            plt.suptitle(f'{Shot_time}ps MCMC posterior for {percentages[0]}% - {percentages[1]}%', y=1.05, fontsize=20)
            if save:
                plt.savefig(os.path.join(MCMC_loc, 'idata.png'))
            plt.show()
        if save:
            idata.to_netcdf(os.path.join(MCMC_loc, 'idata.nc'))
        return samples

def custom_dist_loglike(data, x, data_err, *params):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(params, x, data_err, data)

def save_cmaes_file(save_loc, names, params, units):
    save_file = os.path.join(save_loc, 'Plasma_parameters.txt')
    with open(save_file, 'w') as d:
        for i in range(len(names)):
            d.writelines(
                f'{names[i]}\t\t{params[i]} {units[i]}\n'
            )
        d.close()
    return

def extract_cmaes_file(save_loc, names):
    save_file = os.path.join(save_loc, 'Plasma_parameters.txt')
    params = []
    with open(save_file, 'r') as d:
        lines = d.readlines()
        for i in range(len(names)):
            for j in range(len(lines)):
                if names[i] in lines[j]:
                    params.append(float(lines[j].split()[1]))
    return np.asarray(params)

loglike_op = LogLike_nograd()

if __name__ == '__main__':
    ## Import data
    ###############################################################
    ##                                                           ##
    ##                        DATA TO INPUT                      ##
    ##                                                           ##
    ###############################################################

    ## Define the shot day, shot number, and diagnostic
    Shot_day = 'OMEGA_Jun2023'
    Shot_number = 108616

    ## If using TDYNO_NLUF Box account, User as required in Parent_loc
    User = 'hpoole'

    ## Bools for what you want to run and if you want to save outputs
    Save_info = False
    Use_IAW_data = True
    Run_CMAES = True
    Only_CMAES = False
    Run_MCMC = True
    Show_fits = True
    Run_fits = True

    ## There are a few different options for running the EPW fitting.
    ## The main options are:
    ## 1. Whether you want to use Bohm-Gross or Salpeter for the fitting model
    ##      Bohm-Gross is a lot faster and less susceptile to resolution issues, but Salpeter is more accurate
    ##      If using Bohm-Gross, it is recommended to remove Ti from the fitting process, as there is no dependency on Ti in the Bohm-Gross model
    ## 2. Whether you want to use the IAW fitting results to inform the EPW fitting ranges (Use_IAW_data)
    ##    This is recommended if the breadth of the EPW signal is a result of a range of densities (ie you are trying to fit an ne_range)
    ## 3. Whether you want to include the range of scattering vectors
    ##    With EPW, I typically don't see much influence of the range of scattering vectors, so I typically don't include it in the fitting process as it slows it down
    ## At the minute the catch for the model options are in Initialization.model_options

    ###############################################################

    ###############################################################
    ##                                                           ##
    ##                      FILE LOCATIONS                       ##
    ##                                                           ##
    ###############################################################
    # Global_loc = os.path.join('/', 'Users', User, 'Library', 'CloudStorage', 'Box-Box', 'TDYNO_NLUF', 'OMEGA', Shot_day, 'Data')
    Global_loc = os.path.join('/', 'Users', User, 'Documents', 'TDYNO', 'OMEGA', Shot_day, 'Data')
    Parent_loc = os.path.join(Global_loc, str(Shot_number), 'EPW')

    Raw_files_loc = os.path.join(Parent_loc, 'Scattering_strips')
    Raw_files = sorted(glob.glob(os.path.join(Raw_files_loc, '*ps.txt')))
    Raw_times = [Raw_file.replace('{}/'.format(Raw_files_loc), '').replace('ps.txt', '') for Raw_file in Raw_files]

    for t in [10]:#range(10, len(Raw_times), 1):
        Shot_time = Raw_times[t]
        Raw_file = os.path.join(Raw_files_loc, Raw_files[t])
        print('\nRunning time {} ns ....'.format(int(Shot_time)*1e-3))
        Save_loc = Raw_files_loc.replace('Scattering_strips', 'Results')

        CMAES_loc = os.path.join(Save_loc, 'CMAES', f'{Shot_time}ps')
        MCMC_loc = os.path.join(Save_loc, 'MCMC', f'{Shot_time}ps')
        Save_fits_loc = os.path.join(MCMC_loc, 'Fits')

        if Save_info:
            if not os.path.exists(Save_fits_loc):
                os.makedirs(Save_fits_loc)
            if not os.path.exists(CMAES_loc):
                os.makedirs(CMAES_loc)

        EXPLORE = Exploration(use_IAW_data=Use_IAW_data, parent_loc=Parent_loc, shot_time=Shot_time, shot_num=Shot_number)

        Names, Units, Mins, Maxs, Logbool = EXPLORE.param_names, EXPLORE.param_units, EXPLORE.param_mins, EXPLORE.param_maxs, EXPLORE.param_logbool

        Run = Fitting()
        Run.setup(Shot_day, Shot_number, Shot_time, Raw_file, exploration=EXPLORE)
        
        print('... Running initial test fit')
        Run.initial_test()
   
        if Run_CMAES:
            print('... Running CMAES')
            Best_inits, Best_norm_inits = Run.run_optimisation_model()

            fig, axs = plt.subplots()
            axs.errorbar(Run.raw_lambda, Run.raw_I, yerr=np.abs(Run.raw_err), color='gray',
                         capsize=4, barsabove=True, linestyle='None', marker='D', markersize=2,
                         alpha=0.5)
            axs.plot(Run.raw_lambda, Run.raw_I, 'k-', alpha=0.7, label='Raw data')

            scaled_f = Run.generate_fit(Best_inits)
            axs.plot(Run.raw_lambda, scaled_f, 'r-', label='CMAES fit')
            axs.minorticks_on()
            axs.tick_params('both', which='minor', direction='in', length=4)
            axs.tick_params('both', which='major', direction='in', length=9)
            plt.ylabel('Relative Intensity')
            plt.xlabel('Wavelength (nm)')
            plt.legend(loc='best')
            plt.suptitle(f'{Shot_time}ps')
            if Save_info:
                save_cmaes_file(CMAES_loc, EXPLORE.param_names, Best_inits, EXPLORE.param_units)
                array = np.array([Run.raw_lambda, scaled_f]).T
                np.savetxt(os.path.join(CMAES_loc, 'IAW.txt'), array)
                plt.savefig(os.path.join(CMAES_loc, 'Best_fit.png'))
            plt.show()
        else:
            try:
                Best_inits = extract_cmaes_file(CMAES_loc, EXPLORE.param_names)
                Best_norm_inits = Run.normalize(Best_inits)
                print('Extracted best initial values from CMAES fit')
            except:
                Best_norm_inits = EXPLORE.norm_param_inits
                Best_inits = EXPLORE.param_inits

        if not Only_CMAES:
            if Run_MCMC:
                print('... Running MCMC')
                os.system('rm idata.nc')
                idata = Run.run_mcmc_model(norm_param_inits=Best_norm_inits)
                idata.to_netcdf(os.path.join(os.getcwd(), 'idata.nc'))

                idata = az.from_netcdf(os.path.join(os.getcwd(), 'idata.nc'))
                Params_out = Run.process_idata(idata, percentages=[50, 100], save=Save_info)
                print(idata.posterior)

                Cropped_parameters = np.asarray([Params_out[p] for p in Names])
                if Save_info:
                    # Create a DataFrame with Mins, Maxs, and Best_inits
                    df_params = pd.DataFrame({
                        'Unit': Units,
                        'Min': Mins,
                        'Max': Maxs,
                        'Init': Best_inits,
                        'Logbool': Logbool
                        }, index=Names)
                    df_params.to_csv(os.path.join(MCMC_loc, 'MCMC_parameter_bounds.csv'))
                    for i, p in enumerate(Names):
                        np.savetxt(os.path.join(MCMC_loc, f'{p}.txt'), Params_out[p])

            else:
                idata = az.from_netcdf(os.path.join(MCMC_loc, 'idata.nc'))
                Params_out = Run.process_idata(idata, percentages=[50, 100], save=False)
                Cropped_parameters = np.asarray([np.genfromtxt(os.path.join(MCMC_loc, f'{name}.txt')) for name in Names])

            if Show_fits:
                fig, axs = plt.subplots()
                if Run_fits:
                    Number_fits = 0
                    for i in range(0, len(Cropped_parameters[0]), 1):
                        Number_fits += 1
                        if Number_fits <= 1000:
                            Params = Cropped_parameters[:, i]
                            Fit_y = Run.generate_fit(Params)
                            if Save_info:
                                array = np.asarray([Run.raw_lambda, Fit_y]).T
                                np.savetxt(os.path.join(Save_fits_loc, f'IAW_{Number_fits}.txt'), array)
                        else: pass

                Fitted_Files = sorted(glob.glob(os.path.join(Save_fits_loc, '*.txt')))

                def get_fit(file):
                    Data = np.genfromtxt(file)
                    Lambda = Data[:, 0]
                    Intensity = Data[:, -1]
                    return Lambda, Intensity / np.nanmax(Intensity)

                for f in Fitted_Files:
                    Fit = np.genfromtxt(f)[:, -1]
                    scaled_f = Fit_OTS().scalings(EXPLORE.sigma, Run.raw_I, Run.raw_err, Fit)
                    plt.plot(Run.raw_lambda, scaled_f, '-', color='#20cc5b', alpha=1)

                axs.errorbar(Run.raw_lambda, Run.raw_I, yerr=np.abs(Run.raw_err), color='gray',
                            capsize=4, barsabove=True, linestyle='None', marker='D', markersize=2,
                            alpha=0.5)
                axs.plot(Run.raw_lambda, Run.raw_I, 'k-', alpha=0.7, label='Raw data')

                Mean_params = np.mean(Cropped_parameters, axis=-1)
                scaled_f = Run.generate_fit(Mean_params)
                axs.plot(Run.raw_lambda, scaled_f, 'r-', label='Mean fit')
                plt.ylabel('Relative Intensity')
                plt.xlabel('Wavelength (nm)')
                axs.minorticks_on()
                axs.tick_params('both', which='minor', direction='in', length=4)
                axs.tick_params('both', which='major', direction='in', length=9)
                plt.legend(loc='best')
                plt.suptitle(f'{Shot_time}ps')
                if Save_info:
                    plt.savefig(os.path.join(MCMC_loc, 'MCMC_fits.png'))
                plt.show()

        os.system('rm MCMC_fitting_info.csv')