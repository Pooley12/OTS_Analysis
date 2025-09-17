import os
import sys
import pymc as pm
import numpy as np
import glob
from multiprocessing import Process, Pool, Queue
from run_IAW import Run_OTS, Fit_OTS
## Add the Libraries folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Libraries'))
from normalization import Data_normalisation
from mcmc_iaw import LogLike_nograd
import matplotlib.pyplot as plt
import arviz as az
import cma
import pandas as pd
os.system('scp input_CHCl.dat input.dat')

class Exploration:

    def __init__(self):
        ## SETTING THE PARAMETERS TO EXPLORE

        ## params -> User friendly naming of parameters
        ## units -> Unit of each parameter
        ## inits -> The initial guess for fitting parameter
        ## mins -> Lower bound on the parameter exploration
        ## maxs -> Upper bound on the parameter exploration
        ## logbool -> Bool function whether to explore the parameter in log space
        self.params = ['ELECTRON_TEMP', 'ION_TEMP', 'ELECTRON_DENSITY', 'E_CURRENT', 'FLOW', 'V_GRAD']
        self.units = ['eV', 'eV', '1/cc', 'nm', 'nm', 'nm']
        # self.inits = [276, 30, 1e+20, 0.05, -0.00255, 10e-2]
        # self.inits = [200, 30, 1e+20, -0.1, 0.32, 10e-2]
        # self.mins = [1, 1, 5e19, -1, -0.3, 0]
        # self.maxs = [500, 500, 8e20, 1, 0.5, 0.3]

        self.inits = [400, 700, 1e+20, -0.1, 0, 10e-2]
        self.mins = [1, 100, 5e19, -1, -0.3, 0]
        self.maxs = [550, 1500, 8e20, 1, 0.5, 0.3]
        self.logbool = [False, False, True, False, False, False]

        ## Normalise the parameters, as MCMC explores parameters between 0 and 1
        self.norm_inits = [Data_normalisation().normalise_data(self.inits[i], self.mins[i], self.maxs[i], log=self.logbool[i]) for i in range(len(self.inits))]

        ## Define any fixed parameters required for fitting
        self.z = 6 # Ionisation

        ## Sigma utilised in cost function of MCMC fitting
        self.sigma = 2.5

        ## Define laser parameters of the scattering system
        self.wavelength = 526.5 * 1e-9  # m
        self.wavelength_fwhm = 0.06672 * 1e-9 # m
        self.scattering_angle = 59.9  # deg
        self.power = 30 / (1e-9)  # W
        self.Lts = 70e-6  # laser path length collected, [m]
        self.fcol = 10  # f number for collection optic

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # Properly call the parent class
        except AttributeError:
            # Handle missing attributes gracefully
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # return getattr(self, name)

class Fitting:

    def __init__(self, shot_day, shot_number, shot_time, raw_file):
        ## CLASS WHERE THE FITTING ALGORITHMS ARE DEFINED

        ## Defines the raw data to fit to
        ## Perform an initial test fit to compare to data
        ## Run optimisation (CMA-ES) fit to obtain best fit to data
        ## Run MCMC model to fit to data

        self.shot_day = shot_day
        self.shot_number = shot_number
        self.shot_time = shot_time
        self.raw_file = raw_file

        self.EXPLORE = Exploration()
        self.initiate()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # Properly call the parent class
        except AttributeError:
            # Handle missing attributes gracefully
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # return getattr(self, name)

    def initiate(self):
        self.experimental_data()

    def experimental_data(self):
        ## Defining the raw data

        Raw_data = np.genfromtxt(self.raw_file)
        self.raw_lambda, self.raw_I, Raw_per_err = Raw_data[:, 0], Raw_data[:, 1], Raw_data[:, 2]
        self.raw_err = self.raw_I * Raw_per_err

    def initial_test(self):
        ## Run an initial test of an example fit
        ## Can help debug any issues in the system
        ## NB: Doesn't perform any fitting

        Fit_x, Fit_y = Run_OTS().run_fitting(*self.EXPLORE.inits, unnormalise=False)
        Fit_y = np.interp(self.raw_lambda, Fit_x * 1e9, Fit_y)
        scaled_f = Fit_OTS().scalings(self.EXPLORE.sigma, self.raw_I, self.raw_err, Fit_y)
        cost = -Fit_OTS().likelihood(self.EXPLORE.sigma, scaled_f, self.raw_I, self.raw_err)

        # Code = np.genfromtxt('chains/1/IAW.txt')
        # Info = [Code[:, 0], Code[:,-1]/np.nanmax(Code[:,-1])]

        fig, axs = plt.subplots()
        axs.errorbar(self.raw_lambda, self.raw_I, yerr=np.abs(self.raw_err), color='gray',
                     capsize=4, barsabove=True, linestyle='None', marker='D', markersize=2,
                     alpha=0.5)
        axs.plot(self.raw_lambda, self.raw_I, 'k-', alpha=0.7, label='Raw data')
        axs.plot(self.raw_lambda, scaled_f, 'r-', label='Starting fit')
        axs.plot(self.raw_lambda, cost / np.nanmax(cost), 'b-', alpha=0.5, label='Cost')
        # axs.plot(Info[0], Info[1], 'g--', alpha=1, label='Code')
        plt.legend(loc='best')
        plt.suptitle('{}'.format(np.sum(cost)))
        plt.show()

    def run_optimisation_model(self):
        ## Perform optimisation model fitting (CMA-ES)
        ## Produces an initial best fit to the data based on the defined cost function

        def function(params):
            ## Define the fitting function
            model_x, model_y = Run_OTS().run_fitting(*params)
            model = np.interp(self.raw_lambda, model_x * 1e9, model_y)  # Interpolate model_y to match x
            scaled_model = Fit_OTS().scalings(self.EXPLORE.sigma, self.raw_I, self.raw_err, model)
            costs = -Fit_OTS().likelihood(self.EXPLORE.sigma*0.5, scaled_model, self.raw_I, self.raw_err)
            return np.sum(costs)  # Return chi-squared error or another error metric

        def generate_fit(*params):
            model_x, model_y = Run_OTS().run_fitting(*params)
            model = np.interp(self.raw_lambda, model_x * 1e9, model_y)  # Interpolate model_y to match x
            scaled_model = Fit_OTS().scalings(self.EXPLORE.sigma, self.raw_I, self.raw_err, model)
            return scaled_model

        ## CMAES options with bounds
        ## By definition the bounds for each parameter are 0 and 1
        lower_bounds, upper_bounds = np.zeros(len(self.EXPLORE.norm_inits)), np.ones(len(self.EXPLORE.norm_inits))
        options = {
            'bounds': [lower_bounds, upper_bounds],  # Set the bounds
            'maxiter': 10000,  # Maximum number of iterations
        }

        es = cma.CMAEvolutionStrategy(x0=self.EXPLORE.norm_inits, sigma0=0.1, options=options)
        opt_result = es.optimize(function)

        # Output the results
        best_params = opt_result.result.xbest  # Best parameter set found

        ## Compare the fit of the initial parameters to the optimised parameters
        Initial_y = generate_fit(*self.EXPLORE.norm_inits)
        Optimised_y = generate_fit(*best_params)
        fig, axs = plt.subplots()
        axs.errorbar(self.raw_lambda, self.raw_I, yerr=np.abs(self.raw_err), color='gray',
                     capsize=4, barsabove=True, linestyle='None', marker='D', markersize=2,
                     alpha=0.5)
        axs.plot(self.raw_lambda, self.raw_I, 'k-', alpha=0.7, label='Raw data')
        axs.plot(self.raw_lambda, Initial_y, 'g-', label='Starting fit')
        axs.plot(self.raw_lambda, Optimised_y, 'r-', label='Optimised fit')
        plt.legend(loc='best')
        plt.show()

        ## Print the optimised parameters found
        unnormalise_best_params = [Data_normalisation().unnormalise_data(best_params[i], self.EXPLORE.mins[i], self.EXPLORE.maxs[i],
                                                                  log=self.EXPLORE.logbool[i]) for i in range(len(best_params))]
        df = np.array([self.EXPLORE.inits, unnormalise_best_params]).T
        dataframe = pd.DataFrame(df, index=self.EXPLORE.params, columns=['Initial', 'Optimised'])
        print('\nOptimal parameters')
        print(dataframe)

        return unnormalise_best_params, best_params

    def run_mcmc_model(self, norm_inits=None, core_num=1):
        ## Perform MCMC fitting function
        ## Runs MCMC fitting for one core

        ## Define the starting values for MCMC fitting process
        if norm_inits is None:
            norm_inits = self.EXPLORE.norm_inits
        norm_inits = np.round(norm_inits, decimals=5)

        with pm.Model():
            running_params = [pm.Uniform(f"{self.EXPLORE.params[i]}", lower=0, upper=1, initval=norm_inits[i]) for i in range(len(norm_inits))]

            ## Define the likelihood function used to assess appropriateness of each fit
            likelihood = pm.CustomDist("Likelihood", running_params[0], running_params[1], running_params[2], running_params[3], running_params[4], running_params[5]
                                       , self.EXPLORE.sigma, core_num, self.raw_lambda, self.raw_err,
                                       observed=self.raw_I, logp=custom_dist_loglike)

            ## Inject a random seed into each chain
            rng = np.random.default_rng(666)

            ## Define the sampling step function
            step = pm.Metropolis()
            # step = pm.NUTS(step_scale=0.05)
            # step = pm.HamiltonianMC(step_scale=0.05)

            ## Run the MCMC sampling with certain number of iterations
            ## The number of chains run on each core can only be 1
            trace = pm.sample(draws=7000, tune=0, chains=1, step=step, cores=1, progressbar=True, random_seed=rng)

            ## Save the output trace
            output_folder = os.path.join(os.getcwd(), 'chains', '{}'.format(core_num))
            trace.to_netcdf(os.path.join(output_folder, 'idata.nc'))
        return trace

def custom_dist_loglike(data, te, ti, ne, e_cur, flow, v_grad, sigma, core, x, data_err):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(te, ti, ne, e_cur, flow, v_grad, sigma, core, x, data_err, data)

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
# loglike_op = LogLikeWithGrad()

if __name__ == '__main__':
    print('main')
    ## Import data
    ###############################################################
    ##                                                           ##
    ##                        DATA TO INPUT                      ##
    ##                                                           ##
    ###############################################################

    ## Define the shot day, shot number, and diagnostic
    Shot_day = 'OMEGA_Jun2023'
    Shot_number = 108617

    ## If using TDYNO_NLUF Box account, User as required in Parent_loc
    User = 'hpoole'

    ## Bools for what you want to run and if you want to save outputs
    Save_info = True
    Run_CMAES = True
    Run_MCMC = True
    Run_fits = True

    ###############################################################

    ###############################################################
    ##                                                           ##
    ##                      FILE LOCATIONS                       ##
    ##                                                           ##
    ###############################################################
    Global_loc = os.path.join('/', 'Users', User, 'Library', 'CloudStorage', 'Box-Box', 'TDYNO_NLUF', 'OMEGA', Shot_day, 'Data')
    Parent_loc = os.path.join(Global_loc, str(Shot_number), 'IAW')

    Raw_files_loc = os.path.join(Parent_loc, 'Scattering_strips')
    Raw_files = sorted(glob.glob(os.path.join(Raw_files_loc, '*ps.txt')))
    Raw_times = [Raw_file.replace('{}/'.format(Raw_files_loc), '').replace('ps.txt', '') for Raw_file in Raw_files]

    for t in range(len(Raw_times)):
        Shot_time = Raw_times[t]
        Raw_file = os.path.join(Raw_files_loc, Raw_files[t])
        print('\nRunning time {} ns ....'.format(int(Shot_time)*1e-3))
        os.system('rm -r chains/*')
        Save_loc = Raw_files_loc.replace('Scattering_strips', 'Results')

        CMAES_loc = os.path.join(Save_loc, 'CMAES', f'{Shot_time}ps')
        MCMC_loc = os.path.join(Save_loc, 'MCMC', f'{Shot_time}ps')
        Save_fits_loc = os.path.join(MCMC_loc, 'Fits')
        if not os.path.exists(Save_fits_loc):
            os.makedirs(Save_fits_loc)
        if not os.path.exists(CMAES_loc):
            os.makedirs(CMAES_loc)

        Names, Mins, Maxs, Logbool = Exploration().params, Exploration().mins, Exploration().maxs, Exploration().logbool

        Run = Fitting(Shot_day, Shot_number, Shot_time, Raw_file)
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

            Fit_x, Fit_y = Run_OTS().run_fitting(Best_inits[0], Best_inits[1], Best_inits[2], Best_inits[3],
                                                 Best_inits[4], Best_inits[5], unnormalise=False)
            Fit_y = np.interp(Run.raw_lambda, Fit_x * 1e9, Fit_y)
            scaled_f = Fit_OTS().scalings(Exploration().sigma, Run.raw_I, Run.raw_err, Fit_y)
            if Save_info:
                save_cmaes_file(CMAES_loc, Exploration().params, Best_inits, Exploration().units)
                array = np.array([Run.raw_lambda, scaled_f]).T
                np.savetxt(os.path.join(CMAES_loc, 'IAW.txt'), array)
            axs.plot(Run.raw_lambda, scaled_f, 'r-', label='CMAES fit')
            axs.minorticks_on()
            axs.tick_params('both', which='minor', direction='in', length=4)
            axs.tick_params('both', which='major', direction='in', length=9)
            plt.ylabel('Relative Intensity')
            plt.xlabel('Wavelength (nm)')
            plt.legend(loc='best')
            plt.suptitle(f'{Shot_time}ps')
            plt.savefig(os.path.join(CMAES_loc, 'Best_fit.png'))
            plt.show()
            # quit()
        else:
            try:
                Best_inits = extract_cmaes_file(CMAES_loc, Exploration().params)
                Best_norm_inits = [Data_normalisation().normalise_data(Best_inits[i], Exploration().mins[i],Exploration().maxs[i],
                                                                      log=Exploration().logbool[i]) for i in range(len(Best_inits))]
                print('Extracted best initial values from CMAES fit')
            except:
                Best_norm_inits = Exploration().norm_inits

        if Run_MCMC:
            print('... Running MCMC')
            # construct a different process for each function
            # idata = Fitting().run_mcmc_model(norm_inits=Best_norm_inits)
            num_cores = 4
            processes = []
            processes = [Process(target=Run.run_mcmc_model, args=(Best_norm_inits, i+1)) for i in range(num_cores)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            for process in processes:
                process.terminate()

            idatas = []
            for i in range(num_cores):
                output_folder = os.path.join(os.getcwd(), 'chains', '{}'.format(i+1))
                idatas.append(az.from_netcdf(os.path.join(output_folder, 'idata.nc')))

            idata = az.concat([idatas[i] for i in range(num_cores)], dim="chain")
            print(idata.posterior)

            az.plot_trace(idata)
            if Save_info:
                idata.to_netcdf(os.path.join(MCMC_loc, 'idata.nc'))
                plt.savefig(os.path.join(MCMC_loc, 'idata.png'))
            plt.show()

            def crop_and_save(parameters, fraction):
                # Calculate cropping index
                crop_index = int(len(parameters[0]) * fraction)
                # Crop each parameter
                cropped_parameters = [param[crop_index:] for param in parameters]
                if Save_info:
                    for i in range(len(cropped_parameters)):
                        np.savetxt(os.path.join(MCMC_loc, f'{Names[i]}.txt'), cropped_parameters[i])
                return np.asarray(cropped_parameters)
            Params_out = [Data_normalisation().unnormalise_data(idata.posterior[f"{Names[i]}"].data.flatten(), Mins[i], Maxs[i], log=Logbool[i]) for i in range(len(Names))]
            Cropped_parameters = crop_and_save(Params_out, 0.5)
        else:
            idata = az.from_netcdf(os.path.join(MCMC_loc, 'idata.nc'))
            Cropped_parameters = [np.genfromtxt(os.path.join(MCMC_loc, f'{name}.txt')) for name in Names]

            print(idata.posterior)
            az.plot_trace(idata, Exploration().params)
            plt.show()

        fig, axs = plt.subplots()
        if Run_fits:
            Number_fits = 0
            for i in range(0, len(Cropped_parameters[0]), 1):
                Number_fits += 1
                if Number_fits <= 1000:
                    Params = Cropped_parameters[:, i]
                    Fit_x, Fit_y = Run_OTS().run_fitting(Params[0], Params[1], Params[2], Params[3], Params[4], Params[5], unnormalise=False)
                    Fit_y = np.interp(Run.raw_lambda, Fit_x * 1e9, Fit_y)
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
            scaled_f = Fit_OTS().scalings(Exploration().sigma, Run.raw_I, Run.raw_err, Fit)
            plt.plot(Run.raw_lambda, scaled_f, '-', color='#20cc5b', alpha=1)

        axs.errorbar(Run.raw_lambda, Run.raw_I, yerr=np.abs(Run.raw_err), color='gray',
                    capsize=4, barsabove=True, linestyle='None', marker='D', markersize=2,
                    alpha=0.5)
        axs.plot(Run.raw_lambda, Run.raw_I, 'k-', alpha=0.7, label='Raw data')

        Mean_params = np.mean(Cropped_parameters, axis=-1)
        Fit_x, Fit_y = Run_OTS().run_fitting(Mean_params[0], Mean_params[1], Mean_params[2], Mean_params[3], Mean_params[4], Mean_params[5],
                                             unnormalise=False)
        Fit_y = np.interp(Run.raw_lambda, Fit_x * 1e9, Fit_y)
        scaled_f = Fit_OTS().scalings(Exploration().sigma, Run.raw_I, Run.raw_err, Fit_y)
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
        sys.exit()