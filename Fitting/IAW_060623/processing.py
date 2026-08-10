
import os
import sys
import pymc as pm
import numpy as np
import glob
import scipy.constants as cst
## Add the Libraries folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Libraries'))
from normalisation import Data_normalisation
from OTS_function import initialize, OTS, calculations
from mcmc import Fit_OTS, LogLike_nograd
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy.stats import gaussian_kde

class Load_parameters():
    def __init__(self, MCMC_loc=None, CMAES_loc=None):
        self.MCMC_loc = MCMC_loc
        self.CMAES_loc = CMAES_loc

        if self.MCMC_loc is not None:
            self.load_mcmc_params()
        if self.CMAES_loc is not None:
            self.extract_cmaes_file()

    def load_mcmc_params(self):
        ## Read the MCMC parameter bounds dataframe
        df_params = pd.read_csv(os.path.join(self.MCMC_loc, 'MCMC_parameter_bounds.csv'), index_col=0)
        self.Names = df_params.index.tolist()
        self.Units = df_params['Unit'].tolist()
        self.Mins = df_params['Min'].tolist()
        self.Maxs = df_params['Max'].tolist()
        self.Logbool = df_params['Logbool'].tolist()
        self.Inits = df_params['Init'].tolist()

    def extract_cmaes_file(self):
        cmaes_file = os.path.join(self.CMAES_loc, 'Plasma_parameters.txt')
        params = []
        names = []
        with open(cmaes_file, 'r') as d:
            lines = d.readlines()
            if not hasattr(self, 'Names'):
                for line in lines:
                    names.append(line.split()[0])
                self.Names = names
            for i in range(len(self.Names)):
                for j in range(len(lines)):
                    if self.Names[i] in lines[j]:
                        params.append(float(lines[j].split()[1]))
        self.CMAES_inits = np.asarray(params)
        return

    def unnormalize(self, params):
        orig_params = []
        for i in range(0, len(params), 1):
            orig_params.append(Data_normalisation().unnormalise_data(params[i], self.Mins[i], self.Maxs[i], log=self.Logbool[i]))
        return np.asarray(orig_params)
    
    def normalize(self, params):
        norm_params = []
        for i in range(0, len(params), 1):
            norm_params.append(Data_normalisation().normalise_data(params[i], self.Mins[i], self.Maxs[i], log=self.Logbool[i]))
        return np.asarray(norm_params)

class Convergence_testing():
    def __init__(self, INFO):
        self.INFO = INFO

    def process_idata(self, idata=None, percentages=[50, 100], save=False, plot=False):
        ## Process the idata output from MCMC fitting
        if idata is None:
            idata = self.trace

        posterior = idata.posterior
        n_params = len(self.INFO.Names)
        n_chains = posterior.sizes['chain']
        n_draws = posterior.sizes['draw']

        data_fraction = [int(d*0.01*n_draws) for d in percentages]

        ## Plot idata with arviz
        # az.plot_trace(idata, Names)
        # plt.show()

        if plot:
            ncols = int(n_params)
            nrows = 2
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

            samples_list = [posterior[p].values[c][data_fraction[0]:data_fraction[1]].reshape(-1) for p in self.INFO.Names]
            values = np.vstack(samples_list).T
            unnorm_values = np.array([self.INFO.unnormalize(values[i]) for i in range(len(values))])
            all_params[:, c, :] = unnorm_values.T

            samples = {self.INFO.Names[p]: unnorm_values[:, p] for p in range(n_params)}

            if plot:
                for i, p in enumerate(self.INFO.Names):
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
        samples = {self.INFO.Names[p]: collapsed_params[p, :] for p in range(n_params)}
        stats = {p: {
                    'median': np.median(samples[p]),
                    'p2.5': np.percentile(samples[p], 2.5),
                    'p97.5': np.percentile(samples[p], 97.5)
                    } for p in self.INFO.Names}

        if plot:
            for i, p in enumerate(self.INFO.Names):
                axs[1, i].set_xlabel(f'{p} [{self.Units[i]}]')
                axs[0, i].set_ylabel('Frequency')
                axs[1, i].set_ylabel('Sample')
                axs[0, i].set_title(p)
                for ax in axs[:, i].flat:
                    ax.axvline(self.Inits[i], color='forestgreen', linestyle=':', label='init' if i == 0 else None)
                    ax.axvline(stats[p]['median'], color='maroon', label='median')
                    ax.axvline(stats[p]['p2.5'], color='k', linestyle='--', label='95% CI' if i == 0 else None)
                    ax.axvline(stats[p]['p97.5'], color='k', linestyle='--')
                    if self.Logbool[i]:
                        ax.set_xscale('log')
                    ax.tick_params('both', which='major', direction='in', length=5)
                if i == 0:
                    axs[0, i].legend()
            fig.tight_layout()
            plt.suptitle(f'{Shot_time}ps MCMC posterior for {percentages[0]}% - {percentages[1]}%', y=1.05, fontsize=20)
            if save:
                idata.to_netcdf(os.path.join(self.INFO.MCMC_loc, 'idata.nc'))
                plt.savefig(os.path.join(self.INFO.MCMC_loc, 'idata.png'))
            fig.show()
        return samples

    def convergence_checks(self, idata=None):
        if idata is None:
            idata = self.trace

        self.summary = az.summary(idata, var_names=self.INFO.Names)
        # print(self.summary)

        def rhat():
            """
            Calculate and check R-hat values for convergence.
            R-hat provides a way to quantify whether multiple chains have converged to the same distribution.
            R-hat = sqrt((Variance Between Chains) / (Variance Within Chains))
            R-hat < 1.05:
                Generally indicates good convergence.
            R-hat between 1.05 and 1.1:
                Suggests potential convergence issues; further investigation needed.
            R-hat > 1.1:
                Indicates lack of convergence; chains may not have mixed well.
            """
            for name in self.INFO.Names:
                rhat = self.summary.loc[name, 'r_hat']
                if rhat > 1.01:
                    print(f'Parameter {name} may not have converged based on R-hat > 1.01 criterion.')
            return

        def ess():
            """
            Calculate and check Effective Sample Size (ESS) for convergence.
            ESS Bulk - ESS for the bulk of the posterior distribution (reflects how well chains are mixing for main mass of distribution).
            ESS Tail - ESS for the tails of the posterior distribution (checks sampler is exploring rare events or outliers).
            ESS estimates the number of independent samples in a correlated chain.
            Higher ESS values indicate better sampling efficiency.
            ESS > 400:
                Generally indicates good convergence and sufficient sampling.
            ESS between 100 and 400:
                Suggests potential convergence issues; further investigation needed.
            ESS < 100:
                Indicates lack of convergence; chains may not have mixed well.
            """
            for name in self.INFO.Names:
                ess_bulk = self.summary.loc[name, 'ess_bulk']
                ess_tail = self.summary.loc[name, 'ess_tail']
                if ess_bulk < 400:
                    print(f'Parameter {name} may not have converged based on ESS bulk < 400 criterion.')
                if ess_tail < 400:
                    print(f'Parameter {name} may not have converged based on ESS tail < 400 criterion.')
            return

        def autocorr(max_lag=200):
            """
            Plot autocorrelation for each parameter to visually inspect mixing.
            Autocorrelation measures how correlated samples are with themselves at different lags.
            Rapid decay of autocorrelation indicates good mixing and convergence.
            Slow decay or high autocorrelation at large lags suggests poor mixing and potential convergence issues.
            """
            for name in self.INFO.Names:
                samples = idata.posterior[name].values.reshape(-1)
                autocorr_data = az.autocorr(samples)
                autocorr_data = autocorr_data[:max_lag+1]
                # Assess rapid decay
                if np.all(np.abs(autocorr_data[25:50]) > 0.1):
                    print(f"{name}: Autocorr decays slowly (potential mixing issue).")

                # plt.figure(figsize=(8, 4))
                # plt.bar(range(len(autocorr_data)), autocorr_data)
                # plt.xlabel('Lag')
                # plt.ylabel('Autocorrelation')
                # plt.title(f'Autocorrelation Plot for {name}')
                # plt.show()
            az.plot_autocorr(idata, var_names=self.INFO.Names, max_lag=max_lag, combined=True)
            plt.suptitle(f'Autocorrelation plots', fontsize=16)
            return

        def rank():
            """
            Plot rank plots for each parameter to visually inspect mixing.
            Rank plots show the distribution of ranks of samples from different chains.
            Well-mixed chains should have overlapping rank distributions.
            Non-overlapping or distinct rank distributions suggest poor mixing and potential convergence issues.
            """
            az.plot_rank(idata, var_names=self.INFO.Names, kind='vlines',vlines_kwargs={'lw':0}, marker_vlines_kwargs={'lw':3})
            # az.plot_rank(idata, var_names=self.Names)
            plt.suptitle(f'Rank plots', fontsize=16)
            return
        
        def pair():
            """
            Plot pairplot for joint posterior relationships between parameters.
            Pairplots show scatter plots and density estimates for pairs of parameters.
            Well-mixed chains should show smooth, continuous distributions without gaps or clusters.
            Disjointed or clustered distributions suggest poor mixing and potential convergence issues.
            """
            az.plot_pair(idata, var_names=self.INFO.Names, kind='kde', marginals=True)
            plt.suptitle(f'Pairplot', fontsize=16)
            return
        rhat()
        ess()
        autocorr()
        rank()
        pair()

class Output_processing():
    def __init__(self, INFO, Save_bool=False):
        self.INFO = INFO
        self.MCMC_loc = INFO.MCMC_loc
        self.Names = INFO.Names
        self.Units = INFO.Units
        self.Inits = INFO.CMAES_inits
        self.Save_bool = Save_bool

    def standard_colors(self):
            ## THIS SETS UP THE COLOURS AND LEVELS FOR THE PDFS (PLOTTING 1-sigma, 2-sigma, 3-sigma)
        LEVELS = [1-0.9973, 1-0.9546, 1-0.682, 1.0]
        LEVELSV = [1-0.9973, 1-0.9546, 1-0.682]

        Colour_range = np.arange(LEVELS[0], LEVELS[-1], LEVELS[1]-LEVELS[0])
        COLOURS = []
        for i in Colour_range:
            if i < LEVELS[1]:
                COLOURS.append(mpl.colors.to_rgb('silver'))
            elif i < LEVELS[2]:
                COLOURS.append(mpl.colors.to_rgb('grey'))
            elif i < LEVELS[3]:
                COLOURS.append(mpl.colors.to_rgb('black'))
        COLOURS.append(mpl.colors.to_rgb('white'))
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Maleo', COLOURS, N=len(COLOURS))

        line_color = 'gray'

        return line_color, cmap, LEVELS, LEVELSV

    def mcmc_matrix(self, add_cs=True, line_color=None, cmap=None, plot=False, plot_kde=True, param_crop=30000):
        Parameters = np.asarray([np.genfromtxt(os.path.join(MCMC_loc, f'{name}.txt')) for name in self.Names])
        if len(Parameters[0]) > param_crop:
            ## Only take last param_crop samples to avoid memory issues
            Parameters = Parameters[:, -param_crop:]
        if add_cs:
            Z = 0.499*6 + 0.438*1 + 0.063*17
            m_i = 1e-3 * (0.499*12.01+0.438*1.01+0.063*35.45) / cst.Avogadro  # kg
            for i, Name in enumerate(self.Names):
                if Name == 'TE':
                    Te = Parameters[i]/cst.physical_constants['kelvin-electron volt relationship'][0]
                if Name == 'TI':
                    Ti = Parameters[i]/cst.physical_constants['kelvin-electron volt relationship'][0]
            mu_x = Z*Te+(5*Ti/3)
            cs = np.sqrt(cst.Boltzmann*(mu_x)/(m_i)) # m/s
            print(np.mean(cs)*1e-3, np.std(cs)*1e-3)



        LEVELS = self.standard_colors()[2]
        if line_color is None:
            line_color = self.standard_colors()[0]
        if cmap is None:
            cmap = self.standard_colors()[1]

        parameter_data = {}
        for i, Name in enumerate(self.Names):
            parameter_data[Name] = {}
            parameter_data[Name]['mean'] = np.nanmean(Parameters[i])
            parameter_data[Name]['std'] = np.nanstd(Parameters[i])

        if plot:
            def add_hists(X, ax, hist_bins=60, add_means=True, orientation='vertical', color=line_color):

                    hist, bin_edges = np.histogram(X, bins=hist_bins)
                    bins = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(0, len(bin_edges)-1)]
                    hist = hist/np.max(hist)

                    # ax.hist(x=X, bins=num_bins, alpha=0.5, orientation=orientation, color=color, weights=[1/y.max()]*x_len)
                    ax.hist(x=bins, weights=hist, bins=bins, alpha=0.35, orientation=orientation, color=color)

                    if add_means:
                        if orientation == 'horizontal':
                            ax.axhline(parameter_data[Name]['mean'], linestyle='--', alpha=0.8, color=color)
                        else:
                            ax.axvline(parameter_data[Name]['mean'], linestyle='--', alpha=0.8, color=color)
                    return

            def add_density_maps(x, y, ax, plot_type='pdf'):
                if plot_type=='pdf':
                    sns.kdeplot(x=x, y=y, fill=True, levels=LEVELS, cmap=cmap, ax=ax, alpha=0.45)
                elif plot_type=='heatmap':
                    cmap_show = cmap
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy)
                    ax.scatter(x, y, c=z, s=2, cmap=cmap_show)
                return
            
            fig, axs = plt.subplots(len(self.Names), len(self.Names), figsize=(3*len(self.Names), 3*len(self.Names)))
            for i, Name in enumerate(self.Names):
                X = Parameters[i]
                if i == len(self.Names)-1:
                    add_hists(X, axs[i, i], orientation='horizontal')
                    axs[i, i].set_ylabel(f'{Name} [{self.Units[i]}]')
                    axs[i, i].yaxis.set_label_position('right')
                    axs[i, i].set_xlabel('Normalized counts')
                    axs[i, i].yaxis.tick_right()
                else:
                    add_hists(X, axs[i, i])
                    axs[i, i].set_xlabel(f'{Name} [{self.Units[i]}]')
                    axs[i, i].xaxis.set_label_position('top')
                    axs[i, i].set_ylabel('Normalized counts')
                    axs[i, i].xaxis.tick_top()
                    if i > 0:
                        axs[i, i].yaxis.set_label_position('right')
                        axs[i, i].yaxis.tick_right()
                for ax in axs[i, i+1:].flat:
                    ax.axis('off')
                
                for ax in axs[:, i].flat:
                    if i < len(self.Names)-1:
                        ax.set_xlim(Parameters[i].min(), Parameters[i].max())
                    else:
                        ax.set_ylim(Parameters[i].min(), Parameters[i].max())
                
                for a, ax in enumerate(axs[i+1:, i].flat):                
                    Y = Parameters[i+a+1]

                    if plot_kde:
                        add_density_maps(X, Y, ax, plot_type='pdf')
                    if i == 0:
                        ax.set_ylabel(f'{self.Names[i+a+1]} [{self.Units[i+a+1]}]')
                    
                    if a+i < len(self.Names)-2:
                        ax.set_xticklabels([])
                for a, ax in enumerate(axs[i, :i].flat):
                    ax.set_ylim(Parameters[i].min(), Parameters[i].max())
                    if a > 0:
                        ax.set_yticklabels([])
                    if i == len(self.Names)-1:
                        ax.set_xlabel(f'{self.Names[a]} [{self.Units[a]}]')

            for ax in axs.flat:
                ax.minorticks_on()
                ax.tick_params('both', which='minor', direction='in', length=4, top=True, right=True, left=True, bottom=True)
                ax.tick_params('both', which='major', direction='in', length=8, top=True, right=True, left=True, bottom=True)
            if self.Save_bool:
                fig.savefig(os.path.join(self.MCMC_loc, 'MCMC_matrix.png'), dpi=300, bbox_inches='tight')
            # fig.show()

        means = [parameter_data[Name]['mean'] for Name in self.Names]
        stds = [parameter_data[Name]['std'] for Name in self.Names]
        df_parameters = pd.DataFrame({'Mean': means, 'Std': stds, 'Unit': self.Units}, index=self.Names)
        if self.Save_bool:
            df_parameters.to_csv(os.path.join(self.MCMC_loc, 'MCMC_parameter_stats.csv'))
            print(f'Saved MCMC parameter stats to {self.MCMC_loc}')
        # print(df_parameters)
        return df_parameters
    
    def mcmc_tracking(self, dataframe, axs, color='gray'):
        means = dataframe['Mean'].values
        stds = dataframe['Std'].values
        units = dataframe['Unit'].values
        Names = dataframe.index.tolist()
        for i, Name in enumerate(Names):
            axs[i].errorbar(int(Shot_time)*1e-3, means[i], yerr=stds[i], marker='D', capsize=4, color=color)
            axs[i].set_ylabel(f'{Name} [{units[i]}]')
        return

    def old_mcmc_comparison(self, MCMC_folder, time, axs, color='blue'):
        file = os.path.join(MCMC_folder, 'Average_info.txt')
        mcmc_avg_info = np.genfromtxt(file, usecols=[-2, -1], skip_header=1)
        for i, j in enumerate([0, 1, 4, 2, 3]):
            axs[i].errorbar(time, mcmc_avg_info[j, 0], yerr=mcmc_avg_info[j, 1], marker='D', capsize=4, color=color)

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

    ###############################################################

    ###############################################################
    ##                                                           ##
    ##                      FILE LOCATIONS                       ##
    ##                                                           ##
    ###############################################################
    # Global_loc = os.path.join('/', 'Users', User, 'Library', 'CloudStorage', 'Box-Box', 'TDYNO_NLUF', 'OMEGA', Shot_day, 'Data')
    Global_loc = os.path.join('/', 'Users', User, 'Documents', 'TDYNO', 'OMEGA', Shot_day, 'Data')
    Parent_loc = os.path.join(Global_loc, str(Shot_number), 'IAW')

    Raw_files_loc = os.path.join(Parent_loc, 'Scattering_strips')
    Raw_files = sorted(glob.glob(os.path.join(Raw_files_loc, '*ps.txt')))
    Raw_times = [Raw_file.replace('{}/'.format(Raw_files_loc), '').replace('ps.txt', '') for Raw_file in Raw_files]

    ## I haven't setup the sound speed calculation...

    # Old_MCMC_loc = os.path.join(Raw_files_loc.replace('Scattering_strips', 'OldFitting'), 'Results', 'MCMC')
    # old_folders = sorted([f for f in os.listdir(Old_MCMC_loc) if os.path.isdir(os.path.join(Old_MCMC_loc, f))])
    # print("Folders in Old_MCMC_loc:", old_folders)

    fig_mcmc, axs_mcmc = plt.subplots(nrows=5, ncols=1, figsize=(6, 2*5))
    for t in range(0, len(Raw_times), 1):
        Shot_time = Raw_times[t]
        Raw_file = os.path.join(Raw_files_loc, Raw_files[t])
        print('\nProcessing time {:.3f} ns ....'.format(int(Shot_time)*1e-3))
        Save_loc = Raw_files_loc.replace('Scattering_strips', 'Results')

        CMAES_loc = os.path.join(Save_loc, 'CMAES', f'{Shot_time}ps')
        MCMC_loc = os.path.join(Save_loc, 'MCMC', f'{Shot_time}ps')
        Save_fits_loc = os.path.join(MCMC_loc, 'Fits')

        INFO = Load_parameters(MCMC_loc=MCMC_loc, CMAES_loc=CMAES_loc)
        OUTPUT = Output_processing(INFO, Save_bool=Save_info)
        MCMC_dataframe =OUTPUT.mcmc_matrix(plot=True)

        OUTPUT.mcmc_tracking(MCMC_dataframe, axs_mcmc)

        # OUTPUT.old_mcmc_comparison(os.path.join(Old_MCMC_loc, old_folders[t]), int(old_folders[t].replace('ps', ''))*1e-3, axs_mcmc)

        CONV = Convergence_testing(INFO)
        ## Read the idata
        idata = az.from_netcdf(os.path.join(MCMC_loc, 'idata.nc'))
        Params_out = CONV.process_idata(idata, percentages=[50, 100], save=False, plot=False)
        ## Run convergence checks
        CONV.convergence_checks(idata)
        # # sys.exit()

    axs_mcmc[-1].set_xlabel('Time (ns)')
    fig_mcmc.show()
    plt.show()

    ## Bring all the MCMC parameter stats together into a single dataframe
    Save_file = os.path.join(Raw_files_loc.replace('Scattering_strips', 'Results'), 'MCMC', f'MCMC_fitting_info_s{Shot_number}.csv')
    
    MCMC_stats_df = {
        'Time (ns)': [],
        'Mean Te (eV)': [],
        'Std Te (eV)': [],
        'Mean Ti (eV)': [],
        'Std Ti (eV)': [],
        'Mean E_cur (km/s)': [],
        'Std E_cur (km/s)': [],
        'Mean Flow (km/s)': [],
        'Std Flow (km/s)': [],
        'Mean V_grad (km/s)': [],
        'Std V_grad (km/s)': [],
    }
    for t in range(0, len(Raw_times), 1):
        Shot_time = Raw_times[t]
        MCMC_loc = os.path.join(Raw_files_loc.replace('Scattering_strips', 'Results'), 'MCMC', f'{Shot_time}ps')
        MCMC_stats_file = os.path.join(MCMC_loc, 'MCMC_parameter_stats.csv')
        if os.path.exists(MCMC_stats_file):
            df = pd.read_csv(MCMC_stats_file, index_col=0)
            df['Time'] =int(Shot_time)*1e-3
            MCMC_stats_df['Time (ns)'].append(round(int(Shot_time)*1e-3, 2))
            MCMC_stats_df['Mean Te (eV)'].append(round(df.loc['TE', 'Mean'], 2))
            MCMC_stats_df['Std Te (eV)'].append(round(df.loc['TE', 'Std'], 2))
            MCMC_stats_df['Mean Ti (eV)'].append(round(df.loc['TI', 'Mean'], 2))
            MCMC_stats_df['Std Ti (eV)'].append(round(df.loc['TI', 'Std'], 2))
            MCMC_stats_df['Mean E_cur (km/s)'].append(round(df.loc['E_CURRENT', 'Mean'], 2))
            MCMC_stats_df['Std E_cur (km/s)'].append(round(df.loc['E_CURRENT', 'Std'], 2))
            MCMC_stats_df['Mean Flow (km/s)'].append(round(df.loc['FLOW', 'Mean'], 2))
            MCMC_stats_df['Std Flow (km/s)'].append(round(df.loc['FLOW', 'Std'], 2))
            MCMC_stats_df['Mean V_grad (km/s)'].append(round(df.loc['VELOCITY_GRADIENT', 'Mean'], 2))
            MCMC_stats_df['Std V_grad (km/s)'].append(round(df.loc['VELOCITY_GRADIENT', 'Std'], 2))
        else:
            print(f'ERROR: MCMC parameter stats file not found for time {Shot_time}ps at {MCMC_stats_file}')

    if Save_info:
        pd.DataFrame(MCMC_stats_df).to_csv(Save_file, index=False)
        print(f'Saved combined MCMC parameter stats to {Save_file}')