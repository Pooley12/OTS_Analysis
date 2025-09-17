#%%
import sys
import scipy.constants as cst
import numpy as np
import os
import matplotlib.pyplot as plt
## Add the Libraries folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Libraries'))
from normalization import Data_normalisation
from OTSpower import OTSpwr
from OTSpower import kangsm_with_file

#%%
class Setup_OTS:

    def __init__(self):
        from setup import Exploration
        self.EXPLORE = Exploration()
        self.laser_inputs()
        self.fixed_inputs()
        self.parameter_exploration()

    def __getattr__(self, name):
        return getattr(self, name)

    def laser_inputs(self):
        ## Units:
        ## m, m, deg, W, m
        self.wavelength = self.EXPLORE.wavelength
        self.wavelength_fwhm = self.EXPLORE.wavelength_fwhm
        self.scattering_angle = self.EXPLORE.scattering_angle
        self.power = self.EXPLORE.power
        self.Lts = self.EXPLORE.Lts
        self.fcol = self.EXPLORE.fcol

        self.omgL = cst.c * 2 * np.pi / (self.wavelength) # rad/s
        # self.omg = np.linspace(-0.005 * self.omgL, 0.005 * self.omgL, 2000)
        self.omg = np.linspace(-0.0015 * self.omgL, 0.0015 * self.omgL, 5000) # rad/s
        self.solangcol = ((0.5 / self.fcol) ** 2 * np.pi)
        self.res_om = (self.wavelength_fwhm / self.wavelength) * self.omgL/(2*np.pi)  # frequency resolution 1/s
        self.wvlgnth = (cst.c * 2 * np.pi / (self.omg + self.omgL))

    def fixed_inputs(self):
        self.z = self.EXPLORE.z

    def parameter_exploration(self):
        self.mins = self.EXPLORE.mins
        self.maxs = self.EXPLORE.maxs
        self.logbool = self.EXPLORE.logbool

class Run_OTS:

    def __init__(self):
        self.SETUP = Setup_OTS()

    def run_fitting(self, te, ti, ne, e_cur, flow, v_grad, core=[1], unnormalise=True):

        if unnormalise:
            norm_params = [te, ti, ne, e_cur, flow, v_grad]
            orig_params = []
            for i in range(0, len(norm_params), 1):
                orig_params.append(Data_normalisation().unnormalise_data(norm_params[i], self.SETUP.mins[i], self.SETUP.maxs[i], log=self.SETUP.logbool[i]))
        else:
            orig_params = [te, ti, ne, e_cur, flow, v_grad]
        cwd = os.getcwd()
        output_folder = os.path.join(cwd, 'chains', '{}'.format(core[0]))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            os.system('scp input.dat {}/.'.format(output_folder))
        os.chdir(output_folder)
        self.parameter_inputs(orig_params[0], orig_params[1], orig_params[2], orig_params[3], orig_params[4], orig_params[5])
        self.write_input()
        self.run()
        output_x, output_I = self.convolve_data()
        os.chdir(cwd)
        return output_x, output_I

    def parameter_inputs(self, te, ti, ne, e_cur, flow, v_grad):
        ## Units
        ## eV, eV, /cc, abs, m/s, nm
        try: self.e_temp = te[0]
        except: self.e_temp = te
        try: self.i_temp = ti[0]
        except: self.i_temp = ti
        try: self.e_dens = ne[0]
        except: self.e_dens = ne
        try: self.e_cur = e_cur[0]
        except: self.e_cur = e_cur
        try: self.flow = flow[0]
        except: self.flow = flow
        try: self.v_grad = v_grad[0]
        except: self.v_grad = v_grad

    def write_input(self, deck='input.dat'):
        with open(deck, 'r') as f:
            input_data = f.readlines()
            for i in range(0, len(input_data), 1):
                line = input_data[i]
                if line.startswith('LAMBDA_LASER'):
                    input_data[i] = 'LAMBDA_LASER	{}\n'.format(self.SETUP.wavelength*1e9)
                elif line.startswith('SCATTERING_ANGLE'):
                    input_data[i] = 'SCATTERING_ANGLE	{}\n'.format(self.SETUP.scattering_angle)
                elif line.startswith('ELECTRON_TEMP'):
                    input_data[i] = 'ELECTRON_TEMP		{}		0\n'.format(self.e_temp)
                elif line.startswith('ION_TEMP'):
                    input_data[i] = 'ION_TEMP		{}		0\n'.format(self.i_temp)
                elif line.startswith('ELECTRON_DENSITY'):
                    input_data[i] = 'ELECTRON_DENSITY	{}		0\n'.format(self.e_dens*1e6)
                elif line.startswith('Z_FREE'):
                    input_data[i] = 'Z_FREE                 {}              0\n'.format(self.SETUP.z)
                elif line.startswith('E_CURRENT'):
                    input_data[i] = 'E_CURRENT		{}              0\n'.format(self.e_cur)
                elif line.startswith('FLOW'):
                    input_data[i] = 'FLOW			{}\n'.format(self.flow)
                elif line.startswith('N_DAWSON'):
                    input_data[i] = 'N_DAWSON               1024\n'
                elif line.startswith('N_FFT'):
                    input_data[i] = 'N_FFT                  1024\n'
                elif line.startswith('INST_FWHM'):
                    input_data[i] = 'INST_FWHM              {}\n'.format(self.SETUP.wavelength_fwhm*1e9/(2 * np.sqrt(2 * np.log(2))))
                elif line.startswith('SALPETER'):
                    input_data[i] = 'SALPETER		1		0\n'
                elif line.startswith('TOTAL'):
                    input_data[i] = 'TOTAL			0		0\n'
                elif line.startswith('LAMBDA_MIN'):
                    input_data[i] = 'LAMBDA_MIN		{}\n'.format(self.SETUP.wavelength*1e9-0.65)
                elif line.startswith('LAMBDA_MAX'):
                    input_data[i] = 'LAMBDA_MAX		{}\n'.format(self.SETUP.wavelength*1e9+0.65)
                elif line.startswith('LAMBDA_STEP'):
                    input_data[i] = 'LAMBDA_STEP            0.005\n'
                elif line.startswith('SAVE_FILE'):
                    input_data[i] = 'SAVE_FILE		IAW.txt\n'
        with open(deck, 'w') as f:
            f.writelines(input_data)

    def run(self):
        os.system('/Users/hpoole/Documents/Coding/OTS_code/ots.x > ots_output.txt')
        # os.system('rm INST_FWHM')

    def read_data(self, file):
        d = np.genfromtxt(file)
        Wavelength = d[:, 0]
        I = d[:, -1]
        return Wavelength, I

    def convolve_data(self, plot=False):

        Fit_lambda, Fit_I = self.read_data('IAW.txt')

        flows = np.linspace(self.flow - 3 * self.v_grad, self.flow + 3 * self.v_grad, 300)
        spec_shift = flows - self.flow
        spec_shifts = np.repeat(spec_shift[:, np.newaxis], len(Fit_lambda), axis=-1).T
        Fit_wvlgths = np.repeat(Fit_lambda[:, np.newaxis], len(flows), axis=-1)
        Shifted_wvlgths = Fit_wvlgths - spec_shifts

        if self.v_grad >= 1e-10:
            scalings = np.exp(-(3 * np.power(spec_shifts, 2)) / (np.power(self.v_grad, 2))) * np.abs(flows[1] - flows[0])
            scalings = scalings / np.nanmax(scalings)

            Fit_ys = np.repeat(Fit_I[:, np.newaxis], len(flows), axis=-1) * scalings
            New_fits = np.array(
                [np.interp(Fit_lambda, Shifted_wvlgths[:, i], Fit_ys[:, i]) for i in range(0, len(flows), 1)]).T
            Grad_IAW = np.nansum(New_fits, axis=-1)
            Grad_IAW = Grad_IAW / np.nanmax(Grad_IAW)
            Fit_I = Grad_IAW
        else:
            Fit_I = Fit_I

        Fit_w = (cst.c * 2 * np.pi / (Fit_lambda * 1e-9)) - self.SETUP.omgL

        Fit_Iw, Fit_w = Fit_I[::-1], Fit_w[::-1]
        Omg_fit = np.interp(self.SETUP.omg, Fit_w, Fit_Iw, left=0, right=0)

        specIAW = kangsm_with_file(self.SETUP.omg, Omg_fit, self.SETUP.scattering_angle)
        powspecIAW = OTSpwr(self.SETUP.power, self.SETUP.Lts, self.SETUP.res_om, self.SETUP.omg, self.SETUP.omgL, self.e_dens, specIAW) * self.SETUP.solangcol  # [w/omega]
        show_powspecIAW = powspecIAW / np.nanmax(powspecIAW)

        # np.savetxt('IAW_conv.txt', np.array([self.wvlgnth[::-1]*1e9, show_powspecIAW[::-1]]).T)
        # os.system('mv IAW_conv.txt Raw_IAW.txt')
        if plot:
            # info = np.genfromtxt('IAW.txt')
            # fit_c = info[:,-1]/np.nanmax(info[:,-1])
            # fit_c = fit_c[::-1]
            #
            # plt.figure()
            # plt.plot(Fit_w, Fit_Iw/np.nanmax(Fit_Iw), 'k-')
            # plt.plot(Fit_w, fit_c, 'b-')
            # plt.plot(self.SETUP.omg, specIAW/np.nanmax(specIAW), 'g-.')
            # plt.plot(self.SETUP.omg, show_powspecIAW, 'r--')
            #
            # plt.xlabel('Wavelength (nm)')
            # plt.ylabel('Relative intensity (arb. u.)')
            # plt.show()
            # quit()

            plt.figure()
            plt.plot(self.SETUP.wvlgnth*1e9, show_powspecIAW, 'k-')
            plt.plot(Fit_lambda, Fit_I/np.nanmax(Fit_I), 'r--')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Relative intensity (arb. u.)')
            plt.show()

        return self.SETUP.wvlgnth[::-1], show_powspecIAW[::-1]

class Fit_OTS:

    def likelihood(self, sigma, Fit, Data, Data_err):
        # Calculate the difference between model and experimental data, normalized by the experimental error
        dif = (Data - Fit) / Data_err
        per = (Data - Fit) / (Data*Data_err)

        # Compute the likelihood term based on the condition
        SB = [
            np.power(dif[i] / (np.sqrt(2) * sigma), 2) + np.log(np.sqrt(2 * np.pi)) + np.log(sigma) if Data[
                                                                                                           i] > np.nanmax(
                Data) * 0.1 else np.log(np.sqrt(2 * np.pi)) + np.log(sigma)
            for i in range(len(Fit))
        ]

        return -np.asarray(SB)

    def scalings(self, sigma, Data_y, Data_err, Fit_y):
        Peak_signal = np.where(Data_y == np.nanmax(Data_y))[0][0]
        Peak_data_signal = Data_y[Peak_signal]
        Peak_error = Data_err[Peak_signal]
        Scalings = np.random.normal(Peak_data_signal, Peak_error, size=500)
        Scales = (np.zeros(len(Scalings)) + 1) / Scalings
        # print(Scales)
        # quit()
        Scaled_fit_signals = np.array([s * Fit_y for s in Scales])

        Data_signals = np.tile(Data_y, (len(Scalings), 1))
        Data_errors = np.tile(Data_err, (len(Scalings), 1))

        calc = (Data_signals - Scaled_fit_signals) / (Data_errors)
        costs = np.power(calc / (np.sqrt(2) * sigma), 2)
        sums = np.nansum(costs, axis=1)
        loc = np.argmin(sums)
        # maxs = np.nanmax(costs, axis=1)
        # loc = np.argmin(maxs)
        # Cost = maxs[loc]
        Scale = Scales[loc]
        Fit_y = Scale * Fit_y


        # Fit_y /= np.nanmax(Fit_y)
        return Fit_y