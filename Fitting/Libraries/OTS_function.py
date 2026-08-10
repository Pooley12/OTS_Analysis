import numpy as np
import scipy.constants as cst
from scipy.special import gamma, dawsn
import os
import sys
import fdint
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from numpy import log10
from scipy.interpolate import LinearNDInterpolator, griddata
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

class initialize():

    def model_options(self, use_zbar_eos=True, use_range=True, fit_type='salpeter'):
        self.use_zbar_eos = use_zbar_eos
        self.use_range = use_range
        self.fit_type = fit_type

    def material_params(self, materials, compositions, eos_file=None):
        self.materials = materials
        self.fractions = np.array([float(c/np.sum(compositions)) for c in compositions])
 
        self.NSPEC = len(materials)
        self.atomic_masses()
        
        self.eos_file = eos_file
        if eos_file is not None:
            self.eos_zbar_grid()

    def scattering_params(self, laser_wavelength, scattering_angle, wavelength_fwhm, wavelength_min, wavelength_max, wavelength_step):
        # Wavelength in nm, Angle in degrees

        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.wavelength_step = wavelength_step

        self.wavelength_fwhm = wavelength_fwhm * 1e-9  # m
        self.wavelength = laser_wavelength * 1e-9  # m
        self.theta = np.deg2rad(scattering_angle)  # radians
        self.wavelengths = np.arange(wavelength_min, wavelength_max, wavelength_step) * 1e-9  # m

        self.omgL = calculations.wavelength_to_omega(self.wavelength) # rad/s
        self.omg = calculations.wavelength_to_omega(self.wavelengths) # rad/s

    def plasma_params(self, Te=None, Ti=None, ne=None, e_cur=None, flow=None, ionizations=None, ne_range=None, v_grad=None):
        # Te, Ti in eV, ne in cm^-3, e_cur in nm, flow in nm

        if Te is None:
            pass
        else:
            self.Te = calculations.eV_to_K(Te)  # K

        if Ti is None:
            pass
        else:
            self.Ti = calculations.eV_to_K(Ti)  # K

        if ne is None:
            pass
        else:
            self.ne = ne * 1e6  # m^-3

        if e_cur is None:
            pass
        else:
            self.e_cur = e_cur  # km/s

        if flow is None:
            pass
        else:
            self.flow = flow  # km/s

        if ionizations is None:
            pass
        else:
            self.ionizations = np.array(ionizations, dtype=float)

        if ne_range is None:
            pass
        else:
            self.ne_range = ne_range * 1e6  # m^-3

        if v_grad is None:
            pass
        else:
            self.v_grad = v_grad  # km/s

    def atomic_masses(self):
        ########## CALCULATE MEAN WEIGHT AND IONIZATION OF MATERIAL ##########
        ## This calculates the mean weight and ionization of
        ## the material composition using 'Atomic_data.txt' file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        atomic_data_path = os.path.join(script_dir, 'Atomic_data.txt')
        ATOMIC_DATA = np.genfromtxt(atomic_data_path, skip_header=2, usecols=(0, 1, 3), dtype=None, encoding='utf-8')
        AMU = cst.physical_constants['atomic mass constant'][0] # kg
        Mean_atomic_weight = 0
        self.ZAX = []
        self.ANX = []
        for i, Element in enumerate(self.materials):
            Element_weight = None
            Chemical_name = Element
            Fraction = self.fractions[i]
            for a in ATOMIC_DATA:
                if a[1] == Chemical_name:
                    Element_weight = a[2]
                    Element_ionization = a[0]
                    break
            try:
                self.ZAX.append(Element_ionization)
                self.ANX.append(Element_weight)
                Mean_atomic_weight += Fraction * Element_weight
            except:
                print('Error in Get_Atomic_info\n'
                      '\tAtomic_Masses => Error extracting atomic information.\n'
                      '\tCheck Element {} in Atomic Data file'.format(Chemical_name))
                sys.exit()
        Mass = Mean_atomic_weight * AMU # kg
        self.weight = Mean_atomic_weight
        return 

    def eos_zbar_grid(self):
        if not os.path.exists(self.eos_file):
            raise FileNotFoundError(f"Grid file not found: {self.eos_file}")

        T_min, T_max = 20, 1000 # eV
        Ni_min, Ni_max = 4e18, 2e20 # cm^-3
        with np.load(self.eos_file) as npz:
            Ni_range = np.unique(npz['Ni'])  # cm^-3
            Te_range = np.unique(npz['Te'])  # eV

            ni_low, ni_high = np.argmin(np.abs(Ni_range - Ni_min)), np.argmin(np.abs(Ni_range - Ni_max))
            t_low, t_high = np.argmin(np.abs(Te_range - T_min)), np.argmin(np.abs(Te_range - T_max))

            self.Te_grid = calculations.eV_to_K(npz['Te'][ni_low:ni_high, t_low:t_high])  # K
            self.Ne_grid = npz['Ne'][ni_low:ni_high, t_low:t_high]*1e6  # m^-3
            self.Ni_grid = npz['Ni'][ni_low:ni_high, t_low:t_high]*1e6  # m^-3
            self.Zbar_grid = npz['Zbar'][ni_low:ni_high, t_low:t_high]
            self.Z_element_grids = {}
            for material in self.materials:
                self.Z_element_grids[material] = npz[f'Z{material}'][ni_low:ni_high, t_low:t_high]

        self.pts = np.column_stack((np.log10(self.Te_grid.ravel()), np.log10(self.Ne_grid.ravel())))
        Z_list = [self.Z_element_grids[mat].ravel() for mat in self.materials]
        self.vals_matrix = np.vstack(Z_list).T  # shape (npoints, nmaterials)

        ## I've shrunk the grid to increase interpolation speed; verify the ranges!!
        # plt.figure()
        # plt.contourf(calculations.K_to_eV(self.Te_grid), self.Ni_grid*1e-6, self.Zbar_grid, levels=10)#, norm=LogNorm(), levels=np.arange(5e19, 1e22, 5e19))
        # plt.colorbar(label='Zbar')
        # plt.xlabel('Electron Temperature (eV)')
        # plt.ylabel('Ion Density (cm$^{-3}$)')
        # plt.title('Zbar from EOS Table')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        # plt.figure()
        # plt.contourf(calculations.K_to_eV(self.Te_grid), self.Ni_grid*1e-6, np.log10(self.Ne_grid*1e-6), levels=10)#, norm=LogNorm(), levels=np.arange(5e19, 1e22, 5e19))
        # plt.colorbar(label='log10(Ne)')
        # plt.xlabel('Electron Temperature (eV)')
        # plt.ylabel('Ion Density (cm$^{-3}$)')
        # plt.title('Electron Density from EOS Table')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        # print(np.min(calculations.K_to_eV(self.Te_grid)), np.max(calculations.K_to_eV(self.Te_grid)))
        # print(np.min(self.Ne_grid)*1e-6, np.max(self.Ne_grid)*1e-6)
        # print(np.min(self.Ni_grid)*1e-6, np.max(self.Ni_grid)*1e-6)
        # sys.exit()
        
        ## Build and cache interpolators to avoid repeated expensive setup
        # Build interpolators using scattered-data ND interpolation because the Te/Ne grids are not guaranteed regular
        Te_log = np.log10(self.Te_grid.ravel())
        Ne_log = np.log10(self.Ne_grid.ravel())
        pts = np.column_stack((Te_log, Ne_log))
        self.Z_element_interpolators = {}

        for material, Z_grid in self.Z_element_grids.items():
            vals = Z_grid.ravel()
            self.Z_element_interpolators[material] = LinearNDInterpolator(pts, vals, fill_value=np.nan)

class OTS():
    def __init__(self, init):
        self.init = init
        self.get_params()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # Properly call the parent class
        except AttributeError:
            # Handle missing attributes gracefully
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_params(self):
        self.materials = self.init.materials
        self.fractions = self.init.fractions
        self.NSPEC = self.init.NSPEC
        self.weight = self.init.weight
        self.wavelength = self.init.wavelength
        self.wavelength_fwhm = self.init.wavelength_fwhm
        self.theta = self.init.theta
        self.wavelengths = self.init.wavelengths
        self.omgL = self.init.omgL
        self.omg = self.init.omg
        self.ANX = self.init.ANX
        self.ZAX = self.init.ZAX

        self.use_zbar_eos = self.init.use_zbar_eos
        self.use_range = self.init.use_range
        self.fit_type = self.init.fit_type

        try:
            self.Te = self.init.Te
        except AttributeError:
            pass
        try:
            self.Ti = self.init.Ti
        except AttributeError:
            pass
        try:
            self.ne = self.init.ne
        except AttributeError:
            pass
        try:
            self.e_cur = self.init.e_cur
        except AttributeError:
            pass
        try:
            self.flow = self.init.flow
        except AttributeError:
            pass
        try:
            self.ionizations = self.init.ionizations
        except AttributeError:
            pass
        try:
            self.ne_range = self.init.ne_range
        except AttributeError:
            pass
        try:
            self.v_grad = self.init.v_grad
        except AttributeError:
            pass

    def salpeter(self, Te=None, Ti=None, ne=None):
        if Te is None:
            Te = self.Te
        if Ti is None:
            Ti = self.Ti
        if ne is None:
            ne = self.ne

        if not hasattr(self, 'e_cur'):
            e_cur = 0
        else:
            e_cur = self.e_cur
        if not hasattr(self, 'flow'):
            flow = 0
        else:
            flow = self.flow

        if Te <= 0.0 or Ti <= 0.0 or ne <= 0.0:
            print("Error: Te, Ti, and ne must be set to positive values before calling salpeter().")
            return 0.0
    
        self.dv = self.omgL*e_cur/self.wavelength
        self.LS = self.omgL*flow/self.wavelength

        w = self.omg-self.omgL

        self.wpe = calculations.plasma_frequency(ne) # rad/s
        self.vte = calculations.thermal_velocity(Te, factor=2) # m/s
        self.alpha = calculations.scattering_parameter(ne, Te, self.wavelength, self.theta)
        self.k = calculations.scattering_wavevector(self.wavelength, self.theta, ne)

        xe = (w - self.dv - self.LS) / (self.k * self.vte)
        we_r = 1-2*xe*calculations.dawson(xe)
        we_i = np.sqrt(np.pi)*xe*np.exp(-np.power(xe,2))

        wi_r = 0.0
        wi_i = 0.0
        for i in range(self.NSPEC):
            M = self.ANX[i] * cst.physical_constants['atomic mass constant'][0] # kg
            vti = calculations.thermal_velocity(Ti, m=M, factor=2) # m/s
            xi = (w - self.LS) / (self.k * vti)
            if self.ionizations[i] > self.ZAX[i]:
                print('Error in Get_Atomic_info\n'
                      '\tAtomic_Masses => Ionization state greater than atomic number.\n'
                      '\tCheck Element {} information'.format(self.materials[i]))
                sys.exit()
            wi_r += np.power(self.ionizations[i], 2) * self.fractions[i] * (1.0 - 2.0 * xi * calculations.dawson(xi))
            wi_i += np.power(self.ionizations[i], 2) * self.fractions[i] * np.sqrt(np.pi) * xi * np.exp(-np.power(xi, 2))

        Ntot = np.sum(self.fractions*self.ionizations)
        s1_r = 1.0 + (1.0 / Ntot) * np.power(self.alpha, 2) * (Te / Ti) * wi_r
        s1_i = (1.0 / Ntot) * np.power(self.alpha, 2) * (Te / Ti) * wi_i
        s2_r = 1.0 + np.power(self.alpha, 2) * we_r + (1.0 / Ntot) * np.power(self.alpha, 2) * (Te / Ti) * wi_r
        s2_i = np.power(self.alpha, 2) * we_i + (1.0 / Ntot) * np.power(self.alpha, 2) * (Te / Ti) * wi_i
        s3_r = -np.power(self.alpha, 2) * we_r
        s3_i = -np.power(self.alpha, 2) * we_i
        
        s1 = s1_r + 1j * s1_i
        s2 = s2_r + 1j * s2_i
        s3 = s3_r + 1j * s3_i

        ans = abs(s1 / s2) ** 2 * np.exp(-np.power(xe, 2)) / (self.k * np.sqrt(np.pi) * self.vte)
        for i in range(self.NSPEC):
            M = self.ANX[i] * cst.physical_constants['atomic mass constant'][0] # kg
            vti = calculations.thermal_velocity(Ti, m=M, factor=2) # m/s
            xi = (w - self.LS) / (self.k * vti)
            ans += np.power(self.ionizations[i], 2) * self.fractions[i] / Ntot * abs(s3 / s2) ** 2 * np.exp(-np.power(xi, 2)) / (self.k * np.sqrt(np.pi) * vti)
        self.output_x = self.wavelengths
        self.output_I = ans
        return self.output_x, self.output_I

    def salpeter_range(self, Te=None, Ti=None, ne=None):
        if Te is None:
            Te = self.Te
        if Ti is None:
            Ti = self.Ti
        if ne is None:
            ne = self.ne
        if not hasattr(self, 'e_cur'):
            e_cur = 0
        else:
            e_cur = self.e_cur
        if not hasattr(self, 'flow'):
            flow = 0
        else:
            flow = self.flow

        if Te <= 0.0 or Ti <= 0.0 or ne <= 0.0:
            print("Error: Te, Ti, and ne must be set to positive values before calling salpeter().")
            return 0.0
        
        angddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'angles')
        angs = np.deg2rad(np.load(angddir + '/angOMEGA.npy'))
        fangs = np.load(angddir + '/fangOMEGA.npy')
  
        thetas = self.theta + angs
        sa_fractions = fangs

        self.dv = self.omgL*e_cur/self.wavelength
        self.LS = self.omgL*flow/self.wavelength

        w = self.omg-self.omgL
        kappa_e = calculations.inverse_screening_length(ne, Te)
        self.wpe = calculations.plasma_frequency(ne) # rad/s
        self.vte = calculations.thermal_velocity(Te, factor=2) # m/s

        # Vectorize the sal() function for speed using numpy broadcasting
        ks = np.array([calculations.scattering_wavevector(self.wavelength, theta, ne) for theta in thetas])

        # Precompute constants and arrays for all ks
        Nw = len(self.wavelengths)
        Ntheta = len(ks)
        Nion = self.NSPEC

        # Prepare arrays for all ks and all wavelengths
        w = self.omg - self.omgL  # shape (Nw,)
        dv = self.dv
        LS = self.LS
        vte = self.vte
        fractions = np.array(self.fractions)
        ionizations = np.array(self.ionizations)
        ANX = np.array(self.ANX)
        ZAX = np.array(self.ZAX)
        atomic_mass = cst.physical_constants['atomic mass constant'][0]

        # Allocate output
        Is = np.zeros((Ntheta, Nw))

        for j, k in enumerate(ks):
            alpha = kappa_e / k
            xe = (w - dv - LS) / (k * vte)
            we_r = 1 - 2 * xe * calculations.dawson(xe)
            we_i = np.sqrt(np.pi) * xe * np.exp(-np.power(xe, 2))

            # Ion terms
            wi_r = np.zeros(Nw)
            wi_i = np.zeros(Nw)
            for i in range(Nion):
                M = ANX[i] * atomic_mass
                vti = calculations.thermal_velocity(Ti, m=M, factor=2)
                xi = (w - LS) / (k * vti)
                if ionizations[i] > ZAX[i]:
                    print('Error in Get_Atomic_info\n'
                          '\tAtomic_Masses => Ionization state greater than atomic number.\n'
                          '\tCheck Element {} information'.format(self.materials[i]))
                    sys.exit()
                wi_r += ionizations[i]**2 * fractions[i] * (1.0 - 2.0 * xi * calculations.dawson(xi))
                wi_i += ionizations[i]**2 * fractions[i] * np.sqrt(np.pi) * xi * np.exp(-np.power(xi, 2))

            Ntot = np.sum(fractions * ionizations)
            s1_r = 1.0 + (1.0 / Ntot) * alpha**2 * (Te / Ti) * wi_r
            s1_i = (1.0 / Ntot) * alpha**2 * (Te / Ti) * wi_i
            s2_r = 1.0 + alpha**2 * we_r + (1.0 / Ntot) * alpha**2 * (Te / Ti) * wi_r
            s2_i = alpha**2 * we_i + (1.0 / Ntot) * alpha**2 * (Te / Ti) * wi_i
            s3_r = -alpha**2 * we_r
            s3_i = -alpha**2 * we_i

            s1 = s1_r + 1j * s1_i
            s2 = s2_r + 1j * s2_i
            s3 = s3_r + 1j * s3_i

            ans = np.abs(s1 / s2) ** 2 * np.exp(-np.power(xe, 2)) / (k * np.sqrt(np.pi) * vte)
            for i in range(Nion):
                M = ANX[i] * atomic_mass
                vti = calculations.thermal_velocity(Ti, m=M, factor=2)
                xi = (w - LS) / (k * vti)
                ans += ionizations[i]**2 * fractions[i] / Ntot * np.abs(s3 / s2) ** 2 * np.exp(-np.power(xi, 2)) / (k * np.sqrt(np.pi) * vti)
            Is[j, :] = ans

        self.output_x = self.wavelengths
        self.output_I = np.sum(Is * sa_fractions[:, np.newaxis], axis=0)

        return self.output_x, self.output_I

    def density_range(self, use_range=False, fit_type='salpeter'):
        def density_gaussian(x, mu, sig):
            gaus = np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2)))
            return gaus/np.nanmax(gaus)

        def get_spectral_fit_salpeter(ne, use_range):
            if use_range:
                out_lambda, out_I = self.salpeter_range(ne=ne)
            else:
                out_lambda, out_I = self.salpeter(ne=ne)
            return out_lambda, out_I/np.nanmax(out_I)

        def get_spectral_fit_bohm_gross(theta):
            nes = np.logspace(18, 21, 1000) * 1e6  # m^-3
            ne_array = density_gaussian(nes, self.ne, self.ne_range)
            epw_peaks = calculations.bohm_gross_wavelength(self.wavelength, theta, self.Te, nes)
            signal = epw_peaks * ne_array
            signal /= np.nanmax(signal)
            order = np.argsort(epw_peaks)
            fit_lambda = epw_peaks[order]
            fit_I = signal[order]

            out_lambda = self.wavelengths
            out_I = np.interp(out_lambda, fit_lambda, fit_I, left=0, right=0)
            
            return out_lambda, out_I/np.nanmax(out_I)

        if fit_type == 'salpeter':
            nes = np.logspace(18, 21, 5000) * 1e6  # m^-3
            ne_array = density_gaussian(nes, self.ne, self.ne_range)
            selection = np.arange(0.05, 1, 0.1)
            spectral_functions = []
            for s in range(-1, len(selection), 1):
                if s == -1:
                    ne = self.ne
                    weight = 1.0
                    out_lambda, out_I = get_spectral_fit_salpeter(ne, use_range)
                    spectral_functions.append(out_I*weight)
                else:
                    weight = selection[s]
                    ne = nes[np.where(ne_array >= weight)[0][0]]
                    out_lambda, out_I = get_spectral_fit_salpeter(ne, use_range)
                    spectral_functions.append(out_I*weight)
                    ne = nes[np.where(ne_array >= weight)[0][-1]]
                    out_lambda, out_I = get_spectral_fit_salpeter(ne, use_range)
                    spectral_functions.append(out_I*weight)

            spectral_functions = np.array(spectral_functions)
            max_Is = np.nanmax(spectral_functions, axis=1)
            max_lambdas = out_lambda[np.nanargmax(spectral_functions, axis=1)]
            central_lambda = max_lambdas[0]

            order = np.argsort(max_lambdas)
            max_lambdas = max_lambdas[order]
            max_Is = max_Is[order]
            popt, pcov = curve_fit(density_gaussian, max_lambdas*1e9, max_Is, p0=[central_lambda*1e9, 5], maxfev=20000)
            out_I = density_gaussian(out_lambda*1e9, *popt)

            # plt.figure()
            # for f in range(len(spectral_functions)):
            #     plt.plot(out_lambda*1e9, spectral_functions[f])
            # plt.plot(max_lambdas*1e9, max_Is, 'k-', label='Max Intensity')
            # plt.plot(out_lambda*1e9, out_I, 'r--', label='Gaussian Fit')
            # plt.xlabel('Wavelength (nm)')
            # plt.ylabel('Normalized Intensity')
            # plt.title('Spectral Functions for Different Electron Densities')
            # plt.show()

        elif fit_type == 'bohm-gross':
            nes = np.logspace(18, 21, 1000) * 1e6  # m^-3
            ne_array = density_gaussian(nes, self.ne, self.ne_range)
            if use_range:
                angddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'angles')
                angs = np.deg2rad(np.load(angddir + '/angOMEGA.npy'))
                fangs = np.load(angddir + '/fangOMEGA.npy')
            
                thetas = self.theta + angs
                sa_fractions = fangs
                spectral_functions = []
                for s in range(len(thetas)):
                    theta = thetas[s]
                    out_lambda, out_I = get_spectral_fit_bohm_gross(theta)
                    spectral_functions.append(out_I*sa_fractions[s])
                spectral_functions = np.asarray(spectral_functions)
                out_I = np.sum(spectral_functions, axis=0)
                out_I = out_I/np.nanmax(out_I)
            else:
                out_lambda, out_I = get_spectral_fit_bohm_gross(self.theta)

        return out_lambda, out_I

    def kangsm(self):
        ## This is an approximation for the spectral broadening, from C. Bruulsema (2022)
        ## One form of this approximation is given in Farmer et al., Phys. Plasmas 28, 032707 (2021)
        
        from scipy.interpolate import interp1d
        sa = np.rad2deg(self.theta)
        omg = self.omg-self.omgL
        spec = self.salpeter()

        angddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'angles')

        angs = np.load(angddir + '/angOMEGA.npy')
        fangs = np.load(angddir + '/fangOMEGA.npy')
        valang=np.zeros(len(spec))

        Is = np.zeros((len(angs), len(spec)))

        ## For each angle offset, modify the spectrum by shifting and scaling according to the angle,
        ## then weight by the beam intensity profile
        for i in range(len(angs)):
            intmult=1.0*(angs[i])/sa   #0.8

            ## This is the form Colin provided
            intmult=-1.0*intmult*(1.0-1.1*(intmult+0.138))

            ## This is the form in Phys. Plasmas 28, 032707 (2021)
            # intmult=-1.0*intmult*(0.8482-1.1*(intmult))
            
            intamp=1.0+1.2*intmult
            omgn=omg*(1.0+intmult)
            fva = interp1d(omg, spec, fill_value='extrapolate')     
            vali=fva(omgn)*intamp
            valang=valang+vali*fangs[i]

        return valang

    def instrument_broadening(self, spectrum):
        """
        Calculates the Thomson scattered power spectrum for an optical Thomson scattering (OTS) diagnostic.

        This function takes the input laser power, scattering length, spectral resolution, frequency arrays,
        laser frequency, electron density, and a precomputed spectral shape, and returns the power spectrum
        after applying instrumental broadening.

        Parameters
        ----------
        omg_resolution : float
            Frequency broadening (s^-1).
        omg : np.ndarray
            Array of frequency values at which the spectrum is evaluated (rad/s).
        omgL : float
            Probe laser frequency (rad/s).
        spectrum : np.ndarray
            Precomputed spectral form factor.

        Returns
        -------
        np.ndarray
            The power spectrum after applying instrumental broadening.
        """

        omg_resolution = (self.wavelength_fwhm / self.wavelength) * self.omgL/(2*np.pi)  # frequency resolution 1/s
        domg = np.abs(np.mean(np.diff(self.omg)))

        inst_broadened = gaussian_filter(spectrum, omg_resolution/domg)
        # st = spectral_resolution*(1+2*(self.omg-self.omgL)/self.omgL)
        self.broadened_output = inst_broadened

        return self.broadened_output

    def param_conversions(self, params, names):
        
        for i, name in enumerate(names):
            lname = name.upper()
            if lname in ['TE', 'ELECTRON_TEMP', 'ELECTRON_TEMPERATURE']:
                self.Te = calculations.eV_to_K(params[i]) # K
            elif lname in ['TI', 'ION_TEMP', 'ION_TEMPERATURE']:
                self.Ti = calculations.eV_to_K(params[i]) # K
            elif lname in ['NE', 'ELECTRON_DENSITY', 'E_DENSITY']:
                self.ne = params[i] * 1e6 # m^-3
            elif lname in ['E_CURRENT', 'E_CUR', 'ELECTRON_CURRENT']:
                self.e_cur = params[i] # km/s
            elif lname in ['FLOW']:
                self.flow = params[i] # km/s
            elif lname in ['VELOCITY_GRADIENT', 'V_GRAD']:
                self.v_grad = params[i] # km/s
            elif lname in ['NE_RANGE', 'ELECTRON_DENSITY_RANGE', 'NE_GRADIENT']:
                self.ne_range = params[i] * 1e6 # m^-3
            else:
                print(f"Error: Unrecognized parameter name '{name}'.")
                sys.exit()
            
        k = calculations.scattering_wavevector(self.wavelength, self.theta, self.ne)
        if hasattr(self, 'e_cur'):
            self.e_cur = calculations.kms_to_nm(self.e_cur, self.wavelength, k)*1e-9 # m
        if hasattr(self, 'flow'):
            self.flow = calculations.kms_to_nm(self.flow, self.wavelength, k)*1e-9 # m
        if hasattr(self, 'v_grad'):
            self.v_grad = calculations.kms_to_nm(self.v_grad, self.wavelength, k)*1e-9 # m

    def add_velocity_gradient(self, Fit_lambda, Fit_I):
        ## ADD ARCHIES PAPER

        flows = np.linspace(self.flow - 3 * self.v_grad, self.flow + 3 * self.v_grad, 300)
        spec_shift = flows - self.flow
        spec_shifts = np.repeat(spec_shift[:, np.newaxis], len(Fit_lambda), axis=-1).T
        Fit_wvlgths = np.repeat(Fit_lambda[:, np.newaxis], len(flows), axis=-1)
        Shifted_wvlgths = Fit_wvlgths - spec_shifts

        scalings = np.exp(-(3 * np.power(spec_shifts, 2)) / (np.power(self.v_grad, 2))) * np.abs(flows[1] - flows[0])
        scalings = scalings / np.nanmax(scalings)

        Fit_ys = np.repeat(Fit_I[:, np.newaxis], len(flows), axis=-1) * scalings
        New_fits = np.array(
            [np.interp(Fit_lambda, Shifted_wvlgths[:, i], Fit_ys[:, i]) for i in range(0, len(flows), 1)]).T
        Grad_IAW = np.nansum(New_fits, axis=-1)

        return Grad_IAW

    def zbar_from_eos(self, te, ne):
        ## Faster EOS lookup using pre-built RegularGridInterpolator per material.
        pts = self.init.pts
        vals_matrix = self.init.vals_matrix
        tgt = np.array([np.log10(te), np.log10(ne)])

        ## One call to griddata returns values for all materials at once (linear)
        z_interp = griddata(pts, vals_matrix, tgt, method='linear')

        ## If any entries are NaN (outside convex hull) fallback to nearest for those entries only.
        if z_interp is None or np.any(np.isnan(z_interp)):
            print(f"Warning {round(calculations.K_to_eV(te), 2)}, {round(ne*1e-6, 2)}: Some Zbar interpolation points outside EOS grid convex hull; using nearest-neighbor fallback.")
            z_nearest = griddata(pts, vals_matrix, tgt, method='nearest')
            if z_interp is None:
                z_interp = z_nearest
            else:
                mask = np.isnan(z_interp)
                z_interp = np.array(z_interp, dtype=float)
                z_interp[mask] = z_nearest[mask]

        z_vals = np.atleast_1d(z_interp).astype(float)[0]
        self.ionizations = z_vals
        return

    def zbar_from_eos_save(self, te, ne):
        ## Updating the last ionization state based on Zbar from EOS table
        from scipy.interpolate import griddata
        Te_grid = self.init.Te_grid
        Ne_grid = self.init.Ne_grid
 
        pts = np.column_stack((np.log10(Te_grid.ravel()), np.log10(Ne_grid.ravel())))
        tgt = np.array([np.log10(te), np.log10(ne)])
        def grid_interp(Z):
            vals = Z.ravel()

            z_interp = griddata(pts, vals, tgt, method='linear')
            if z_interp is None or (hasattr(z_interp, 'size') and np.isnan(z_interp).all()) or np.isnan(z_interp):
                # fallback to nearest if linear returns NaN (outside convex hull)
                z_interp = griddata(pts, vals, tgt, method='nearest')
            z_value = z_interp[0]
            return z_value

        for i, material in enumerate(self.materials):
            Z_grid = self.init.Z_element_grids[material]
            z_value = grid_interp(Z_grid)
            self.ionizations[i] = z_value
        # print(self.ionizations)
        return

    def run_fitting(self, params, names):
        self.param_conversions(params, names)
        if self.use_zbar_eos:
            self.zbar_from_eos(max(self.Te, self.Ti), self.ne)
            # self.zbar_from_eos_save(self.Te, self.ne)
        if hasattr(self, 'ne_range'):
            out_lambda, out_I = self.density_range(use_range=self.use_range, fit_type=self.fit_type)
        else:
            if self.use_range:
                out_lambda, out_I = self.salpeter_range()
            else:
                out_lambda, out_I = self.salpeter()
        if hasattr(self, 'v_grad'):
            if self.v_grad >= 1e-20:
                out_I = self.add_velocity_gradient(out_lambda, out_I)
        out_I = self.instrument_broadening(out_I)
        return out_lambda, out_I/np.nanmax(out_I)
 
class calculations():

    @staticmethod

    def dawson(w):
        return dawsn(w)

    def bohm_gross_wavelength(wavelength, theta, Te, ne):
        # wavelength in m, theta in radians, Te in K, ne in m^-3
        Scattering_vector = calculations.scattering_wavevector(wavelength, theta, ne)

        Omega_p2 = calculations.plasma_frequency(ne)**2
        Omega_frac = Omega_p2/(np.power(cst.c, 2)*np.power(2*cst.pi/wavelength,2))
        a = cst.Boltzmann*Te/cst.m_e

        Omega_epw2 = Omega_p2 + 3*a*np.power(Scattering_vector,2) + np.power((cst.hbar*np.power(Scattering_vector, 2))/(2*cst.m_e), 2)
        Omega_epw = np.sqrt(Omega_epw2)
        Lambda_epw = 1/(Omega_epw/(2*cst.pi*cst.c) + 1/wavelength)

        return Lambda_epw

    def critical_density(wavelength):
        # wavelength in m
        w = 2*cst.pi*cst.c/wavelength
        return np.power(w, 2)*cst.epsilon_0*cst.m_e/np.power(cst.e, 2) # m^-3

    def plasma_frequency(n, m=cst.m_e):
        # n in m^-3
        # m in kg
        return np.sqrt(np.power(cst.e, 2)*n/(cst.epsilon_0*m)) # rad s^-1

    def dimensionless_chemical_potential(ne, T):
        # density in m^-3, T in K
        beta = 1 / (cst.Boltzmann * T)
        de_Broglie = cst.hbar*np.sqrt(2*cst.pi*beta/cst.m_e)
        degeneracy = ne * (de_Broglie**3) / (2*0.5+1)
        return calculations.inverse_fermi_integral(1/2, degeneracy)

    def inverse_screening_length(ne, T):
        # ne in m^-3, T in K
        beta = 1/(cst.Boltzmann*T)
        de_Broglie = cst.hbar*np.sqrt(2*cst.pi*beta/cst.m_e)
        degeneracy = ne * (de_Broglie**3) / (2*0.5+1)
        return np.sqrt(((cst.e**2)*ne*beta*calculations.fermi_integral(-1/2, calculations.dimensionless_chemical_potential(ne, T)))/(cst.epsilon_0*degeneracy))

    def scattering_wavevector(wavelength, theta, n):
        # wavelength in m, theta in radians, n in m^-3
        wpe = calculations.plasma_frequency(n)
        w = calculations.wavelength_to_omega(wavelength)
        theta = theta/2
        correction = np.sqrt(1-np.power(wpe, 2)/np.power(w, 2))
        return correction*4*cst.pi*np.sin(theta)/(wavelength)

    def scattering_parameter(ne, T, wavelength, theta):
        # ne in m^-3, T in K, wavelength in m, theta in radians
        k = calculations.scattering_wavevector(wavelength, theta, ne)
        kappa_e = calculations.inverse_screening_length(ne, T)
        return kappa_e/k

    def thermal_velocity(T, m=cst.m_e, factor=1):
        # T in K, m in kg
        return np.sqrt(factor*cst.Boltzmann*T/m) # m/s

    def fermi_energy(ne):
        # ne in m^-3
        kf = np.power(3*(cst.pi**2)*ne, 1/3)
        return ((cst.hbar**2)*(kf**2))/(2*cst.m_e)

    def degeneracy(ne, T):
        # ne in m^-3, T in K
        beta = 1/(cst.Boltzmann*T)
        return 1/(calculations.fermi_energy(ne)*beta)

    def fermi_integral(o, x):
        return fdint.fdk(o, x)/gamma(o+1)

    def inverse_fermi_integral(o, x):
        u = x * gamma(o+1)
        return fdint.ifdk(o, u)
    
    def eV_to_K(T):
        # T in eV
        return T*(cst.physical_constants['electron volt-kelvin relationship'][0])
    def K_to_eV(T):
        # T in K
        return T/(cst.physical_constants['electron volt-kelvin relationship'][0])
    def wavelength_to_energy(wavelength):
        # wavelength in m
        return cst.h*cst.c/wavelength # J
    def energy_to_wavelength(E):
        # E in J
        return cst.h*cst.c/E # m
    def J_to_eV(E):
        # E in J
        return E/cst.e # eV
    def eV_to_J(E):
        # E in eV
        return E*cst.e # J
    def wavelength_to_omega(wavelength):
        # wavelength in m
        return 2*cst.pi*cst.c/wavelength # rad/s
    def omega_to_wavelength(omg):
        # omg in rad/s
        return 2*cst.pi*cst.c/omg # m
    def nm_to_kms(SHIFT, wavelength, scattering_vector):
        # SHIFT in nm, wavelength in m, scattering_vector in m^-1
        dw = SHIFT*1e-9*2*cst.pi*cst.c/np.power(wavelength, 2)
        return 1e-3*dw/scattering_vector # km/s
    def kms_to_nm(v, wavelength, scattering_vector):
        # v in km/s, wavelength in m, scattering_vector in m^-1
        dw = 1e3*v*scattering_vector
        return dw*np.power(wavelength, 2)/(2*cst.pi*cst.c)*1e9 # nm


# class Initialization:

#     def __init__(self):
#         self.scattering_params()
#         self.material_params()
#         self.OTS_init = self.input()
#         self.OTS_model = OTS(self.OTS_init)

#     def scattering_params(self):
#         # Wavelength in nm, Angle in degrees

#         self.wavelength_min = 430  # nm
#         self.wavelength_max = 455  # nm
#         self.wavelength_step = 0.05  # nm

#         self.wavelength_fwhm = 3.208  # nm
#         self.wavelength = 1053/2  # nm
#         self.theta = 59.9  # degrees
#         self.wavelengths = np.arange(self.wavelength_min, self.wavelength_max, self.wavelength_step)  # nm

#     def material_params(self):
#         # Material parameters
#         self.elements = ['C', 'H', 'Cl']
#         self.fractions = [0.499, 0.438, 0.063]  # Relative fractions of each element
#         self.zs = [6, 1, 17]  # Ionisation states
#         self.eos_file = '/Users/hpoole/Documents/Simulations/PROPACEOS/C49_9H43_8Cl6_3/grid_data.npz'

#         ## Fixed parameters
#         self.e_cur = 0 # km/s
#         self.flow = 0 # km/s

#     def input(self):
#         OTS_init = initialize()
#         OTS_init.scattering_params(self.wavelength, self.theta, self.wavelength_fwhm, self.wavelength_min, self.wavelength_max, self.wavelength_step)
#         OTS_init.material_params(self.elements, self.fractions, eos_file=self.eos_file)
#         OTS_init.plasma_params(ionizations=self.zs, e_cur=self.e_cur, flow=self.flow)
#         return OTS_init

    
# ME_INIT = Initialization()
# OTS_MODEL = OTS(ME_INIT.OTS_init)

# param_names = ['TE', 'TI', 'NE', 'NE_RANGE']
# param_units = ['eV', 'eV', '1/cc', '1/cc']

# param_inits = [358, 85, 1.4e20, 1.7e19]

# # import time
# # start_time = time.time()
# # # fit_x1, fit_y1 = OTS_MODEL.run_fitting(param_inits, param_names, use_zbar_eos=True, fit_type='salpeter', use_range=True)
# # fit_x2, fit_y2 = OTS_MODEL.run_fitting(param_inits, param_names, use_zbar_eos=True, fit_type='salpeter', use_range=False)

# # end_time = time.time()
# # t1 = end_time - start_time
# # print(f"Fitting with Salpeter took {t1} seconds")
# # start_time = time.time()
# # # fit_x3, fit_y3 = OTS_MODEL.run_fitting(param_inits, param_names, use_zbar_eos=True, fit_type='bohm-gross', use_range=True)
# # fit_x4, fit_y4 = OTS_MODEL.run_fitting(param_inits, param_names, use_zbar_eos=True, fit_type='bohm-gross', use_range=False)


# # end_time = time.time()
# # t2 = end_time - start_time
# # print(f"Fitting with Bohm-Gross took {t2} seconds")
# # # # print(f"Zbar EOS fitting is x{t1/t2} times slower than without")

# # plt.figure()
# # # plt.plot(fit_x1*1e9, fit_y1, label='Salpeter Range', color='blue')
# # plt.plot(fit_x2*1e9, fit_y2, label='Salpeter', color='orange')
# # # plt.plot(fit_x3*1e9, fit_y3, label='Bohm-Gross Range', color='green')
# # plt.plot(fit_x4*1e9, fit_y4, label='Bohm-Gross', color='red')
# # plt.xlabel('Wavelength (nm)')
# # plt.ylabel('Intensity')
# # plt.legend()
# # plt.show()