import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
elements = ['C', 'H', 'Cl']
fractions = [0.499, 0.428, 0.063]  # Relative fractions of each element
zs = [6, 1, 17]  # Ionisation states
# print(np.sum([f*z for f, z in zip(fractions, zs)]))  # Average Zbar if fully ionized
# sys.exit()

propaceos_file = '/Users/hpoole/Documents/Simulations/PROPACEOS/C49_9H43_8Cl6_3/C49_9H43_8Cl6_3.prp'



def read_propaceos(filename):

    def get_mesh(lines, j):
        while j < len(lines):
            s = lines[j].strip()
            if s == '' or any(c.isalpha() for c in s):
                j += 1
                continue
            try:
                array_num = int(s.split()[0])
                j += 1
                break
            except ValueError:
                j += 1
        else:
            raise ValueError("Could not find mesh grid count after mesh header")
    
        array = []
        while len(array) < array_num and j < len(lines):
            for tok in lines[j].strip().split():
                try:
                    array.append(float(tok))
                except ValueError:
                    pass
            j += 1
        if len(array) != array_num:
            raise ValueError(f"Expected {array_num} mesh grid points, got {len(array)}")
        return array_num, np.asarray(array), j

    def get_element_ionizations(lines, elements, j):
        ionizations = {}
        j_orig = j
        for element in elements:
            j = j_orig
            while j < len(lines):
                if f'   {element.lower()}   ' in lines[j].lower():
                    j += 1
                    break
                j += 1
            else:
                raise ValueError(f"Could not find {element} header")

            z = []
            while j < len(lines):
                if '*' in lines[j]:
                    break
                for tok in lines[j].strip().split():
                    try:
                        z.append(float(tok))
                    except ValueError:
                        pass
                j += 1
            ionizations[element] = np.asarray(z)
        return ionizations, j

    def get_zbar(lines, j):
        while j < len(lines):
            if '  zbar  ' in lines[j].lower():
                j += 1
                break
            j += 1
        else:
            raise ValueError("Could not find Zbar header")  
        zbar = []
        while j < len(lines):
            if '*' in lines[j]:
                break
            for tok in lines[j].strip().split():
                try:
                    zbar.append(float(tok))
                except ValueError:
                    pass
            j += 1
        return np.asarray(zbar), j

    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

        mesh_idx = None
        for i, line in enumerate(lines):
            if 'mesh parameters for EoS' in line:
                mesh_idx = i
                break    
        if mesh_idx is None:
            raise ValueError(f"Could not find 'mesh parameters for EoS' in {filename}")
        mesh_line = lines[mesh_idx].strip()

        j = mesh_idx + 1
        temp_num, temperatures, j = get_mesh(lines, j)
        dens_num, ion_densities, j = get_mesh(lines, j)
        data['nT'] = temp_num
        data['nD'] = dens_num
        data['temperatures'] = np.array(temperatures)
        data['ion_densities'] = np.array(ion_densities)

        element_ionizations, j = get_element_ionizations(lines, elements, j)
        zbar, j = get_zbar(lines, j)
        data['ionizations'] = element_ionizations
        data['zbar'] = zbar

        zbar = zbar.reshape((dens_num, temp_num))  # rows = ion_densities, cols = temperatures

        zH = element_ionizations['H']
        zC = element_ionizations['C']
        zCl = element_ionizations['Cl']

        def collapse_z(Z, F):
            Z = Z.reshape((dens_num, temp_num, F+1))
            ions = np.arange(F+1, dtype=int).reshape(1, 1, F+1)
            ions = np.tile(ions, (dens_num, temp_num, 1))  # shape -> (dens_num, temp_num, #ions)
            z_avg = np.sum(Z*ions, axis=-1)
            return np.asarray(z_avg)
        
        zH = collapse_z(element_ionizations['H'], 1)
        zC = collapse_z(element_ionizations['C'], 6)
        zCl = collapse_z(element_ionizations['Cl'], 17)
        z_test = fractions[0]*zC + fractions[1]*zH + fractions[2]*zCl

        # create meshgrid: Te (x), Ni (y). With indexing='xy' shapes will match zbar
        Te, Ni = np.meshgrid(temperatures, ion_densities, indexing='xy')
        Ne = zbar * Ni

        ## contour plot
        # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
        # cf = axs[0].contourf(Te, Ni, zbar, levels=50, cmap='viridis')
        # fig.colorbar(cf, ax=axs[0], label='Zbar')
        # axs[0].set_xlabel('Temperature (eV)')
        # axs[0].set_ylabel('Ion Density (cm$^{-3}$)')
        # axs[0].set_title('Zbar from PROPACEOS Data')

        # levels = np.power(10, np.arange(15, np.max(np.log10(Ne)), 0.5))
        # cf = axs[1].contourf(Te, Ni, Ne, levels=levels, norm=LogNorm(), cmap='viridis')
        # fig.colorbar(cf, ax=axs[1], label='Electron Density (cm$^{-3}$)')
        # axs[1].set_xlabel('Temperature (eV)')
        # axs[1].set_ylabel('Ion Density (cm$^{-3}$)')
        # axs[1].set_title('Ne from PROPACEOS Data')
        # for ax in axs:
        #     ax.set_xscale('log')
        #     ax.set_yscale('log')
        # plt.tight_layout()
        # plt.show()
        


        Te, Ni = np.meshgrid(temperatures, ion_densities, indexing='xy')
        Ne = zbar * Ni
        Te_grid, Ne_grid = Te, Ne
        ne, te = 1e20, 80  # Example values

        grid_data = {
            'Ni_grid': Ni,
            'Te_grid': Te_grid,
            'Ne_grid': Ne_grid,
            'Zbar': zbar,
            'ZC': zC,
            'ZH': zH,
            'ZCl': zCl
        }
        outfn = '/Users/hpoole/Documents/Simulations/PROPACEOS/C49_9H43_8Cl6_3/grid_data.npz'
        np.savez_compressed(outfn, Te=Te_grid, Ne=Ne_grid, Ni=Ni, Zbar=zbar, ZC=zC, ZH=zH, ZCl=zCl)
        # print(f"Saved grid data to {outfn}")


        # Interpolate zbar at (te, ne) in log10-space (more stable over wide dynamic ranges)

        pts = np.column_stack((np.log10(Te_grid.ravel()), np.log10(Ne_grid.ravel())))
        vals = zbar.ravel()
        tgt = np.array([np.log10(te), np.log10(ne)])

        zbar_interp = griddata(pts, vals, tgt, method='linear')
        if zbar_interp is None or (hasattr(zbar_interp, 'size') and np.isnan(zbar_interp).all()) or np.isnan(zbar_interp):
            # fallback to nearest if linear returns NaN (outside convex hull)
            zbar_interp = griddata(pts, vals, tgt, method='nearest')

        zbar_value = float(zbar_interp)
        print(f"Interpolated Zbar at Te={te} eV, Ne={ne} cm^-3 -> Zbar ≈ {zbar_value:.6g}")




        # Find nearest grid indices to (te, ne) in log-space (good for wide dynamic ranges)
        if te <= 0 or ne <= 0:
            raise ValueError("te and ne must be positive for log-scale nearest search")

        with np.errstate(divide='ignore', invalid='ignore'):
            log_Te = np.log10(Te_grid)
            log_Ne = np.log10(Ne_grid)

        target = np.array([np.log10(te), np.log10(ne)])
        d2 = (log_Te - target[0])**2 + (log_Ne - target[1])**2
        d2 = np.where(np.isfinite(d2), d2, np.inf)  # treat invalid entries as far away

        ne_idx, te_idx = np.unravel_index(np.argmin(d2), d2.shape)

        print(f"Nearest grid indices -> ne_idx: {ne_idx}, te_idx: {te_idx}")
        print(f"Nearest grid values -> Te: {Te_grid[ne_idx, te_idx]}, Ne: {Ne_grid[ne_idx, te_idx]}, Zbar: {zbar[ne_idx, te_idx]}")


        sys.exit()
        te_idx = (np.abs(temperatures - te)).argmin()
        ni_idx = (np.abs(ion_densities - ne / zbar[0, te_idx])).argmin()
        zbar_value = zbar[ni_idx, te_idx]
        print(f'At Te={te} eV and Ne={ne} cm^-3, Zbar={zbar_value}, Ni={ion_densities[ni_idx]} cm^-3')

        # electron_densities = zbar*ion_densities
        # data['electron_densities'] = electron_densities
        sys.exit()
        return data


def read_propaceos_test(filename):

    def get_mesh(lines, j):
        while j < len(lines):
            s = lines[j].strip()
            if s == '' or any(c.isalpha() for c in s):
                j += 1
                continue
            try:
                array_num = int(s.split()[0])
                j += 1
                break
            except ValueError:
                j += 1
        else:
            raise ValueError("Could not find mesh grid count after mesh header")
    
        array = []
        while len(array) < array_num and j < len(lines):
            for tok in lines[j].strip().split():
                try:
                    array.append(float(tok))
                except ValueError:
                    pass
            j += 1
        if len(array) != array_num:
            raise ValueError(f"Expected {array_num} mesh grid points, got {len(array)}")
        return array_num, np.asarray(array), j

    def get_element_ionizations(lines, elements, j):
        ionizations = {}
        j_orig = j
        for element in elements:
            j = j_orig
            while j < len(lines):
                if f'   {element.lower()}   ' in lines[j].lower():
                    j += 1
                    break
                j += 1
            else:
                raise ValueError(f"Could not find {element} header")

            z = []
            while j < len(lines):
                if '*' in lines[j]:
                    break
                for tok in lines[j].strip().split():
                    try:
                        z.append(float(tok))
                    except ValueError:
                        pass
                j += 1
            ionizations[element] = np.asarray(z)
        return ionizations, j

    def get_zbar(lines, j):
        while j < len(lines):
            if '  zbar  ' in lines[j].lower():
                j += 1
                break
            j += 1
        else:
            raise ValueError("Could not find Zbar header")  
        zbar = []
        while j < len(lines):
            if '*' in lines[j]:
                break
            for tok in lines[j].strip().split():
                try:
                    zbar.append(float(tok))
                except ValueError:
                    pass
            j += 1
        return np.asarray(zbar), j

    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        # print(lines[:50])  # Print first 50 lines for inspection
        # sys.exit()
        mesh_idx = None
        for i, line in enumerate(lines):
            if 'mesh parameters for EoS' in line:
                mesh_idx = i
                break    
        if mesh_idx is None:
            raise ValueError(f"Could not find 'mesh parameters for EoS' in {filename}")
        mesh_line = lines[mesh_idx].strip()

        j = mesh_idx + 1
        temp_num, temperatures, j = get_mesh(lines, j)
        dens_num, ion_densities, j = get_mesh(lines, j)
        data['nT'] = temp_num
        data['nD'] = dens_num
        data['temperatures'] = np.array(temperatures)
        data['ion_densities'] = np.array(ion_densities)

        element_ionizations, j = get_element_ionizations(lines, ['C'], j)
        zbar, j = get_zbar(lines, j)
        data['ionizations'] = element_ionizations
        data['zbar'] = zbar
        zbar = np.asarray(zbar)
        if zbar.size != dens_num * temp_num:
            raise ValueError(f"zbar size {zbar.size} doesn't match dens*temp {dens_num * temp_num}")
        zbar = zbar.reshape((dens_num, temp_num))  # rows = ion_densities, cols = temperatures

        print(zbar)

        zC = element_ionizations['C']
        zC = zC.reshape((dens_num, temp_num, 7))
        ions = np.arange(7, dtype=int).reshape(1, 1, 7)
        ions = np.tile(ions, (dens_num, temp_num, 1))  # shape -> (dens_num, temp_num, 7)
        zC = np.sum(zC*ions, axis=-1)



        print(zC[3, 4], zbar[3, 4])

        sys.exit()
        return data



read_propaceos(propaceos_file)
# read_propaceos_test('/Users/hpoole/FLASH/EOS/C_test/C_test.prp')