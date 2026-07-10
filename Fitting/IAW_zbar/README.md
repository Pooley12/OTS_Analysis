# IAW Thomson Scattering Analysis

Bayesian fitting code for Ion Acoustic Wave (IAW) spectra from Optical Thomson Scattering (OTS) diagnostics. Generates forward model spectra using the Salpeter formulation and fits them to experimental data via CMA-ES optimisation followed by MCMC posterior sampling.

Developed for OMEGA laser facility data, but the physics model is general.

---

## Package Structure

```
IAW_zbar/
├── run.py              # Main script: CMA-ES optimisation + MCMC fitting
├── mcmc.py             # PyTensor Op wrapping the log-likelihood for PyMC
├── processing.py       # Post-processing: convergence checks, corner plots, parameter tracking
├── get_ionization.py   # Extracts Zbar grid from PROPACEOS EOS files
└── ../Libraries/
    ├── OTS_function.py     # Core OTS physics (Salpeter formula, instrument broadening)
    ├── normalisation.py    # Parameter normalisation utilities
    ├── Atomic_data.txt     # Atomic mass and charge data
    └── angles/             # Angular weighting files for solid-angle integration
```

---

## Dependencies

```
pymc
pytensor
numpy
scipy
matplotlib
arviz
cma
pandas
seaborn
fdint
```

Install into a conda environment:

```bash
conda create -n myenv python=3.12
conda activate myenv
conda install -c conda-forge pymc arviz numpy scipy matplotlib pandas seaborn
pip install cma
pip install ../Libraries/fdint.zip  # patched version of fdint required for Python 3.12
```

> **macOS note (Xcode 15+):** Apple's new linker can break PyTensor's C compilation. If you see `library imit=] is not found` errors, add the following to the top of `mcmc.py` (already included):
> ```python
> import pytensor
> pytensor.config.cxx = "/usr/bin/clang++"
> ```
> Or set `PYTENSOR_FLAGS="cxx="` in your shell to disable C compilation entirely (slight performance penalty).

---

## Workflow

### 1. Prepare EOS Zbar grid (once per material)

If using EOS-based mean ionisation (Zbar) from a PROPACEOS table, first generate the interpolation grid:

```bash
python get_ionization.py
```

This reads the `.prp` file and saves a compressed `.npz` grid used by the forward model. Update the `propaceos_file` path and output path inside the script before running.

### 2. Configure and run fitting (`run.py`)

Edit the configuration blocks at the bottom of `run.py`:

**Data input:**
```python
Shot_day    = 'OMEGA_Jun2023'
Shot_number = 108615
User        = 'hpoole'        # used to build file paths
```

**Run flags:**
```python
Save_info  = False   # save outputs to disk
Run_CMAES  = True    # run CMA-ES optimisation first
Only_CMAES = False   # stop after CMA-ES (no MCMC)
Run_MCMC   = True    # run MCMC posterior sampling
Show_fits  = False   # overlay posterior draws on data
```

**Fit parameters** (edit the `Exploration` class):

| Parameter | Description | Units |
|---|---|---|
| `TE` | Electron temperature | eV |
| `TI` | Ion temperature | eV |
| `E_CURRENT` | Electron current (Doppler shift) | km/s |
| `FLOW` | Bulk flow velocity | km/s |
| `VELOCITY_GRADIENT` | Velocity gradient width | km/s |

Then run:

```bash
python run.py
```

The script loops over all time-resolved scattering strips found under `Scattering_strips/`. For each time step it:
1. Loads the raw data (wavelength, intensity, fractional error).
2. Optionally runs **CMA-ES** to find a best-fit starting point.
3. Runs **MCMC** (PyMC Metropolis sampler) to sample the posterior.
4. Saves the `idata.nc` ArviZ inference data object and per-parameter posterior arrays.

### 3. Post-process results (`processing.py`)

```bash
python processing.py
```

Loads saved MCMC outputs and produces:
- Corner/matrix plot of the joint posterior (`mcmc_matrix`)
- Time-resolved parameter tracking plots
- Convergence diagnostics: R-hat, ESS, autocorrelation, rank plots, pairplots (via ArviZ)

---

## Input Data Format

Raw scattering strip files are plain text with three columns:

```
wavelength(nm)   intensity   fractional_error
```

Files should be named `{time}ps.txt` and placed under:
```
.../Data/{shot_number}/IAW/Scattering_strips/
```

---

## Output Structure

```
.../Data/{shot_number}/IAW/Results/
├── CMAES/{time}ps/
│   ├── Plasma_parameters.txt   # Best-fit parameter values
│   ├── IAW.txt                 # Best-fit spectrum [lambda, intensity]
│   └── Best_fit.png
└── MCMC/{time}ps/
    ├── idata.nc                # ArviZ inference data (all chains/draws)
    ├── MCMC_parameter_bounds.csv
    ├── {PARAM}.txt             # Flattened posterior samples per parameter
    ├── idata.png
    └── Fits/                   # Optional: posterior draw spectra
```

---

## Physics Model

The forward model implements the **Salpeter (1960) form factor** for a multi-species plasma. Key features:

- Multi-species ion terms (arbitrary element mix with user-defined fractions)
- Electron current and bulk flow Doppler shifts
- Velocity gradient broadening (convolution over a Gaussian flow distribution)
- Solid-angle integration over the collection optic acceptance cone (`salpeter_range`)
- Gaussian instrument function broadening
- EOS-based mean ionisation (Zbar) lookup via `LinearNDInterpolator` on a PROPACEOS grid

The scattering parameter is:

$$\alpha = \frac{\kappa_e}{k} = \frac{1}{k \lambda_{De}}$$

where $k$ is the scattering wavevector and $\lambda_{De}$ is the electron Debye length.
