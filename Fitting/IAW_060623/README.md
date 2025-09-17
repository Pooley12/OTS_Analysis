## MCMC fitting for ion-acoustic wave (IAW) scattering

The main python script to run is setup.py.
This defines the raw data and the various fitting functions that can be used.
E.g. it can run just an initial test fit, an optimized fitting function (CMAES) or the MCMC function

It calls in run_IAW.py which runs the optical Thomson scattering (OTS) code to produce an IAW fit based on the laser and plasma parameters.
run_IAW.py also defines the cost function used to fit the produced IAW signal to the raw data.
The OTS code (ots.x) produces an IAW form function based on the parameters defined in the input deck, input.dat.
The code will use the input deck in the Active Directory. I've set it up to use input_CHCl.dat as it's base deck.

The likelihood function used for the MCMC function is defined in mcmc_iaw.py.

The Libraries packages contains information about MCMC normalization and the instrument function broadening (called in by run_IAW.py).

The OTS code (ots.x) may need to be compiled before running on another computer...
Before running please update the ots.x executable location in run_IAW.py line 137 [under Run_OTS.run()] eg:
def run(self):
    os.system('/Users/hpoole/Documents/Coding/OTS_code/ots.x > ots_output.txt')
