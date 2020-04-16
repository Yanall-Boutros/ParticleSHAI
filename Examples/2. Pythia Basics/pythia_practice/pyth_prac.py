# ----------------------------------------------------------------------
# Import Statements
# ----------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpythia import Pythia, hepmc_write, hepmc_read
from numpythia import STATUS, HAS_END_VERTEX, ABS_PDG_ID
from numpythia.testcmnd import get_cmnd
from numpy.testing import assert_array_equal
# ----------------------------------------------------------------------
# Constant Definitions
# ----------------------------------------------------------------------
inp        = [12, 14, 16]
pythia     = Pythia(get_cmnd('w'), random_state=1)
inv_mass   = []
num_events = 1
# ----------------------------------------------------------------------
# Function Definitions
# ----------------------------------------------------------------------
def gen_arrays(selection):
    # Shows how to generate the  events and save those to a plaintext
    # file or load the events in a plaintext file and 
    print("Selection = ", selection)
    for event in hepmc_write('events.hepmc', pythia(events=num_events)):
        array1 = event.all(selection)
    for event in hepmc_read('events.hepmc'):
        array2 = event.all(selection)
    print(array1 == array2)
    return np.array(array1), np.array(array2)

def set_particle_selection(params):
   # sets the filter for the selection variable based on an array of
   # string or int input vars where a string represents the particulate
   # name and the int represents the pdg_id 
   # convert to int if necessary
   pdgs = list()
   selection = ((STATUS == 1) & ~HAS_END_VERTEX)
   for pdg in params: selection = selection & (ABS_PDG_ID != int(pdg))
   return selection

def calc_mom(x, y, z):
   return x**2 + y**2 + z**2

def calc_mass(fvec):
   i_mass = fvec[0]**2
   p_sq = calc_mom(fvec[1], fvec[2], fvec[3])
   return ((i_mass + p_sq)**0.5)
# ----------------------------------------------------------------------
# Generate using pythia, calculate invariant mass, plot
# ----------------------------------------------------------------------
selection = set_particle_selection(inp)
a1, a2 = gen_arrays(selection)    # a1 == a2 since a1 is the array which 
for four_vec in a1:                   # writes the file a2 reads from.
    inv_mass.append(calc_mass(four_vec))
plt.hist(inv_mass, 30, range=[0, 100])
plt.title("Histogram of Invariant Mass")
plt.xlabel("Invariant Mass [GeV]")
plt.ylabel("Couts per Event")
plt.savefig("inv_mass_hist.png")
