# ----------------------------------------------------------------------
# Import Statements
# ----------------------------------------------------------------------
from numpythia import Pythia, hepmc_write, hepmc_read
from numpythia import STATUS, HAS_END_VERTEX, ABS_PDG_ID
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyjet import cluster
from matplotlib.colors import ListedColormap
import sys
if len(sys.argv) != 2:
    print("Error: No CMND file provided.")
    exit(1)
# -----------------------------------------------------------------------
# Initalize
# -----------------------------------------------------------------------
id                    = 'pdgid'
cmd_file              = sys.argv[1]
part_name             = cmd_file[:cmd_file.find('.')]
num_events            = 10
make_plots            = True
unclustered_particles = []
# -----------------------------------------------------------------------
# Generate Jets and Histogram Data
# -----------------------------------------------------------------------
def is_massless_or_isolated(jet):
    # Returns true if a jet has nconsts = 1 and has a pdgid equal to that
    # of a photon or a gluon
    if len(jet.constituents_array()) == 1: 
        if np.abs(jet.userinfo[id]) == 21 or np.abs(jet.userinfo[id]) == 22:
            return True
        # if a muon is outside of the radius of the jet, discard it
        if np.abs(jet.userinfo[id]) == 13:
            if 2*jet.mass/jet.pt > 0.4: return True
     # Remove Jets with too high an eta
    if np.abs(jet.eta) > 5.0:
        return True
    # Remove any jets less than an arbitrary near zero mass
    if jet.mass < 0.4:
        return True
    return False
def return_particle_data(jet):
    # return the array containing all the eta, phi, and energies of the
    # particles in a jets constituent array
    vals = []
    has_eta_phi = jet.constituents_array()
    eta = []
    phi = []
    m = []
    pt = []
    for i in range(len(has_eta_phi)):
        pt.append(has_eta_phi[i][0])
        eta.append(has_eta_phi[i][1])
        phi.append(has_eta_phi[i][2])
        m.append(has_eta_phi[i][3])
    m = np.array(m)
    pt = np.array(pt)
    e = (pt**2 + m**2)**0.5
    return [eta, phi, e]
def pythia_sim(cmd_file, part_name="", make_plots=False):
    # The main simulation. Takes a cmd_file as input. part_name 
    # is the name of the particle we're simulating decays from.
    # Only necessary for titling graphs.
    # Returns an array of 2D histograms, mapping eta, phi, with transverse
    # energy.
    a                     = 0
    unclustered_particles = []
    part_tensor           = []
    pythia                = Pythia(cmd_file, random_state=1)
    selection             = ((STATUS == 1) & ~HAS_END_VERTEX)
    for event in pythia(events=num_events):
        jets_particle_eta    = []
        jets_particle_phi    = []
        jets_particle_energy = []
        part_data = []
        vectors   = event.all(selection)
        sequence  = cluster(vectors, R=1.0, p=-1, ep=True)
        jets      = sequence.inclusive_jets()
        unclustered_particles.append(sequence.unclustered_particles())
        for i, jet in enumerate(jets):
            part_data = return_particle_data(jet)
            if is_massless_or_isolated(jet):
                discarded_data.append(jet)
            else:
                jets_particle_eta.extend(part_data[0])
                jets_particle_phi.extend(part_data[1])
                jets_particle_energy.extend(part_data[2])
        plt.figure()
        part_tensor.append(plt.hist2d(jets_particle_eta, jets_particle_phi,
                    weights=jets_particle_energy, density=True,
                    range=[(-5,5),(-1*np.pi,np.pi)],
                    bins=(20,32), cmap='binary')[0]) # We're only taking the zeroth element
        if not make_plots: plt.close() # which is the raw data of the 2D Histogram
        else:
            plt.xlabel("$\eta$")
            plt.ylabel("$\phi$")
            plt.title("Particles from "+part_name)
            cbar = plt.colorbar()
            cbar.set_label('Tranverse Energy of Each Particle ($GeV$)')
            plt.savefig("hists/Jets_Particles_"+part_name+str(a)+".png")
            plt.close()
        a += 1
    return np.array(part_tensor)
discarded_data = [] 
# -----------------------------------------------------------------------
# Main Loop for Storing Jet Data
# -----------------------------------------------------------------------
pythia_sim(cmd_file, part_name, make_plots)
