# ====================================================================== 
# Import Statements
# ====================================================================== 
from numpythia import Pythia, hepmc_write, hepmc_read
from numpythia import STATUS, HAS_END_VERTEX, ABS_PDG_ID
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyjet import cluster
from matplotlib.colors import ListedColormap
import pickle as pic
import os.path
# -----------------------------------------------------------------------
# Initalize
# -----------------------------------------------------------------------
pid                   = 'pdgid'
num_events            = 1000 # Number of events to process per parent
test                  = 100  # particle. Number of test events to reserve
discarded_data        = []   # Archive of any particles discarded
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
def is_massless_or_isolated(jet):
    # Returns true if a jet is only constituated of one particle
    # (nconsts == 1) and has a pdgid equal to that
    # of a photon or a gluon
    if len(jet.constituents_array()) == 1: 
        if np.abs(jet.info[pid]) == 21 or np.abs(jet.info[pid]) == 22:
            return True
        # if a muon is outside of the radius of the jet, discard it
        if np.abs(jet.info[pid]) == 13 and 2*jet.mass/jet.pt > 1.0:
            return True
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
    eta         = [] # Eta and phi are coordinates, similar to cylindrical
    phi         = [] # coordinates. But is more like the coordinates are  
    m           = [] # Wrapepd around the inside of the cyilnder. Phi is
    pt          = [] # the azimuthal. Eta is the psuedo-rapidity. m is 
    has_eta_phi = jet.constituents_array() # the mass. and pt is the 
    for i in range(len(has_eta_phi)):      # Transverse momentum (momentum  
        pt.append(has_eta_phi[i][0])       # in the direction of travel.
        eta.append(has_eta_phi[i][1])
        phi.append(has_eta_phi[i][2])
        m.append(has_eta_phi[i][3])
    m = np.array(m)
    pt = np.array(pt)
    e = (pt**2 + m**2)**0.5 # This is the transverse energy
    return [eta, phi, e]
def pythia_sim(cmd_file, part_name=""):
    # The main simulation. Takes a cmd_file as input. part_name 
    # is the name of the particle we're simulating decays from.
    # Only necessary for titling graphs.
    # Returns an array of 2D histograms, mapping eta, phi, with transverse
    # energy.
    pythia = Pythia(cmd_file, random_state=1)
    selection = ((STATUS == 1) & ~HAS_END_VERTEX)
    unclustered_particles = []
    debug_data        = [] # Deprecated but costs 1 operation per function call so negligble
    discarded_data    = [] # For analyzing what gets thrown out from function is_massless_or_isolated
    sj_data_per_event = [] # sub jet data indexed into each event

    for event in pythia(events=num_events):
        lead_jet_invalid  = False
        sub_jet_data      = [] # There are multiple jets in each event
        vectors           = event.all(selection)
        sequence          = cluster(vectors, R=0.4, p=-1, ep=True) # Note to self: R might need to be changed to 1
        jets              = sequence.inclusive_jets()
        unclustered_particles.append(sequence.unclustered_particles())

        for i, jet in enumerate(jets):
            data = (
                    jet.mass, jet.eta, jet.phi, jet.pt,
                    len(jet.constituents_array()), 2*jet.mass/jet.pt
            )
            if data[5] > 0.4:
                debug_data.append((jet, data))
            if is_massless_or_isolated(jet):
                discarded_data.append(jet)
                if i == 0: lead_jet_invalid = True 
            if i < 3:
                sub_jet_data.append(data)
        if not lead_jet_invalid:
            sj_data_per_event.append(np.array(sub_jet_data))
    sj_data_per_event = np.array(sj_data_per_event)
    return sj_data_per_event

def shuffle_and_stich(A, B, X, Y):
    # A and B are both tensors which map to X and Y respectively
    # In this file, A will be ttbar, X will be the array of 1's
    # Y is the array of 0's. Need one input tensor, and one output
    # array to train model.
    if len(A) != len(B) != len(X) != len(Y):
        print("All tensors must have same length")
        return
    T_i = [] # Input Tensor for NN
    T_o = [] # Output Tensor for NN. Input maps to output.
    i_a = 0 # Iterator for ttbar mappings 
    i_b = 0 # Iterator for ZZ mappings.
    # Randomly select between tensor A or B to append their next
    # item. Unless one is empty then just shuffle in
    prob = 0.5
    while len(T_i) != (len(A) + len(B)):
        if np.random.rand() > prob and i_a < len(A):
            T_i.append(A[i_a]) # A[i_a] maps to X[i_a]
            T_o.append(X[i_a])
            i_a += 1
        elif i_b < len(B):
            T_i.append(B[i_b]) # Similarily, B[i_b] maps
            T_o.append(Y[i_b]) # to Y[i_b]
            i_b += 1
            if i_b == len(B):
                prob = 0 # Only add from Tensor A now, B is empty.
    T_i = np.array(T_i)
    T_o = np.array(T_o)
    return T_i, T_o

def ship(carepack):
    # carepack is a numpy multidim array
    # if there does not exist a care_package0 file, write the carepakcage
    # as that. otherwise keep incrementing 1 until there does not exist
    # such a file, then write it as such
    i = 0
    base_name = "care_package"
    if os.path.isfile(base_name+str(i)):
        i += 1
        while os.path.isfile(base_name+str(i)):
            i += 1
    np.save(open((base_name+str(i)), "wb"), carepack)  
# -----------------------------------------------------------------------
# Main process for generating tensor data
# -----------------------------------------------------------------------
while np.load(open("control", "rb")):
    # ttbar_tensor has indices of event, followed by eta, followed by phi.
    # The value of the h_tensor is the associated transverse energy.
    ttbar_data = pythia_sim('ttbar.cmnd', "TTbar")
    zz_data = pythia_sim('zz.cmnd', 'ZZ')
    
    ttbar_training = ttbar_data[:test] # Set the training data to be everything up to the test index
    ttbar_training_map = np.ones(test) # All of those are definitely ttbar so they get a value of 1
    ttbar_data = ttbar_data[test:]     # Let's reassign the data to be the rest of the values not in training
    
    zz_training = zz_data[:test]       # Do the same for zz_data
    zz_training_map = np.zeros(test)   # Except these get a value of 0
    zz_data = zz_data[test:] 
    
    ttbar_mapping = np.ones(num_events - test) # The NN will map ttbar information to the value it maps to
    zz_mapping = np.zeros(num_events - test)   # That value is 0 for zz and 1 for ttbar
    
    # T_i is the training data / the input tensor
    # T_o is the expected value (closer to 1 is ttbar, closer to 0 is zz
    T_i, T_o = shuffle_and_stich(ttbar_data, zz_data,
                                 ttbar_mapping, zz_mapping)
    Test_i, Test_o = shuffle_and_stich(ttbar_training, zz_training,
                                       ttbar_training_map, zz_training_map)
    care_package = np.array([T_i, T_o, Test_i, Test_o])
    ship(care_package)
print("Data Generation Halted")
