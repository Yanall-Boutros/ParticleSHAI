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
bjet_id               = 5
num_events            = 10 # Number of events to process per parent
test                  = 1  # particle. Number of test events to reserve
discarded_data        = [] # Archive of any particles discarded
titles = [
            "Histogram of bjet eta",
            "Histogram of bjet phi",
            "Histogram of bjet mass",
            "Histogram of bjet pt",
            "Histogram of nonbjet eta",
            "Histogram of nonbjet phi",
            "Histogram of nonbjet mass",
            "Histogram of nonbjet pt",
            "Histogram of number of b jets",
        ]
xlabels = [
            "Eta ($\eta$)",
            "Phi ($\phi$)",
            "Mass ($GeV$)",
            "Pt  ($jet.pt$)",
            "Eta ($\eta$)",
            "Phi ($\phi$)",
            "Mass ($GeV$)",
            "Pt  ($jet.pt$)",
            "Number of b jets"
        ]
ylabel = "Counts in Event"
n_bins = [
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100
        ]
class event_hists(object):
    # A data structure which contains the Eta, Phi, pt, and invariant
    # mass of the jet, maintaing those two sets for jets of 
    # pdgid = 5 AKA  'bjet', and non bjets.
    def __init__(self):
        self.hists = {}
        for i, title in enumerate(titles): self.hists[i] = []

    def update(self, pdgid, eta, mass, phi, pt):
        update_vals = [eta, mass, phi, pt]
        if pdgid == "bjet":
            for i, title in enumerate(titles[:4]): 
                self.hists[i].append(update_vals.pop(0))
        else:
            dev = 4
            for i, title in enumerate(titles[4:-1]):
                self.hists[i+dev].append(update_vals.pop(0))
        self.hists[len(titles) - 1].append(len(self.hists[0]))
        
    def save_1d_hists(self):
        out_list = []
        for i, title in enumerate(titles):
            plt.figure()
            plt.hist(self.hists[i], bins=n_bins[i])
            plt.title(titles[i])
            plt.xlabel(xlabels[i])
            plt.ylabel(ylabel)
            plt.savefig(titles[i]+".png")
            # out_list.append(plt.object) whatever code here
            plt.close()
                
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
def print_jet_infos(jet):
    for const in jet.constituents(): print(const.userinfo['pdgid'])
def print_jet_parents(jet):
    if jet.userinfo is None:
        print_jet_parents(jet.parents[1])
        print_jet_parents(jet.parents[0])
    else:
        print(jet.userinfo['pdgid'])
def print_nonetype_consts(jet):
    if len(jet.constituents_array()) == 1 and jet.userinfo is not None:
        print(jet.userinfo['pdgid'])
        return
    for const in jet.constituents():
        if const.userinfo is None:
            print_nonetype_consts(const)
            return
def is_bjet(jet):
    for const in jet.constituents():
        if const.userinfo is None:
            is_bjet(const)
        else:
            if const.userinfo['pdgid'] == 5:
                print(5)

def pythia_sim(cmd_file, part_name=""):
    # The main simulation. Takes a cmd_file as input. part_name 
    # is the name of the particle we're simulating decays from.
    # Only necessary for titling graphs.
    # Returns an array of 2D histograms, mapping eta, phi, with transverse
    # energy.
    pythia      = Pythia(cmd_file, random_state=1)
    selection   = ((STATUS == 1) & ~HAS_END_VERTEX)
    events_data = []
    for event in pythia(events=num_events):
        vectors              = event.all(selection)
        sequence             = cluster(vectors, R=0.4, p=-1, ep=True) #nts:Rval update
        jets                 = sequence.inclusive_jets()
#       unclustered_particles.append(sequence.unclustered_particles())
        event_data_package = event_hists()
        for jet in jets:
            event_data_package.update (
                    is_bjet(jet), jet.eta, jet.phi, jet.mass, jet.pt
            )
        events_data.append(event_data_package)
    return events_data

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

def structure_data_into_care_package(particle_data_list):
    # Eventually, this will be dynamic. For now, ttbar and zz are 
    # statically assigned
    ttbar_data = particle_data_list[0]
    zz_data    = particle_data_list[1]
    ttbar_training = ttbar_data[:test] # Set the training data to be everything
                                       # up to the test index
    ttbar_training_map = np.ones(test) # All of those are definitely ttbar so
                                       # they get a value of 1
    ttbar_data = ttbar_data[test:]     # Let's reassign the data to be the rest of
                                       # the values not in training
    zz_training = zz_data[:test]       # Do the same for zz_data
    zz_training_map = np.zeros(test)   # Except these get a value of 0
    zz_data = zz_data[test:] 
    # The NN will map ttbar information to the value it maps to
    ttbar_mapping = np.ones(num_events - test) 
    # That value is 0 for zz and 1 for ttbar
    zz_mapping = np.zeros(num_events - test)   
    
    # T_i is the training data / the input tensor
    # T_o is the expected value (closer to 1 is ttbar, closer to 0 is zz
    T_i, T_o = shuffle_and_stich(ttbar_data, zz_data,
                                 ttbar_mapping, zz_mapping)
    Test_i, Test_o = shuffle_and_stich(ttbar_training, zz_training,
                                       ttbar_training_map, zz_training_map)
    care_package = np.array([T_i, T_o, Test_i, Test_o], dtype=object)
    return care_package

def make_plots(data_pak):
    #Aggregate_pak = event_hists()
    for event_pak in data_pak:
        event_pak.save_1d_hists()
# -----------------------------------------------------------------------
# Main process 
# -----------------------------------------------------------------------
while np.load(open("control", "rb")):
    # ttbar_tensor has indices of event, followed by eta, followed by phi.
    # The value of the h_tensor is the associated transverse energy.
    higgs_ww_data    = pythia_sim('higgsww.cmnd', "higgsWW")
    make_plots(higgs_ww_data)
    higgs_zz_data    = pythia_sim('higgszz.cmnd', 'higgsZZ')
    make_plots(higgs_zz_data)
    care_package  = structure_data_into_care_package([ttbar_data, zz_data])
    ship(care_package)
print("Data Generation Halted")
