#!/usr/bin/python3
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
from skhep.math.vectors import *
import pickle as pic
import os.path
# -----------------------------------------------------------------------
# Initalize
# -----------------------------------------------------------------------
pid                   = 'pdgid'
bjet_id               = 5
num_events            = 1000 # Number of events to process per parent
test                  = 10  # particle. Number of test events to reserve
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
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10
        ]
class event_hists(object):
    # A data structure which contains the Eta, Phi, pt, and invariant
    # mass of the jet, maintaing those two sets for jets of 
    # pdgid = 5 AKA  'bjet', and non bjets.
    def __init__(self):
        self.hists = {}
        for i, title in enumerate(titles): self.hists[i] = []

    def update(self, isbjet, eta, mass, phi, pt):
        update_vals = [eta, mass, phi, pt]
        if isbjet:
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
            plt.savefig("hists/"+titles[i]+".png")
            # out_list.append(plt.object) whatever code here
            plt.close()
                
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
def update(bquarks, particle):
    # if the particle has the pid of a b quark and its status is such that 
    # it is done with its iterative process, append that particle to the 
    # list of bquarks
    if abs(particle.pid) == 5 and particle.status == 71:
        bquarks.append(particle)

def update_if_bjet(jet, bquarks, events_hists):  
    jet_lv = LorentzVector(jet.px, jet.py, jet.pz, jet.e)
    for bquark in bquarks:
        bquark_lv = LorentzVector(bquark.px, bquark.py, bquark.pz, bquark.e)
        if (jet_lv.deltar(bquark_lv) < 0.4):
              events_hists.update(True,  jet.eta, jet.mass, jet.phi, jet.pt)
        else: events_hists.update(False, jet.eta, jet.mass, jet.phi, jet.pt)

def pythia_sim(cmd_file, part_name=""):
    # The main simulation. Takes a cmd_file as input. part_name 
    # is the name of the particle we're simulating decays from.
    # Only necessary for titling graphs.
    # Returns an array of 2D histograms, mapping eta, phi, with transverse
    # energy.
    pythia                   = Pythia(cmd_file, random_state=1)
    events_data_package      = event_hists()
    bquarks                  = []
    bjets                    = []
    otherjets                = []
    for event in pythia(events=num_events):
        final_state_selection    = ((STATUS == 1)   & 
                                 ~HAS_END_VERTEX    &
                                 (ABS_PDG_ID != 12) &
                                 (ABS_PDG_ID != 14) &
                                 (ABS_PDG_ID != 16))
        particles                = event.all(return_hepmc=True)
        for particle in particles: update(bquarks, particle)
        jet_inputs               = event.all(final_state_selection)
        jet_sequence             = cluster(jet_inputs, ep=True, R=0.4, p=-1)
        jets                     = jet_sequence.inclusive_jets(ptmin=20)
        for jet in jets          : update_if_bjet(jet, bquarks, events_data_package)
    return events_data_package

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
    T_i, T_o       = shuffle_and_stich(ttbar_data, zz_data,
                                       ttbar_mapping, zz_mapping)
    Test_i, Test_o = shuffle_and_stich(ttbar_training, zz_training,
                                       ttbar_training_map, zz_training_map)
    care_package = np.array([T_i, T_o, Test_i, Test_o], dtype=object)
    return care_package

def make_plots(data_pak): data_pak.save_1d_hists()
# -----------------------------------------------------------------------
# Main process 
# -----------------------------------------------------------------------
while np.load(open("control", "rb")):
    # ttbar_tensor has indices of event, followed by eta, followed by phi.
    # The value of the h_tensor is the associated transverse energy.
    higgs_ww_data = pythia_sim('higgsww.cmnd', "higgsWW"); make_plots(higgs_ww_data)
    higgs_zz_data = pythia_sim('higgszz.cmnd', 'higgsZZ'); make_plots(higgs_zz_data)
    break
    #care_package  = structure_data_into_care_package([ttbar_data, zz_data])
    #ship(care_package)
print("Data Generation Halted")
