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
import itertools
import pickle as pic
import os.path
# -----------------------------------------------------------------------
# Initalize
# -----------------------------------------------------------------------
pid                   = 'pdgid'
bjet_id               = 5
num_events            = 1000 # Number of events to process per parent
test                  = 100  # particle. Number of test events to reserve
discarded_data        = [] # Archive of any particles discarded
higgs_dat             = [] # Should restructure classes to avoid needing to mmake
                           # a global variable but this is quicker.
titles = [
            "Histogram of bjet eta",
            "Histogram of bjet phi",
            "Histogram of bjet mass",
            "Histogram of bjet pt",
            "Histogram of nonbjet eta",
            "Histogram of nonbjet phi",
            "Histogram of nonbjet mass",
            "Histogram of nonbjet pt",
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
        ]
ranges = [
            (-4, 4),
            (-np.pi, np.pi),
            (0, 40),
            (0, 200),
            (-6, 6),
            (-np.pi, np.pi),
            (0, 40),
            (0, 1600)
        ]

ylim = [
            (0, 55),
            (0, 30),
            (0, 150),
            (0, 140),
            (0, 170),
            (0, 140),
            (0, 800),
            (0, 2000),
        ]
class event_hists(object):
    # A data structure which contains the Eta, Phi, pt, and invariant
    # mass of the jet, maintaing those two sets for jets of 
    # pdgid = 5 AKA  'bjet', and non bjets.
    def __init__(self, folder_dest=""):
        self.hists = {}
        for i, title in enumerate(titles): self.hists[i] = []
        self.fdest = folder_dest

    def update(self, isbjet, eta, mass, phi, pt):
        update_vals = [eta, phi, mass, pt]
        if isbjet:
            for i, title in enumerate(titles[:4]): 
                self.hists[i].append(update_vals.pop(0))
        else:
            dev = 4
            for i, title in enumerate(titles[4:]):
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
            plt.ylim(ylim[i][0], ylim[i][1])
            plt.savefig("hists/"+self.fdest+"/"+titles[i]+".png")
            # out_list.append(plt.object) whatever code here
            plt.close()
                
class higgs_hists(object):
    # Like event hists, but designed only for higgs. The two classes should
    # be generalized better, inheriting from a mutual parent class
    def __init__(self, folder_dest=""):
        self.hists = {}
        self.titles = [
                    "Higgs Eta",
                    "Higgs Phi",
                    "Higgs Mass",
                    "Higgs Pt"
            ]
        self.nranges = [
                (0, 125),
                (0, 75),
                (0, 12500),
                (0, 7200)
            ]
        self.nbins = [
                   250,
                   250,
                   250,
                   250,
                   250,
                   250,
                   250,
                   250
                ]
        for i, title in enumerate(self.titles): self.hists[i] = []
        self.fdest = folder_dest

    def update(self, eta, phi, mass, pt):
        for i, title in enumerate(self.titles):
            self.hists[i].append([eta, phi, mass, pt][i])

    def save_1d_hists(self):
        out_list = []
        for i,title in enumerate(self.titles):
            plt.figure()
            plt.hist(self.hists[i], bins=self.nbins[i])
            plt.title(self.titles[i])
            plt.xlabel(xlabels[i])
            plt.ylabel(ylabel)
            plt.ylim(self.nranges[i][0], self.nranges[i][1])
            plt.savefig("hists/"+self.fdest+"/"+titles[i]+".png")
            plt.close()

# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
def update(bquarks, higgs, particle):
    # if the particle has the pid of a b quark and its status is such that 
    # it is done with its iterative process, append that particle to the 
    # list of bquarks
    if abs(particle.pid) == 5 and particle.status == 71:
        bquarks.append(particle)
    if particle.status == 62:
        higgs.update(particle.eta, particle.phi, particle.mass, particle.pt)

def update_if_bjet(jet, bquarks, events_hists, bjets, otherjets):  
    jet_lv = LorentzVector(jet.px, jet.py, jet.pz, jet.e)
    for bquark in bquarks:
        bquark_lv = LorentzVector(bquark.px, bquark.py, bquark.pz, bquark.e)
        if (jet_lv.deltar(bquark_lv) < 0.4):
              events_hists.update(True,  jet.eta, jet.mass, jet.phi, jet.pt)
              bjets.append(jet)

        else: 
            events_hists.update(False, jet.eta, jet.mass, jet.phi, jet.pt)
            otherjets.append(jet)

def pythia_sim(cmd_file, part_name=""):
    # The main simulation. Takes a cmd_file as input. part_name 
    # is the name of the particle we're simulating decays from.
    # Only necessary for titling graphs.
    # Returns an array of 2D histograms, mapping eta, phi, with transverse
    # energy.
    pythia                   = Pythia(cmd_file, random_state=1)
    if part_name == "higgsWW":
        events_data_package  = event_hists("ff2HffTww")
        higgs_data_package   = higgs_hists("higgsww")
    else:
        events_data_package  = event_hists("ff2HffTzz")
        higgs_data_package   = higgs_hists("higgszz")
    bquarksAtEvent           = []
    bjetsAtEvent             = []
    otherJetsAtEvent         = []
    for event in pythia(events=num_events):
        bjets                    = []
        bquarks                  = []
        otherjets                = []
        final_state_selection    = ((STATUS == 1)   & 
                                 ~HAS_END_VERTEX    &
                                 (ABS_PDG_ID != 12) &
                                 (ABS_PDG_ID != 14) &
                                 (ABS_PDG_ID != 16))
        particles                = event.all(return_hepmc=True)
        for particle in particles: update(bquarks, higgs_data_package, particle)
        jet_inputs               = event.all(final_state_selection)
        jet_sequence             = cluster(jet_inputs, ep=True, R=0.4, p=-1)
        jets                     = jet_sequence.inclusive_jets(ptmin=20)
        for jet in jets          : update_if_bjet(jet, bquarks, events_data_package,
                                                  bjets, otherjets)
        bquarksAtEvent.append(bquarks)
        bjetsAtEvent.append(bjets)
        otherJetsAtEvent.append(otherjets)
    higgs_data_package.save_1d_hists()
    higgs_dat.append(higgs_data_package)
    return events_data_package, bquarksAtEvent, bjetsAtEvent, otherJetsAtEvent

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

def make_plots(data_pak):
    data_pak.save_1d_hists()

def plot_num_part_type(list_of_parts, process_type):
    titles = [
                "Number of bQuarks in Event",
                "Number of bjets in Event",
                "Number of other jets in Event"
            ]
    nbins   = [
                100,
                100,
                100
            ]
    xlabel = "Counts in Event"
    ylabel = "In $N$ Events"
    if process_type == "ww": fdest = "ff2HffTww"
    else                   : fdest = "ff2HffTzz"
    # list contains list of bquarks then bjets then otherjets
    for i, part in enumerate(list_of_parts):
        hist_data = []
        for j in range(len(part)): part[j] = len(part[j])
        plt.figure()
        plt.hist(part, bins=nbins[i])
        plt.title(titles[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("hists/"+fdest+"/"+titles[i]+".png")
        plt.close()

def plot_together(hwwdata, hzzdata):
    for i, title in enumerate(titles):
        plt.figure()
        nbins = n_bins[i]
        plt.hist(hwwdata.hists[i], alpha=0.75, label="Events from ww", bins=nbins)
        plt.hist(hzzdata.hists[i], alpha=0.75, label="Events from zz", bins=nbins)
        plt.legend(loc=4)
        plt.title(title)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabel)
        plt.savefig("hists/together/"+title+".png")
        plt.close()
        
def plot_higgs_together(hfww, hfzz):
    for i,title in enumerate(hfww.titles):
        plt.figure()
        nbins = hfww.nbins[i]
        plt.hist(hfww.hists[i], alpha=0.75, label="Higgs from ww", bins=nbins)
        plt.hist(hfzz.hists[i], alpha=0.75, label="Higgs from zz", bins=nbins)
        plt.legend(loc=4)
        plt.title(title)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabel)
        plt.savefig("hists/higgstogether/"+title+".png")
        plt.close()

def momentum_sqrd(fvec):
    m = 0
    for p in fvec[:3]: m += p**2
    return m
def calc_inv_mass(fvec):
    # indices 0-3 are momentum, last is energy
    return (fvec[3]**2 + momentum_sqrd(fvec))**0.5

def invmass(jet1, jet2):
    # first, get the jets into fourvecs
    j1 = np.array([jet1.px, jet1.py, jet1.pz, jet1.e])
    j2 = np.array([jet2.px, jet2.py, jet2.pz, jet2.e])
    fvec = np.add(j1, j2)
    return calc_inv_mass(fvec)

def most_rubenesque(wwjets, zzjets):
    ruben_likes  = [-1, -1]
    wruben_likes = []
    for wevent in wwjets:
        wmax = -1
        for jet1, jet2 in list(itertools.combinations(wevent, 2)):
            if (result := invmass(jet1, jet2)) > wmax:
                wmax = result
                ruben_likes[0] = jet1
                ruben_likes[1] = jet2
        wruben_likes.append(np.array([ruben_likes[0], ruben_likes[1], wmax]))
    zruben_likes = []
    for zevent in zzjets:
        zmax = -1
        for jet1, jet2 in list(itertools.combinations(zevent, 2)):
            if (result := invmass(jet1, jet2)) > zmax:
                zmax = result
                ruben_likes[0] = jet1
                ruben_likes[1] = jet2
        zruben_likes.append(np.array([ruben_likes[0], ruben_likes[1], zmax]))
    return np.array([wruben_likes, zruben_likes])

def draw_ruben(wf, zf):
    wmasses = wf[:,2]
    zmasses = zf[:,2]
    plt.figure()
    plt.hist(wmasses, label="Non b invariant masses from ww", bins=250, alpha=0.75)
    plt.hist(zmasses, label="Non b invariant masses from zz", bins=250, alpha=0.75)
    plt.xlabel("Invariant Mass $GeV$")
    plt.ylabel("Counts per event")
    plt.legend(loc=1)
    plt.title("Invariant Masses of Non-b jets")
    plt.savefig("hists/rubentogether/invmass.png")
# -----------------------------------------------------------------------
# Main process 
# -----------------------------------------------------------------------
while np.load(open("control", "rb")):
    # ttbar_tensor has indices of event, followed by eta, followed by phi.
    # The value of the h_tensor is the associated transverse energy.
    higgs_ww_data, wwbquarks, wwbjets, wwotherjets = pythia_sim('higgsww.cmnd',
                                                                     "higgsWW")
    make_plots(higgs_ww_data)
    plot_num_part_type([wwbquarks.copy(), wwbjets.copy(), wwotherjets.copy()], "ww")

    higgs_zz_data, zzbquarks, zzbjets, zzotherjets = pythia_sim('higgszz.cmnd',
                                                                     'higgsZZ')
    make_plots(higgs_zz_data)
    plot_num_part_type([zzbquarks.copy(), zzbjets.copy(), zzotherjets.copy()], "zz")

    plot_together(higgs_ww_data, higgs_zz_data)
    plot_higgs_together(*higgs_dat)
    ww_nonB_fattest, zz_nonB_fattest = most_rubenesque(wwotherjets, zzotherjets)
    draw_ruben(ww_nonB_fattest, zz_nonB_fattest)
    break
    #care_package  = structure_data_into_care_package([ttbar_data, zz_data])
    #ship(care_package)
print("Data Generation Halted")
