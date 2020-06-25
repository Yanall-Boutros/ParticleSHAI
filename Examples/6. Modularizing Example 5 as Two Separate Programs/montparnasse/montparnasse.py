#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import tensorflow as tf
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
        if np.abs(jet.userinfo[pid]) == 21 or np.abs(jet.userinfo[pid]) == 22:
            return True
        # if a muon is outside of the radius of the jet, discard it
        if np.abs(jet.userinfo[pid]) == 13 and 2*jet.mass/jet.pt > 1.0:
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

def pythia_sim(cmd_file, part_name="unnamed", make_plots=False):
    # The main simulation. Takes a cmd_file as input. part_name 
    # is the name of the particle we're simulating decays from.
    # Only necessary for titling graphs.
    # Returns an array of 2D histograms, mapping eta, phi, with transverse
    # energy.
    pythia = Pythia(cmd_file, random_state=1)
    selection = ((STATUS == 1) & ~HAS_END_VERTEX)
    unclustered_particles = []
    a = 0
    part_tensor = []
    sj_data_per_event = []
    for event in pythia(events=num_events):
        lead_jet_invalid = False
        sub_jet_data = [] # There are multiple jets in each event
        jets_particle_eta = []
        jets_particle_phi = []
        jets_particle_energy = []
        vectors = event.all(selection)
        sequence = cluster(vectors, R=1.0, p=-1, ep=True)
        jets = sequence.inclusive_jets()
        unclustered_particles.append(sequence.unclustered_particles())
        part_data = []
        for i, jet in enumerate(jets):
            jet_data = (
                jet.mass, jet.eta, jet.phi, jet.pt,
                len(jet.constituents_array()), 2*jet.mass/jet.pt
            )
            part_data = return_particle_data(jet)
            if is_massless_or_isolated(jet):
                discarded_data.append(jet)
                if i == 0: lead_jet_invalid = True
            else:
                jets_particle_eta.extend(part_data[0])
                jets_particle_phi.extend(part_data[1])
                jets_particle_energy.extend(part_data[2])
            if i < 3:
                sub_jet_data.append(jet_data)
        lead_jet_valid = not lead_jet_invalid
        if lead_jet_valid:
            sj_data_per_event.append(np.array(sub_jet_data))
            plt.figure()
            part_tensor.append(plt.hist2d(jets_particle_eta, jets_particle_phi,
                        weights=jets_particle_energy, normed=True,
                        range=[(-5,5),(-1*np.pi,np.pi)],
                        bins=(20,32), cmap='binary')[0]) # We're only taking the
            plt.close() # Zeroth element, which is the raw data of the 2D Histogram
            if make_plots:
                plt.xlabel("$\eta$")
                plt.ylabel("$\phi$")
                plt.title("Particles from "+part_name)
                cbar = plt.colorbar()
                cbar.set_label('Tranverse Energy of Each Particle ($GeV$)')
                plt.savefig("hists/Jets_Particles_"+part_name+str(a)+".png")
            a += 1
    return np.array(part_tensor), np.array(sj_data_per_event)

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
def split_to_train_test(data):
    # Given a data set, it splits / reserves a fraction of data defined
    # at a cut off point for validation purposes, defined by the global
    # variable test
    data_validation_set = data[:test]
    data = data[test:]
    return data_validation_set, data
# -----------------------------------------------------------------------
# Main process for generating data
# -----------------------------------------------------------------------
while np.load(open("control", "rb")):
    # ttbar_tensor has indices of event, followed by eta, followed by phi.
    # The value of the h_tensor is the associated transverse energy.
    ttbar_part_tensor, ttbar_jet_tensor = pythia_sim('ttbar.cmnd', "TTbar")
    zz_part_tensor, zz_jet_tensor = pythia_sim('zz.cmnd', 'ZZ')
    
    ttbar_part_validation, ttbar_part_data = split_to_train_test(
        ttbar_part_tensor
    )
    ttbar_jet_validation, ttbar_jet_data = split_to_train_test(
        ttbar_jet_tensor
    )
    ttbar_training_answers = np.ones(num_events - test)
    ttbar_validation_answers = np.ones(test)
    
    zz_part_validation, zz_part_data = split_to_train_test(
        zz_part_tensor
    )
    zz_jet_validation, zz_jet_data = split_to_train_test(
        zz_jet_tensor
    )
    zz_training_answers = np.zeros(num_events - test)
    zz_validation_answers = np.zeros(test)
    
    # Learning_set_i is the training data / the input tensor
    # Learning_set_o is the simulated known value
    # (closer to 1 is ttbar, closer to 0 is zz
    Learning_part_set_i, Learning_part_set_o = shuffle_and_stich(
        ttbar_part_data,
        zz_part_data,
        ttbar_training_answers,
        zz_training_answers
    )
    Learning_jet_set_i, Learning_jet_set_o = shuffle_and_stich(
        ttbar_jet_data,
        zz_jet_data,
        ttbar_training_answers,
        zz_training_answers
    )
    Validation_part_i, Validation_part_o = shuffle_and_stich(
        ttbar_part_validation,
        zz_part_validation,
        ttbar_validation_answers,
        zz_validation_answers
    )
    Validation_jet_i, Validation_jet_o = shuffle_and_stich(
        ttbar_jet_validation,
        zz_jet_validation,
        ttbar_validation_answers,
        zz_validation_answers
    )
    care_package = np.array(
        [
            Learning_part_set_i,
            Learning_part_set_o,
            Validation_part_i,
            Validation_part_o,
            Learning_jet_set_i,
            Learning_jet_set_o,
            Validation_jet_i,
            Validation_jet_o
        ]
    )
    ship(care_package)
print("Data Generation Halted")
