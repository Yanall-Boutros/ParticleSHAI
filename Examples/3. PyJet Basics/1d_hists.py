# !/usr/bin/env python3
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
import sys
from make_the_plots import make_the_plots
# -----------------------------------------------------------------------
# Generate Events
# -----------------------------------------------------------------------
top_i_jets = 1
num_events = 1000
cmd_file   = sys.argv[1]
part_name  = cmd_file[:cmd_file.find('.')]
make_plots = True
# -----------------------------------------------------------------------
# Graph Labels and Ranges
# -----------------------------------------------------------------------
columns = [
           ("Mass", "GeV"), ("Eta", "$\eta$"), ("Phi", "$\phi$"),
           ("Transverse Momentum", "GeV"),
           ("Number of constituents", "$n$"), ("Eff Radius", "$R$")
          ]
leading_ranges = [
                  (-1,100), (-5,5), (-4,4), (-5,800), (-0.5,81.5),
                  (-0.25, 0.75)
                 ]
agg_ranges = [
              (-1,100), (-15,15), (-4,4), (-5,100), (-0.5,41.5), (-0.25,40)
             ]
nbins = [100]*len(columns)
nbins[4] = 21

def pick_highest(jetpts):
    r = []               
    cur = 0
    for i, pt in enumerate(jetpts):
        if pt > cur:
            r.append(pt)
        cur = pt
    return r

def print_stuff(jet):
    for i, jets in enumerate(jet):
        print("In event ", i, " there are ", len(jets), " jets")
        print(jets[:50])

def is_massless_or_isolated(jet):
   # Returns true if a jet has nconsts = 1 and has a pdgid equal to that
   # of a photon or a gluon
   if len(jet.constituents_array()) == 1: 
      if np.abs(jet.userinfo['pdgid']) == 21 or np.abs(jet.userinfo['pdgid']) == 22:
         return True
      # if a muon is outside of the radius of the jet, discard it
      if np.abs(jet.userinfo['pdgid']) == 13:
         if 2*jet.mass/jet.pt > 0.4: return True
   # Remove Jets with too high an eta
   if np.abs(jet.eta) > 5.0:
      return True
   # Remove any jets less than an arbitrary near zero mass
   if jet.mass < 0.4:
      return True
   return False
# -----------------------------------------------------------------------
# Generate Jets and Histogram Data
# -----------------------------------------------------------------------
def pythia_sim(cmd_file, part_name="", make_plots=False):
    # The main simulation. Takes a cmd_file as input. part_name 
    # is the name of the particle we're simulating decays from.
    # Only necessary for titling graphs.
    # Returns an array of 2D histograms, mapping eta, phi, with transverse
    # energy.
    debug_data            = [] # Deprecated but costs 1 operation per function call so negligble
    event_data            = [] # Event data -> the leading jet(s) information
    sub_jet_data          = [] # Continer for the multiple jets in each event
    discarded_data        = [] # For analyzing what is thrown out from function is_massless_or_isolated
    unclustered_particles = []
    pythia                = Pythia(cmd_file, random_state=1)
    selection             = ((STATUS == 1) & ~HAS_END_VERTEX)
    for event in pythia(events=num_events):
        jetpts   = []
        vectors  = event.all(selection)
        sequence = cluster(vectors, R=0.4, p=-1, ep=True) # R is radius of jets.
        jets     = sequence.inclusive_jets()
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
            else:
                if i < top_i_jets: # Todo: Append the leading 3 jets of the event
                    event_data.append(data)
                sub_jet_data.append(data)
    if make_plots:
        event_data = np.array(event_data)
        sub_jet_data = np.array(sub_jet_data)
        make_the_plots(event_data, sub_jet_data, part_name)
    event_data = np.array(event_data)
    sub_jet_data = np.array(sub_jet_data)
    return event_data, sub_jet_data
# -----------------------------------------------------------------------
# Execute Pythia Simulation
# -----------------------------------------------------------------------
event_data, sjdata = pythia_sim(cmd_file, part_name, make_plots)
