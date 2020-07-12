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
# -----------------------------------------------------------------------
# Initalize
# -----------------------------------------------------------------------
pid                   = 'pdgid'
bjet_id               = 5
num_events            = 1000 # Number of events to process per parent
test                  = 100  # particle. Number of test events to reserve
discarded_data        = []   # Archive of any particles discarded
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
def is_massless_or_isolated(jet, i):
    # Returns true if a jet is only constituated of one particle
    # (nconsts == 1) and has a pdgid equal to that
    # of a photon or a gluon
    noneflag = False
    if jet.userinfo is None:
        noneflag = True
        print("Nonetype in userinfo, jet i = ", i, "Inside inner function call") 
    else: print("Sometype in userinfo, jet i = ", i, "Inside inner function call")
    if len(jet.constituents_array()) == 1: 
        # Count number of bjets
        if noneflag and jet.userinfo is not None:
            print("what, why?")
        else: print("Sometype in userinfo, jet i = ", i, "jet.userinfo[pid] = ",
                jet.userinfo[pid])
    return False

def pythia_sim(cmd_file, part_name=""):
    # The main simulation. Takes a cmd_file as input. part_name 
    # is the name of the particle we're simulating decays from.
    # Only necessary for titling graphs.
    # Returns an array of 2D histograms, mapping eta, phi, with transverse
    # energy.
    pythia = Pythia(cmd_file, random_state=1)
    selection = ((STATUS == 1) & ~HAS_END_VERTEX)
    unclustered_particles = []
    num_b_jets_per_event  = []
    for event in pythia(events=num_events):
        vectors              = event.all(selection)
        sequence             = cluster(vectors, R=0.4, p=-1, ep=True) 
        jets                 = sequence.inclusive_jets()
        unclustered_particles.append(sequence.unclustered_particles())
        num_b_jets  = 0
        user_info_none_count = 0
        for i, jet in enumerate(jets):
            if jet.userinfo is None:
                print("Nonetype in userinfo, jet i = ", i)
            else: print("Sometype in userinfo, jet i = ", i, "Outside inner function call")
            if is_massless_or_isolated(jet, i):
                pass
# -----------------------------------------------------------------------
# Main process for generating tensor data
# -----------------------------------------------------------------------
higgs_ww_data    = pythia_sim('higgsww.cmnd', "higgsWW")
higgs_zz_data    = pythia_sim('higgszz.cmnd', 'higgsZZ')
