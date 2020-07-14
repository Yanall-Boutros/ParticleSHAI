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
num_events            = 1000 # Number of events to process per parent
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
class NoneTypes_in_event():
    def __init__(self, nn, nt, pcnts, hpids):
        self.num_none = nn
        self.num_total = nt
        self.percents = pcnts
        self.hist_pids = hpids

def is_massless_or_isolated(jet, i):
    # Returns true if a jet is only constituated of one particle
    # (nconsts == 1) and has a pdgid equal to that
    # of a photon or a gluon
    noneflag = False
    if jet.userinfo is None:
        noneflag = True
    if len(jet.constituents_array()) == 1: 
        # Count number of bjets
        if noneflag and jet.userinfo is not None:
            print("what, why?")
    return False

def NoneType_statistics(jets, debug_data):
    num_nonetypes = 0
    hist_pids = {}
    for jet in jets:
        if jet.userinfo is None: num_nonetypes += 1
        else:
            label = np.abs(jet.userinfo[pid])
            if hist_pids.get(label) is None:
                hist_pids[label] = 1
            else:
                hist_pids[label] += 1
    data = NoneTypes_in_event(num_nonetypes, len(jets),
        num_nonetypes/len(jets), hist_pids)
    debug_data.append(data)
    return ("Number of jet.userinfo NoneTypes: " + str(num_nonetypes) +
            "\nTotal Number of Jets: " + str(len(jets)) +
            "\nPercent NoneType: " + str(num_nonetypes/len(jets)) +
            "\nPID count: " + hist_pids.__str__() + "\n" +
            "-"*72)

def pythia_sim(cmd_file):
    # The main simulation. Takes a cmd_file as input. 
    pythia                   = Pythia(cmd_file, random_state=1)
    selection                = ((STATUS == 1) & ~HAS_END_VERTEX)
    unclustered_particles    = []
    for event in pythia(events=num_events):
        vectors               = event.all(selection)
        sequence              = cluster(vectors, R=0.4, p=-1, ep=True) 
        jets                  = sequence.inclusive_jets()
        print(NoneType_statistics(jets, debug_data))
        # 
        # Vestigial code that I didn't delete because at first I thought
        # the context might be useful, I do not believe that to still be
        # the case.
        #
        unclustered_particles.append(sequence.unclustered_particles())
        for i, jet in enumerate(jets):
            if is_massless_or_isolated(jet, i): pass
# -----------------------------------------------------------------------
# Main process for generating tensor data
debug_data = []
pythia_sim('ff2hfftww.cmnd')

