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
num_events            = 10000
titles = [
            'Number of Jets with NoneType jet.userinfo',
            'Number of Jets in the event',
            'Perentage of NoneType jets in an event',
            'Aggregate histogram of SomeType jet.userinfo["pdgid"]'
          ]
xlabels = [
            "Number of NoneType Jets",
            "Total Jets",
            "Percentages",
            "Pdgid"
        ]
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
    data = NoneTypes_in_event(
            num_nonetypes, len(jets),
            num_nonetypes/len(jets), hist_pids
           )
    debug_data.append(data)
    return ("Number of jet.userinfo NoneTypes: " + str(num_nonetypes) +
            "\nTotal Number of Jets: " + str(len(jets)) +
            "\nPercent NoneType: " + str(num_nonetypes/len(jets)) +
            "\nPID count: " + hist_pids.__str__() + "\n" +
            "-"*72)

def aggregate_pdgid_counts_across_events(debug_data):
    out_dict = {}
    for event in debug_data:
        for key in event.hist_pids:
            if out_dict.get(key) is None: out_dict[key] = 1
            else: out_dict[key] += 1
    return out_dict

def extract_data(debug_data):
    event_nones, event_ttal, event_percents, event_hist_pids = [], [], [], []
    for event in debug_data:
        event_nones.append(event.num_none)
        event_ttal.append(event.num_total)
        event_percents.append(event.percents)
    event_hist_pids = aggregate_pdgid_counts_across_events(debug_data)
    return [event_nones, event_ttal, event_percents, event_hist_pids]

def hist_debug_data(debug_data):
    event_stats = extract_data(debug_data)
    for i, event_stat in enumerate(event_stats):
        plt.figure()
        plt.title(titles[i])
        plt.xlabel(xlabels[i])
        plt.ylabel("Counts per event")
        if i == 3:
            x = list(event_stat.keys())
            y = list(event_stat.values())
            plt.bar(x, y)
            plt.savefig(('hists/'+xlabels[i]+'.pdf')) # pdf needed because png can't
            # render extremely thins bars
        else:
            plt.hist(event_stat)
            plt.savefig(('hists/'+xlabels[i]+'.png'))
        plt.close()

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
# -----------------------------------------------------------------------
# Main process for generating tensor data
# -----------------------------------------------------------------------
print(np.__version__)
print
debug_data = []             # Global Variable used locally in pythia_sim
pythia_sim('ff2hfftww.cmnd')
hist_debug_data(debug_data) # Remove this line if you don't want debug 
# statistics to be saved as .png/.pdf
# Some bjets were supposed to be created (pdgid = 5), but none are reported
