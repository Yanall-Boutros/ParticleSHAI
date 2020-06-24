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
# If you're 
def print_debug_data(dd):
    for tup in dd:
        j = tup[1]
        print("Mass = ", j[0], "\tpT = ", j[3], "\tEffR = ", j[5])
# -----------------------------------------------------------------------
# Generate Events
# -----------------------------------------------------------------------
selection = ((STATUS == 1) & ~HAS_END_VERTEX)
unclustered_particles = list()
num_events = 1000
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
# -----------------------------------------------------------------------
# Plot Data
# -----------------------------------------------------------------------
# Create a histogram of counts per event with respect to the event number
# for each physical property
# Plot of number of counts of mass of jet in the jets of that event
def make_the_plots(event_data, jets_data):
    # So much plotting, so little time. this section of the code might not be
    # worth reviewing, unless you're interested in various ways of formatting
    # and presenting a histogram
    for i,(name,units) in enumerate(columns):
        # Plot the same data but in the 0-10GeV range for Transverse Momentum
        # and Mass. As well as plot the entire data but with 5GeV increments
        if i == 0 or i == 3:
             # length of range divided by 5 = num of bins
             b_l = int((leading_ranges[i][1] - leading_ranges[i][0])/5)+1
             b_a = int((agg_ranges[i][1] - agg_ranges[i][0])/5)+1
             # plot the 10GeV range for for leading jet data
             fig, ax = plt.subplots()
             r = (-0.5, 10.5) # Set the range to 10 GeV
             ag_tit = "Aggregate Data 0-10Gev"
             lead_tit = "Leading Jet Data 0-10GeV"
             if i == 3: # Unless we're plotting the Momentum
                 r=(-1, 100) # then set the range to -1 to 100
                 ag_tit = "Aggregate Data 0-100Gev" # and adjust the titles
                 lead_tit = "Leading Jet Data 0-100GeV" # accordingly
             ax.hist(event_data[:,i], bins=10, range=r, align='right')
             plt.title(lead_tit)
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_event10GeV.png'.format(name))
             plt.close()
             
             # plot the 10GeV range for aggregate data
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=10, range=r, align='right')
             plt.title(ag_tit)
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_jets10GeV.png'.format(name))
             plt.close()
             
             # plot the standard ranges.
             # Plot the data from each leading jet in each event
             fig, ax = plt.subplots()
             ax.hist(event_data[:,i], bins=nbins[i], range=leading_ranges[i])
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Events')
             plt.title("Leading Jet Data")
             plt.savefig('{0:s}_event.png'.format(name))
             plt.close()
             
             # Plot the data from all the jets in all events
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=nbins[i], range=agg_ranges[i])
             plt.title("Aggregate Data")
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_jets.png'.format(name))
             plt.close()
             
             # Plots with increments being 5GeV
             # Plot the data from each leading jet in each event
             fig, ax = plt.subplots()
             ax.hist(event_data[:,i], bins=(range(-1, int(max(event_data[:,i])) + 5, 5)))
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Events')
             plt.title("Leading Jet Data (bin width = 5 GeV)")
             plt.savefig('{0:s}_event_by_5.png'.format(name))
             plt.close()
             
             # Plot the data from all the jets in all events
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=(range(-1, int(max(jets_data[:,i])) + 5, 5)))
             plt.title("Aggregate Data (bin width = 5 GeV)")
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_jets_by_5.png'.format(name))
             plt.close()
             
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=nbins[i], range=agg_ranges[i])
             plt.title("Aggregate Data (log scale)")
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.yscale('log', nonposy='clip')
             plt.savefig('{0:s}_log_jets.png'.format(name))
             plt.close()
        else: 
             # Plot the data from each leading jet in each event
             fig, ax = plt.subplots()
             if i == 4:
                 nbins[i] = int(leading_ranges[i][1] - leading_ranges[i][0])
                 ax.hist(event_data[:,i], bins=nbins[i],
                         range=leading_ranges[i], align='left')
             else: 
                 ax.hist(event_data[:,i], bins=nbins[i],
                         range=leading_ranges[i])
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Events')
             plt.title("Leading Jet Data")
             plt.savefig('{0:s}_event.png'.format(name))
             plt.close()
             # Plot the data from all the jets in all events
             fig, ax = plt.subplots()
             if i == 4:
                 nbins[i] = int(agg_ranges[i][1] - agg_ranges[i][0])
                 ax.hist(jets_data[:,i], bins=nbins[i],
                         range=agg_ranges[i], align='left')
             else:    
                 ax.hist(jets_data[:,i], bins=nbins[i], range=agg_ranges[i])
             plt.title("Aggregate Data")
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_jets.png'.format(name))
             plt.close()
             # Plot the log data for all events
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=nbins[i], range=agg_ranges[i])
             plt.title("Aggregate Data (log scale)")
             ax.set_xlabel('{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.yscale('log', nonposy='clip')
             plt.savefig('{0:s}_log_jets.png'.format(name))
             plt.close()

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
      if np.abs(jet.info['pdgid']) == 21 or np.abs(jet.info['pdgid']) == 22:
         return True
      # if a muon is outside of the radius of the jet, discard it
      if np.abs(jet.info['pdgid']) == 13:
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
    pythia = Pythia(cmd_file, random_state=1)
    selection = ((STATUS == 1) & ~HAS_END_VERTEX)
    unclustered_particles = []
    debug_data     = [] # Deprecated but costs 1 operation per function call so negligble
    event_data     = [] # Eent data means the leading jet information
    sub_jet_data   = [] # There are multiple jets in each event
    discarded_data = [] # For analyzing what kids thrown out from function is_massless_or_isolated
    for event in pythia(events=num_events):
        jetpts   = []
        vectors  = event.all(selection)
        sequence = cluster(vectors, R=0.4, p=-1, ep=True) # Note to self: R might need to be changed to 1
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
                if i == 0: # Todo: Append the leading 3 jets of the event
                    event_data.append(data)
                sub_jet_data.append(data)
    if make_plots:
        event_data = np.array(event_data)
        sub_jet_data = np.array(sub_jet_data)
        make_the_plots(event_data, sub_jet_data)
    event_data = np.array(event_data)
    sub_jet_data = np.array(sub_jet_data)
    return event_data, sub_jet_data
# -----------------------------------------------------------------------
# Main Loop for Storing Jet Data
# -----------------------------------------------------------------------
event_data, sjdata = pythia_sim("ttbar.cmnd", make_plots=True)
