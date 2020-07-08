# Separate file to clean the clutter from all the irregularities in
# formatting plots
# ----------------------------------------------------------------------
# Import Statements
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# -----------------------------------------------------------------------
# Graph Labels and Ranges
# -----------------------------------------------------------------------
columns = [
           ("Mass", "GeV"), ("Eta", "$\eta$"), ("Phi", "$\phi$"),
           ("Transverse Momentum", "GeV"),
          ]
leading_ranges = [
                  (-1,100), (-5,5), (-4,4), (-5,800), (-0.5,81.5),
                 ]
agg_ranges = [
              (-1,100), (-15,15), (-4,4), (-5,100)
             ]
nbins = [100]*len(columns)
nbins[3] = 21
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
# Create a histogram of counts per event with respect to the event number
# for each physical property
# Plot of number of counts of mass of jet in the jets of that event
def make_the_plots(event_data, jets_data, part_name=""):
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
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_event10GeV_'.format(name)+part_name+'.png')
             plt.close()
             
             # plot the 10GeV range for aggregate data
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=10, range=r, align='right')
             plt.title(ag_tit)
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_jets10GeV_'.format(name)+part_name+'.png')
             plt.close()
             
             # plot the standard ranges.
             # Plot the data from each leading jet in each event
             fig, ax = plt.subplots()
             ax.hist(event_data[:,i], bins=nbins[i], range=leading_ranges[i])
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Events')
             plt.title("Leading Jet Data")
             plt.savefig('{0:s}_event_'.format(name)+part_name+'.png')
             plt.close()
             
             # Plot the data from all the jets in all events
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=nbins[i], range=agg_ranges[i])
             plt.title("Aggregate Data")
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_jets_'.format(name)+part_name+'.png')
             plt.close()
             
             # Plots with increments being 5GeV
             # Plot the data from each leading jet in each event
             fig, ax = plt.subplots()
             ax.hist(event_data[:,i], bins=(range(-1, int(max(event_data[:,i])) + 5, 5)))
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Events')
             plt.title("Leading Jet Data (bin width = 5 GeV)")
             plt.savefig('{0:s}_event_by_5_'.format(name)+part_name+'.png')
             plt.close()
             
             # Plot the data from all the jets in all events
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=(range(-1, int(max(jets_data[:,i])) + 5, 5)))
             plt.title("Aggregate Data (bin width = 5 GeV)")
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_jets_by_5_'.format(name)+part_name+'.png')
             plt.close()
             
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=nbins[i], range=agg_ranges[i])
             plt.title("Aggregate Data (log scale)")
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.yscale('log', nonposy='clip')
             plt.savefig('{0:s}_log_jets_'.format(name)+part_name+'.png')
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
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Events')
             plt.title("Leading Jet Data")
             plt.savefig('{0:s}_event_'.format(name)+part_name+'.png')
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
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.savefig('{0:s}_jets_'.format(name)+part_name+'.png')
             plt.close()
             # Plot the log data for all events
             fig, ax = plt.subplots()
             ax.hist(jets_data[:,i], bins=nbins[i], range=agg_ranges[i])
             plt.title("Aggregate Data (log scale)")
             ax.set_xlabel(part_name+' '+'{0:s} [{1:s}]'.format(name, units))
             ax.set_ylabel('Jets')
             plt.yscale('log', nonposy='clip')
             plt.savefig('{0:s}_log_jets_'.format(name)+part_name+'.png')
             plt.close()
