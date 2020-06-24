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
# -----------------------------------------------------------------------
# Initalize
# -----------------------------------------------------------------------
#model                 = tf.keras.models.load_model("first_model") 
pid                   = 'pdgid'
num_events            = 10000 # Number of events to process per parent
test                  = 1000  # particle. Number of test events to reserve
discarded_data        = []   # Archive of any particles discarded
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
def is_massless_or_isolated(jet):
    # Returns true if a jet is only constituated of one particle
    # (nconsts == 1) and has a pdgid equal to that
    # of a photon or a gluon
    if len(jet.constituents_array()) == 1: 
        if np.abs(jet.info[pid]) == 21 or np.abs(jet.info[pid]) == 22:
            return True
        # if a muon is outside of the radius of the jet, discard it
        if np.abs(jet.info[pid]) == 13 and 2*jet.mass/jet.pt > 1.0:
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
    for event in pythia(events=num_events):
        jets_particle_eta = []
        jets_particle_phi = []
        jets_particle_energy = []
        vectors = event.all(selection)
        sequence = cluster(vectors, R=1.0, p=-1, ep=True)
        jets = sequence.inclusive_jets()
        unclustered_particles.append(sequence.unclustered_particles())
        part_data = []
        for i, jet in enumerate(jets):
            part_data = return_particle_data(jet)
            if is_massless_or_isolated(jet):
                discarded_data.append(jet)
            else:
                jets_particle_eta.extend(part_data[0])
                jets_particle_phi.extend(part_data[1])
                jets_particle_energy.extend(part_data[2])
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
    return np.array(part_tensor)

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

def pred_comp(pred, real):
    # Predictions compare takes a predictions array as input,
    # and compares its results to an expected output (real array)
    # Returns the percentage of predictions which were accurate.
    c = 0
    predictions = np.round(pred)
    elems = len(predictions)
    for i in range(elems):
        if predictions[i] == real[i]:
            c += 1
    return c / elems
# -----------------------------------------------------------------------
# Main process for generating tensor data
# -----------------------------------------------------------------------
# ttbar_tensor has indices of event, followed by eta, followed by phi.
# The value of the h_tensor is the associated transverse energy.
ttbar_tensor = pythia_sim('ttbar.cmnd', "TTbar")
zz_tensor    = pythia_sim('zz.cmnd', 'ZZ')

ttbar_training     = ttbar_tensor[:test]
ttbar_training_map = np.ones(test)
ttbar_tensor       = ttbar_tensor[test:]

zz_training     = zz_tensor[:test]
zz_training_map = np.zeros(test)
zz_tensor       = zz_tensor[test:]

ttbar_mapping = np.ones(num_events - test)
zz_mapping    = np.zeros(num_events - test)

# T_i is the training data / the input tensor
# T_o is the expected value (closer to 1 is ttbar, closer to 0 is zz
T_i, T_o       = shuffle_and_stich(ttbar_tensor, zz_tensor,
                             ttbar_mapping, zz_mapping)
Test_i, Test_o = shuffle_and_stich(ttbar_training, zz_training,
                                   ttbar_training_map, zz_training_map)
# -----------------------------------------------------------------------
# Build the Feed-Forward neural network
# -----------------------------------------------------------------------
ffmodel = tf.keras.models.Sequential()
ffmodel.add(tf.keras.layers.Flatten())
ffmodel.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
ffmodel.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
ffmodel.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))
ffmodel.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
ffmodel.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metric=['acurracy'])
# -----------------------------------------------------------------------
# Train and Test the Networks
# -----------------------------------------------------------------------
ffmodel.fit(T_i, T_o, epochs=100)
ffmodel.save("first_model")
predicitons = ffmodel.predict(Test_i)
print("Neural Network correctly evaluated ", 100*pred_comp(predicitons, Test_o)
      ,"% of test data")
