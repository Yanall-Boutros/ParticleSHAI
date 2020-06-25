# ---------------------------------------------------------------------
# Import Statements
# ---------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pickle as pic
import os
import os.path
import matplotlib.pyplot as plt
import time
# ---------------------------------------------------------------------
# Function Definitions
# ---------------------------------------------------------------------
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

def get_care_packages():
    base_name = "care_package"
    i = 0
    # If there are no care packages, then wait
    while not os.path.isfile(base_name+str(i)):
        print("No care_package file detected, waiting 10 seconds...")
        time.sleep(10)
        # quick check to see if it's time to stop
        if not np.load(open("control", "rb")):
            print("Training Halted")
            exit()
    # iterate through all the care_packagen files where n is an arbitrary
    # number for valid carepackages, then export all as a list, deleting as
    # we append to the list
    care_pallet = [] # A pallet contains many packages
    while os.path.isfile(base_name+str(i)):
        care_pallet.append(np.load(open(base_name+str(i), "rb"),
                                   allow_pickle=True))
        os.remove(base_name+str(i))
        i += 1
    return care_pallet
def ROC(ans, pred):
    tp = []
    fp = []
    for trigger in np.linspace(0,1,1000):
        total = 0
        tcount = 0
        fcount = 0
        for a,p in zip(ans,pred):
            if a: total += 1
            if p > trigger and a: tcount += 1
            elif p > trigger and not a: fcount += 1
        tp.append(t/total)
        fp.append(f/total)
    plt.plot(fp, tp)
    plt.title("ROC Curve for Identifying $T\overline{T}$")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("ROC.pdf")
    return tp, fp
# ---------------------------------------------------------------------
# Check if model exists, if so, load it. Otherwise, make new network
# ---------------------------------------------------------------------
if os.path.isfile("ff_model"):
    ffmodel = tf.keras.models.load_model("ff_model")
else:
    # Redesign neural network to take in as input 997 indexed jet objects,
    # Consisting of 3x6 dimensional float values per index
    ffmodel = tf.keras.models.Sequential()
    ffmodel.add(tf.keras.layers.Flatten())
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(18, activation=tf.nn.relu))
    ffmodel.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

ffmodel.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metric=['acurracy'])
# -----------------------------------------------------------------------
# Train and Test the Network
# -----------------------------------------------------------------------
tp, fp = 0, 0
while np.load(open("control", "rb")):
    care_pallet = get_care_packages()
    for care_package in care_pallet:
        if not np.load(open("control", "rb")):
            break
        T_i, T_o = care_package[0], care_package[1] 
        Test_i, Test_o = care_package[2], care_package[3]
        history = ffmodel.fit(T_i, T_o, epochs=30)
        ffmodel.save("ff_model")
        predictions = ffmodel.predict(Test_i) # vtr stands for
        vtr = pred_comp(predictions, Test_o) # validation testing results
        print("Neural Network correctly evaluated ", 100*vtr ,"% of test data")
        tp, fp = ROC(Test_o, predictions)
print("Training Halted")
