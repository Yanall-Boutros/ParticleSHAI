# prints to the user if the control variable is set to allow training or not
import numpy as np
if np.load(open("control", "rb")):
    print("control is currently set to 1, Data Generating and training is allowed")
else:
    print("control is currently set to 0, Data Generating and training is *NOT* allowed")
