# ----------------------------------------------------------------------
# Import Statements
# ----------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------
# Function Definitions
# ----------------------------------------------------------------------
def str_to_vec(line):
    out = []
    trimmed = line.split(" ")
    for i in range(0, 7, 2): out.append(float(trimmed[i]))
    return out

def calc_mom(vec):
   mom = 0
   for elem in vec[:3]:
      mom += (elem)**2
   return np.sqrt(mom)

def calc_mass(vec):
   i_mass = vec[3] * vec[3]
   p = calc_mom(vec)
   return ((i_mass + (p**2)) ** 0.5)

def str_starts_as_num(s):
    return s[0].isdigit() or s[0] == "-"
# ----------------------------------------------------------------------
# Data Initialization and Calculations
# ----------------------------------------------------------------------
with open("events2.out") as f: content = f.readlines()
content = [x.strip() for x in content]
#    Go through the file line by line. If there's a next event, then add
#                         the two events together and append to the data
vecs            = []
empty           = ''
last_elem_index = len(content) - 1
# For every line in all lines of input text
for i in range(len(content)): 
    line = content[i]
    # Initial check to prevent accessing an out of bounds element.
    if i != last_elem_index:
        next_line = content[i+1]
    if next_line is not empty and line is not empty:
        if str_starts_as_num(line) and str_starts_as_num(next_line):
            vec_a   = str_to_vec(line)
            vec_b   = str_to_vec(next_line)
            vec_sum = np.add(np.array(vec_a), np.array(vec_b))
            vecs.append(calc_mass(vec_sum))
vecs = np.array(vecs)
# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.hist(vecs, 600, range=[0, 1000])
plt.title("Hist. of Inv. Mass"), plt.xlabel("Mass [GeV]"), plt.ylabel("Events")
plt.savefig("hist.pdf")
plt.close()
