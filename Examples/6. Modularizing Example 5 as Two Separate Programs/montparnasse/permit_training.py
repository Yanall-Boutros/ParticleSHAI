#!/usr/bin/env python3
# A simply python script that sets a file value to 1 to indicate continue the process
# of training
import numpy as np
np.save(open("control", "wb"), 1)
