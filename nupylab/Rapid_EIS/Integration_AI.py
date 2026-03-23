####### This script is intended to eventually interface with LabVIEW at the microprobe station
####### for the BEF AI/ML rapid impedance project

####### The script should first run a chronoamperometry experiment and pull impedance data from it
####### The AI program should receive each new set of impedance data and output the next frequency and a
####### boolean of whether the experiement is completed or not. In the Interim, I will just feed it frequencies
####### from a numpy logspace command.


from nupylab.drivers.biologic import BiologicPotentiostat
from time import sleep
import numpy as np
import pandas as pd


# Set your measurement parameters - these would be inputs from LabVIEW
amplitude = 0.03 #Volts
bias_volt = 0 #Volts
bias_dur = 0 #s, how long to hold the bias for before beginning measurement

#Just using this as a placeholder for the AI program
frequencies = np.logspace(6, -1, 70)
print(frequencies)

# Connect Biologic
potentiostat = BiologicPotentiostat("KBIO_DEV_SP200","USB0")
potentiostat.connect()
for c in potentiostat.channels
potentiostat.load_firmware(1)
