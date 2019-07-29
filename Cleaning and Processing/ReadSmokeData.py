# import numpy as np
import os
import pandas as pd
# import csv
# Extracted C71 Code descriptions - SQL code in current directory

# Note Text Compilation per patient
    # To be replaced with direct data pull
notes = pd.read_csv('smoking.csv', encoding = "ISO-8859-1")

PatientList = notes.MRN.unique()
len(PatientList) # 

os.mkdir('C:/Users/srajendr/WFU/SmokingData') # These are named after patient MRN
i=0
for i in range(0,len(PatientList)):
    note = notes[notes.MRN == PatientList[i]]
    filename = 'C:/Users/srajendr/WFU/SmokingData/PatientData'+ str(PatientList[i]) + '.csv'
    note['Smoking'].to_csv(filename, index=False, header=False) #, quoting=csv.QUOTE_NONE)