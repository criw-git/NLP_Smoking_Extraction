# import numpy as np
import os
import pandas as pd
# import csv
# Extracted C71 Code descriptions - SQL code in current directory

# Note Text Compilation per patient
    # To be replaced with direct data pull
notes = pd.read_csv('Patient data.csv', encoding = "ISO-8859-1")

PatientList = notes.PAT_MRN_ID.unique()
len(PatientList) # 782 Patients! :D

os.mkdir('C:/Users/srajendr/WFU/PatientData') # These are named after patient MRN
i=0
for i in range(0,len(PatientList)):
    note = notes[notes.PAT_MRN_ID == PatientList[i]]
    filename = 'C:/Users/srajendr/WFU/PatientData'+ str(PatientList[i]) + '.csv'
    note['NOTE_TEXT'].to_csv(filename, index=False, header=False) #, quoting=csv.QUOTE_NONE)
