import os
import csv
import pickle

# function to read in all the csv files
def readcsv(filename, root):
        filename = root + filename
        # fields = [] 
        rows = [] 

        with open(filename, 'r', encoding='utf8', errors='ignore') as csvfile: 
                csvreader = csv.reader(csvfile) 
                for row in csvreader: 
                        rows.append(row) 
        text = []
        for row in rows[:]: 
        # parsing each column of a row 
                for col in row: 
                        text.append(col) # at this point each row is what was in the original csv but each column is one character

        full_characters = []
        for row in text:
                full_characters += row

        full_text = ""
        for i in full_characters:
                full_text += i

        return full_text

# creating a list of all the files in the PatientData folder 
files = os.listdir('PatientData/')
root = 'PatientData/'
notes = []

# going through each of the documents and converting it into a string to be added to notes[]
for document in files:
            notes.append(readcsv(document, root))

# saving data via pickle
f = open('PickleFiles/original_notes.pckl', 'wb')
pickle.dump(notes, f)
f.close()