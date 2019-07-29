import bs4
from bs4 import BeautifulSoup
import os
import progressbar as pb
import pickle

#initialize widgets for progress bar
widgets = ['COMPILING CONCEPTS: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
#initialize timer for progress bar
t = pb.ProgressBar(widgets=widgets, maxval=1000000).start()

files = os.listdir(r'C:\\Users\\srajendr\\WFU\\Output\\PatientData\\reports')
notes_concepts = []
for i, html in enumerate(files):
    t.update(i)
    with open(r"C:\\Users\\srajendr\\WFU\\Output\\PatientData\\reports" + '\\' + html) as fp:
        this_concepts = []
        soup = BeautifulSoup(fp, features="html.parser")
        concepts = soup.find_all('a')
        i=0
        for i in range(len(concepts)):
            this_concepts.append(concepts[i].text.lower())
        notes_concepts.append(this_concepts)

print(len(notes_concepts))

f = open('PickleFiles/notes_concepts.pckl', 'wb')
pickle.dump(notes_concepts, f)
f.close()
print("Saved concepts of notes")

t.finish() # takes around 20 minutes to finish