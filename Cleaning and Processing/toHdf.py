import pandas as pd
import pickle

# Tokenized notes whole
f = open('PickleFiles/tokenized_notes.pckl', 'rb')
notes = pickle.load(f)
f.close()
df = pd.DataFrame(notes)
df.to_hdf('PandaFiles/tokenized_notes.h5', key='df')

#Emebedding matrix for w2v whole
f = open('PickleFiles/embedding_matrix_w2v.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('PandaFiles/embedding_matrix_w2v.h5', key='df')

#Embedding matrix for gnv whole
f = open('PickleFiles/embedding_matrix_GNV.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('PandaFiles/embedding_matrix_GNV.h5', key='df')

#####################################################################################

# Tokenized notes eff
f = open('PickleFiles/tokenized_notes_eff.pckl', 'rb')
notes = pickle.load(f)
f.close()
df = pd.DataFrame(notes)
df.to_hdf('PandaFiles/tokenized_notes_eff.h5', key='df')

#Emebedding matrix for w2v eff
f = open('PickleFiles/embedding_matrix_w2v_eff.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('PandaFiles/embedding_matrix_w2v_eff.h5', key='df')

#Embedding matrix for gnv eff
f = open('PickleFiles/embedding_matrix_GNV_eff.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('PandaFiles/embedding_matrix_GNV_eff.h5', key='df')

#############################################################################################