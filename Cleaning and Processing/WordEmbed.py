from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models import Word2Vec
import numpy as np
import pickle
from timeit import default_timer as timer
from nltk import word_tokenize
import statistics

f = open('PickleFiles/cleaned_notes.pckl', 'rb')
notes = pickle.load(f)
f.close()
start = timer()

def lengths(x):
    length=[]
    for t in x:
        length.append(len(t))
    return length

# another parameter was data --> avoiding that for now
def textTokenize(text):    
    t = Tokenizer()
    t.fit_on_texts(text) #training phase
    word_index = t.word_index #get a map of word index
    sequences = t.texts_to_sequences(text)
    max_len=max(lengths(sequences))
    print('Found %s unique tokens' % len(word_index))
    text_tok=pad_sequences(sequences, maxlen=max_len)
    return text_tok, word_index, max_len

def textTokenizeEff(text):
    t = Tokenizer()
    t.fit_on_texts(text) #training phase
    word_index = t.word_index #get a map of word index
    sequences = t.texts_to_sequences(text)
    mean = statistics.mean(lengths(sequences))
    std_dev = statistics.stdev(lengths(sequences))
    max_len = int((mean + 4*std_dev) + 1)
    print('Found %s unique tokens' % len(word_index))
    text_tok=pad_sequences(sequences, maxlen=max_len)
    return text_tok, word_index, max_len

def word_Embed_GNV(word_index):   
    pretrain = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    #convert pretrained word embedding to a dictionary
    embedding_index=dict()
    for i in range(len(pretrain.wv.vocab)):
        word=pretrain.wv.index2word[i]
        if word is not None:
            embedding_index[word]=pretrain.wv[word]  
    #extract word embedding for train and test data
    vocab_size=len(word_index)+1
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_index.items():
    	embedding_vector = embedding_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return embedding_matrix

def make_w2v_model(notes, window, workers, epochs):
    model = gensim.models.Word2Vec(notes, size=300, window=window, min_count=2, workers=workers)
    print('Start training process...') 
    model.train(notes,total_examples=len(notes),epochs=epochs)
    model.save("w2v.model")
    print("Model Saved")


def word_Embed_w2v(word_index, model):   
    pretrain = model
    #convert pretrained word embedding to a dictionary
    embedding_index=dict()
    for i in range(len(pretrain.wv.vocab)):
        word=pretrain.wv.index2word[i]
        if word is not None:
            embedding_index[word]=pretrain.wv[word]  
    #extract word embedding for train and test data
    vocab_size=len(word_index)+1
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_index.items():
    	embedding_vector = embedding_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Execution of variable creation for both whole and efficient
notes_tok, word_index, max_len = textTokenize(notes)
notes_tok_eff, word_index_eff, max_len_eff = textTokenizeEff(notes)
embedding_matrix_GNV = word_Embed_GNV(word_index)
embedding_matrix_GNV_eff = word_Embed_GNV(word_index_eff)
make_w2v_model(notes, 5, 10, 20)
w2v_model = Word2Vec.load("w2v.model")
embedding_matrix_w2v = word_Embed_w2v(word_index, w2v_model)
embedding_matrix_w2v_eff = word_Embed_w2v(word_index_eff, w2v_model)

def pickle_whole_variable():
    f = open('PickleFiles/tokenized_notes.pckl', 'wb')
    pickle.dump(notes_tok, f)
    f.close()
    print("Saved Tokenized Notes")

    f = open('PickleFiles/embedding_matrix_GNV.pckl', 'wb')
    pickle.dump(embedding_matrix_GNV, f)
    f.close()
    print("Saved Google Vector Word Embedding Matrix")

    f = open('PickleFiles/embedding_matrix_w2v.pckl', 'wb')
    pickle.dump(embedding_matrix_w2v, f)
    f.close()
    print("Saved Word 2 Vector Embedding Matrix")

    f = open('PickleFiles/word_index.pckl', 'wb')
    pickle.dump(word_index, f)
    f.close()
    print("Saved Word Indices")

    f = open('PickleFiles/max_len.pckl', 'wb')
    pickle.dump(max_len, f)
    f.close()
    print("Saved Maximum Length of One Patient's Notes")

def pickle_eff_variable():
    f = open('PickleFiles/tokenized_notes_eff.pckl', 'wb')
    pickle.dump(notes_tok_eff, f)
    f.close()
    print("Saved Tokenized Notes")

    f = open('PickleFiles/embedding_matrix_GNV_eff.pckl', 'wb')
    pickle.dump(embedding_matrix_GNV_eff, f)
    f.close()
    print("Saved Google Vector Word Embedding Matrix")

    f = open('PickleFiles/embedding_matrix_w2v_eff.pckl', 'wb')
    pickle.dump(embedding_matrix_w2v_eff, f)
    f.close()
    print("Saved Word 2 Vector Embedding Matrix")

    f = open('PickleFiles/word_index_eff.pckl', 'wb')
    pickle.dump(word_index_eff, f)
    f.close()
    print("Saved Word Indices")

    f = open('PickleFiles/max_len_eff.pckl', 'wb')
    pickle.dump(max_len_eff, f)
    f.close()
    print("Saved Maximum Length of One Patient's Notes")

# Save variables
#pickle_whole_variable()
pickle_eff_variable()

end = timer() # around 5 - 8 minutes to run the whole thing
print("Done within " + str(end-start) + " seconds")