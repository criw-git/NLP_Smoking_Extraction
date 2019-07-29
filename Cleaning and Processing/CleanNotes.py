import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import os
import csv
import pickle
from timeit import default_timer as timer
import inflect
from autocorrect import spell
from collections import OrderedDict
import progressbar as pb

#initialize widgets for progress bar
widgets = ['CLEANING NOTES: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
#initialize timer for progress bar
t = pb.ProgressBar(widgets=widgets, maxval=1000000).start()

# function that cleans text
# still need to account for contractions, abbreviations, and numbers/fractions
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your choice
def clean_text(i, text, notes_concepts, replace_numbers = False, remove_rare = False, remove_punctuation = False, stem_text = False, remove_stopwords = False, remove_num = False , spell_check = False, remove_repeat = False):
        def misc_cleaning(text):
                text = re.sub("-([a-zA-Z]+)", r"\1", text) # replaces hyphen with spaces in case of strings
                text = re.sub(' y ', '', text) # gets rid of random y accent stuff scattered through the text
                text = re.sub('yyy', 'y', text)
                text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
                text = re.sub(r"what's", "what is ", text)
                text = re.sub(r"\'s", " ", text)
                text = re.sub(r"\'ve", " have ", text)
                text = re.sub(r"can't", "cannot ", text)
                text = re.sub(r"n't", " not ", text)
                text = re.sub(r"i'm", "i am ", text)
                text = re.sub(r"\'re", " are ", text)
                text = re.sub(r"\'d", " would ", text)
                text = re.sub(r"\'ll", " will ", text)
                text = re.sub(r",", " ", text)
                text = re.sub(r"\.", " ", text)
                text = re.sub(r"!", " ! ", text)
                text = re.sub(r"\/", " ", text)
                text = re.sub(r"\^", " ^ ", text)
                text = re.sub(r"\+", " + ", text)
                text = re.sub(r"\-", " - ", text)
                text = re.sub(r"\=", " = ", text)
                text = re.sub(r"'", " ", text)
                text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
                text = re.sub(r":", " : ", text)
                text = re.sub(r" e g ", " eg ", text)
                text = re.sub(r" b g ", " bg ", text)
                text = re.sub(r" u s ", " american ", text)
                text = re.sub(r"\0s", "0", text)
                text = re.sub(r" 9 11 ", "911", text)
                text = re.sub(r"e - mail", "email", text)
                text = re.sub(r"j k", "jk", text)
                text = re.sub(r"\s{2,}", " ", text)
                return text

        # function to tokenize text which is used in a lot of the later processing
        def tokenize_text(text):
                return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

        text = text.strip(' ') # strip whitespaces
        text = misc_cleaning(text) # look at function, random cleaning stuff
        text = text.lower() # lowercase

        if remove_repeat:
                sentences = sent_tokenize(text)
                sentences = list(dict.fromkeys(sentences))
                text = " ".join(sentences)
        
        # removes punctuation
        if remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))

        # does a spellcheck of the text and corrects words that are misspelled if they're least frequent
        if spell_check:
                concepts = notes_concepts[i]
                tokens = word_tokenize(text)
                freq_dist = nltk.FreqDist(tokens)
                testword = (list(freq_dist.keys())[-1:])[0]
                lowest_frequency = freq_dist.freq(testword)
                rarewords = []
                for word in tokens:
                        if freq_dist.freq(word) == lowest_frequency:
                                rarewords.append(word)
                rarewords[:] = [word for word in rarewords if word not in concepts]
                correctwords = []
                for word in rarewords:
                        correctwords.append(spell(word))
                for i in range(len(tokens)):
                        for a in range(len(rarewords)):
                                if tokens[i] == rarewords[a]:
                                        tokens[i] = correctwords[a]
                text = " ".join(tokens)

        # optional: replaces numbers ("3") with their word counterparts ("three")
        if replace_numbers:
                words = word_tokenize(text)
                p = inflect.engine()
                new_words = []
                for word in words:
                        if word.isdigit():
                                new_word = p.number_to_words(word)
                                new_words.append(new_word)
                        else:
                                new_words.append(word)
                text = " ".join(new_words)

        # optional: removes the rarest words in each text --> right now it's 10
        if remove_rare:
                tokens = word_tokenize(text)
                freq_dist = nltk.FreqDist(tokens)
                rarewords = list(freq_dist.keys())[-10:]
                new_words = [word for word in tokens if word not in rarewords]
                text = " ".join(new_words)

        # optional: stems text using Porter Stemmer
        if stem_text:
                stemmer = default_stemmer
                tokens = tokenize_text(text)
                text = " ".join([stemmer.stem(t) for t in tokens])

        # removes stop words such as "a", "the", etc.
        if remove_stopwords:
                stop_words = default_stopwords
                tokens = [w for w in tokenize_text(text) if w not in stop_words]
                text = " ".join(tokens)
        
        # optional: removes numbers completely from the ext
        if remove_num:
                text=text.split()
                text=[x for x in text if not x.isnumeric()]
                text= " ".join(text)

        return text

# load all the original unclean notes
f = open('PickleFiles/original_notes.pckl', 'rb')
old_notes = pickle.load(f)
f.close()

# load all the concepts
f = open('PickleFiles/notes_concepts.pckl', 'rb')
notes_concepts = pickle.load(f)
f.close()

start = timer()
notes = []
for i, note in enumerate(old_notes): # takes --- seconds to go through the cleaning for-loop for all notes
        t.update(i)
        notes.append(clean_text(i, note, notes_concepts, remove_punctuation = True, remove_stopwords = True, remove_repeat = True, spell_check=True))
        # notes.append(clean_text(note, remove_punctuation = True, remove_stopwords = True, spell_check = True, remove_repeat = True))
end = timer()
print(end - start)
print("Ended cleaning progress")

# save cleaned notes into a pickle file
f = open('PickleFiles/cleaned_notes.pckl', 'wb')
pickle.dump(notes, f)
f.close()
print("Saved cleansed notes")

# take around 1 - 1.5 hours to complete for 782 files
print("Done cleaning and saving within " + str(end-start) + " seconds")
t.finish()