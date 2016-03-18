import csv, itertools, nltk
import numpy as np
from config import *
from pprint import pprint

def load_data(filename='../data/reddit-comments-2015-08.csv', vocabulary_size = 8000, sentences_count = None):
# Read the data and append SENTENCE_START and SENTENCE_END tokens
    with open(filename, 'rb') as f:
        if 'csv' in filename:
            print "Reading CSV file..."
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        else:
            sentences = [x.strip() for x in f if len(x) > 4]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    pprint(sentences[:10]) 
    if sentences_count:
        sentences = sentences[:sentences_count]
    print "Parsed %d sentences." % (len(sentences))

# Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
    print "\nExample sentence after word_to_index: \n\tx = %r\n\ty= %r " % (X_train[0], y_train[0])

    return X_train, y_train, index_to_word

if __name__ == '__main__':
    load_data()
