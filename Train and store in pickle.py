
__author__="rojin.varghese"
__date__ ="$Oct 8, 2013 2:15:50 PM$"

#import nltk.classify.util
#from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
#from nltk.classify import NaiveBayesClassifier
#from nltk.classify.util import accuracy
import pickle


def extract_words(text):
    '''
    here we are extracting features to use in our classifier. We want to pull all the words in our input
    porterstem them and grab the most significant bigrams to add to the mix as well.
    '''

    stemmer = PorterStemmer()

    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)

    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)

    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)

    result =  [stemmer.stem(x.lower()) for x in tokens if x not in stopwords.words('english') and len(x) > 1]
    return result


def word_feats(words):
    return dict([(word, True) for word in words])

#print "Training strated............"
#negids = movie_reviews.fileids('neg')
#posids = movie_reviews.fileids('pos')
#
#negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'Negative') for f in negids]
#posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'Positive') for f in posids]
#
#negcutoff = len(negfeats)*3/4
#poscutoff = len(posfeats)*3/4
#
#trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
#testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
#print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
#
#classifier = NaiveBayesClassifier.train(trainfeats)
#
#f = open('my_classifier.pickle', 'wb')
#print "Storing to file...."
#pickle.dump(classifier, f)
#f.close()
#print "Completed........"

f= open('my_classifier.pickle')
classifier = pickle.load(f)

print "Sentiment Analysis"
review = raw_input('Enter a movie review: \n')
tokens = word_feats(extract_words(review))
output = classifier.classify(tokens)
print "\nSentiment of the sentence is "+output
f.close()
