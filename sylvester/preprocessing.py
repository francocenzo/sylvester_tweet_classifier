# -*- coding:utf-8 -*-
import re
#import nltk

#from nltk.stem.snowball import GermanStemmer
#german=GermanStemmer()

def preprocess_sentence(sentence, spacy_model=False, snowball_stemmer=False):
    '''
    define all steps that we want to change on the used texts (train and input test) here in this method.
    '''

    # change to lower case
    result = str.lower(sentence)
    # remove all http links
    result = re.sub("http[\S/]*", "", result)
    # remove all user mentions (@)
    result = re.sub("@\w*", "", result)
    # remove all hashtags
    result = re.sub("#\w*", "", result)
    # remove RT mentions
    result = re.sub("rt(?=[\W$])(?<=[^\W])", "", result)

    # deal with leading/tailing whitespaces, as well as multi-whitespaces
    result = re.sub("\s\s", " ", result.strip())

    # remove anything other than a letter, digit or underscore
    result = re.sub("\W+", " ", result)
    # deal with single letters (standing alone)
    result = re.sub("\b\w{0,1}\b", " ", result)
    # remove numbers
    #result = re.sub(" \d+", " ", result)

    result = re.sub("(?!^10vor10$) (\d+)", " ", result)
    #   (?! ^ 10vor10$)
    
    # remove

    #result =

    return result


if __name__ == "__main__":
    result = preprocess_sentence("Mäh")