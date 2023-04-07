# -*- coding:utf-8 -*-
import re
# python -m spacy download de_core_news_sm

import nltk
#from nltk.stem.snowball import GermanStemmer
#stemmer = GermanStemmer()


def spacy_lemma(sentence, load_model):

    doc = load_model(sentence)
    sentence = " ".join([token.lemma_ for token in doc])
    return sentence


def snowball_stemming(sentence, stemmer):
    # halt do was denn s andere model oder algo macha söll
    words = sentence.split()
    sentence = " ".join([stemmer.stem(word) for word in words])
    return sentence


def preprocess_sentence(sentence, spacy_model=False, snowball_stemmer=False):
    '''
    define all steps that we want to change on the used texts (train and input test) here in this method.
    '''

    # change to lower case
    result = str.lower(sentence)
    # remove "tweet by"
        #result = re.sub("(?:tweet by )", " ", result)
    # remove "Tweet by"
    #   result = re.sub("(?:Tweet by )", " ", result)


    # remove all http links
    result = re.sub("http[\S/]*", "", result)

    # remove all user mentions (@)
    #result = re.sub("@\w*", "", result)
    # remove all user mentions (@)
    result = re.sub("(?:@)", "", result)

    # remove all hashtags
    #result = re.sub("#\w*", "", result)
    # remove all hashtags
    result = re.sub("(?:#)", "", result)

    # remove RT mentions
    result = re.sub("rt(?=[\W$])(?<=[^\W])", "", result)

    # deal with leading/tailing whitespaces, as well as multi-whitespaces
    #result = re.sub("\s\s", " ", result.strip())

    # remove anything other than a letter, digit or underscore
    result = re.sub("\W+", " ", result)
    # deal #with single letters (standing alone)
    #result = re.sub("\b\w{0,1}\b", " ", result)
    # remove numbers
    result = re.sub("[\d-]", " ", result)

    result = re.sub("(?!^10vor10$) (\d+)", " ", result)
    #   (?! ^ 10vor10$)
    result = re.sub("(?!^10 vor 10$) (\d+)", " ", result)
    #   (?! ^ 10vor10$)

    # remove multiple white spaces
    result = re.sub(" +", " ", result)

    # remove one letter
    #result = re.sub("\b[a-zA-Z0-9]{1}\b", "", result)

    # remove two letters
   # result = re.sub("\b[a-zA-Z0-9]{2}\b", "", result)

    # remove three letters
    #result = re.sub("\b[a-zA-Z0-9]{3}\b", "", result)

    # diese verschlächtern den Classifier
    # remove all "Tweet by" mentions
    # result = re.sub("tweet by\w*", "", result)

    if spacy_model:
        result = spacy_lemma(result, spacy_model)

    #stemmer.stem(result)
    if snowball_stemmer:
        result = snowball_stemming(result, snowball_stemmer)

    return result



if __name__ == "__main__":
    result = preprocess_sentence("Mäh")