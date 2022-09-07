# -*- coding:utf-8 -*-
import re


def preprocess_sentence(sentence):
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
    result = re.sub("\W+", " ", result.strip())

    # remove

    return result


if __name__ == "__main__":
    result = preprocess_sentence("Mäh")