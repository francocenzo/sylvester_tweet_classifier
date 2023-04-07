import spacy
from nltk import SnowballStemmer


def ask_snowball(loaded_snowball_stemmer):
    """
    blueprint
    """
    snowball_stemmer = input("Use Snowball Stemmer? (Y/n)")
    if snowball_stemmer.lower() != "n":
        snowball_stemmer = True
    else:
        snowball_stemmer = False

    if snowball_stemmer:
        if not loaded_snowball_stemmer:
            snowball_stemmer = SnowballStemmer("german")
            loaded_snowball_stemmer = snowball_stemmer
        else:
            snowball_stemmer = loaded_snowball_stemmer
    return snowball_stemmer


def ask_spacy(loaded_spacy_model):
    spacy_model = input("Use Spacy Lemmatizer? (Y/n)")
    if spacy_model.lower() != "n":
        spacy_model = True
    else:
        spacy_model = False

    if spacy_model:
        if not loaded_spacy_model:
            spacy_model = spacy.load('de_core_news_sm', disable=['parser', 'ner'])
            loaded_spacy_model = spacy_model
        else:
            spacy_model = loaded_spacy_model
    return spacy_model


def ask_save_feature():
    save_feature = input("Save features? (Y/n)")
    if save_feature.lower() != "n":
        save_feature = True
    else:
        save_feature = False
    return save_feature


def ask_save_trainings():
    save = input("Save trainings? (Y/n)")
    if save.lower() != "n":
        save = True
    else:
        save = False
    return save


def ask_run_number():
    runs = input("How man times: ")
    try:
        runs = int(runs)
    except:
        runs = 1
    return runs


def ask_corpus_path():
    corpus = input("Full path to corpus: ")
    return corpus


def ask_use_stopwords():
    pass
