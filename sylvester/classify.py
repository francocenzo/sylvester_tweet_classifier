from pathlib import Path

from sylvester.data import load_training
from sylvester.preprocessing import preprocess_sentence

def classify_item(item, classifier, vectorizer, spacy_model=False, snowball_stemmer=False):
    """
    building Naive Bayes Classifier
    """
    to_classify = preprocess_sentence(item, spacy_model=spacy_model, snowball_stemmer=snowball_stemmer)
    sentence_matrix = vectorizer.transform([to_classify])
    sentence_matrix = sentence_matrix.toarray()
    prediction = classifier.predict(sentence_matrix)
    probability = classifier.predict_proba(sentence_matrix)
    return prediction, probability


if __name__ == "__main__":

    item = "This is a test!"
    training_file = sorted([file for file in Path().glob("../training_sessions/*.pkl")])[0]
    classifier, vectorizer, label_predicted, label_test = load_training(training_file)
    prediction, probability = classify_item(item, classifier, vectorizer)