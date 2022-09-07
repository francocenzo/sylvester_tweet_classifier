# -*- coding:utf-8 -*-
from datetime import datetime

from sklearn import naive_bayes as nb, model_selection as ms
from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.ensemble import RandomForestClassifier

from sylvester.analyzer import calculate_metrics
from sylvester.data import get_data, generate_stopwords, save_training


def get_classifier(train_data, train_labels, save=False, load=False):
    # change here if we want to use another naive bayes algorithm:
    # https: // scikit - learn.org / stable / modules / naive_bayes.html
    classifier = nb.BernoulliNB()
    # classifier = RandomForestClassifier()
    classifier.fit(train_data, train_labels)
    return classifier


def transform_to_matrix(data_test, data_train, stopwords, save_feature=False):
    # transform training data to matrices
    vectorizer = CountVectorizer(tokenizer=str.split, stop_words=stopwords, ngram_range=[1, 3])
    train_matrix = vectorizer.fit_transform(data_train)

    # Ausgabe der bzgl der letzten Frage:
    feature_names = list(vectorizer.get_feature_names_out())
    print("\n\nVectorizer")
    print(feature_names)

    if save_feature:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"../debugging/feature_names_{timestamp}.txt", "w", encoding="utf-8") as open_file:
            for feature in feature_names:
                open_file.write(feature + "\n")


    train_matrix = train_matrix.toarray()
    test_matrix = vectorizer.transform(data_test)
    test_matrix = test_matrix.toarray()
    return test_matrix, train_matrix, vectorizer


def train(corpus_path, stopword_path=False, save=False, folder=False, file_name=False, save_feature=False):
    # prepare data
    data, label = get_data(corpus_path)

    # split between training and test
    data_train, data_test, label_train, label_test = ms.train_test_split(data, label, test_size=0.1)

    # load stopwords
    stopwords = generate_stopwords(stopword_path or "../data/stopwords-de.txt")

    test_matrix, train_matrix, vectorizer = transform_to_matrix(data_test, data_train, stopwords, save_feature=save_feature)

    # train training
    classifier = get_classifier(train_matrix, label_train)

    # train eval
    label_predicted = classifier.predict(test_matrix)
    # TODO: get most probable sentences by using classifier.predict_proba  (see while loop below)

    if save:
        save_training(classifier, vectorizer, label_predicted, label_test, folder=folder, file_name=file_name)

    return classifier, vectorizer, label_predicted, label_test


def train_on_corpus(corpus, save=False, save_feature=False):
    classifier, vectorizer, label_predicted, label_test = train(corpus, save_feature=save_feature)
    accuracy, precision_mk, recall, f1 = calculate_metrics(label_predicted, label_test)

    timestamp, file_name = False, False
    if save:
        timestamp, file_name = save_training(classifier, vectorizer, label_predicted, label_test)

    return classifier, vectorizer, label_predicted, label_test, accuracy, precision_mk, recall, f1, timestamp, file_name


if __name__ == "__main__":
    corpus = '../data/Tweets_1-1200.csv'

    classifier, vectorizer, label_predicted, \
    label_test, accuracy, precision_mk, \
    recall, f1, timestamp, file_name = train_on_corpus(corpus, save=True)