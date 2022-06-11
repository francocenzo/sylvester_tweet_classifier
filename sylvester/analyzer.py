# -*- coding:utf-8 -*-
from collections import Counter
from sklearn import metrics as metrics


def analyze(classifier, vectorizer):
    # analyze features
    # inverse vocabulary from term -> id, to id -> term
    vocabulary = {value: key for key, value in vectorizer.vocabulary_.items()}
    num_results = 10
    # map arrays with [id] = frequency to dict id -> frequency and return most frequent
    feature_count_mk = {i: classifier.feature_count_[0][i] for i in range(len(classifier.feature_count_[0]))}
    feature_count_mk = Counter(feature_count_mk)
    mk_most_common = feature_count_mk.most_common(num_results)
    print("most common mediacriticism")
    for i in mk_most_common:
        print(vocabulary[i[0]], " :", i[1], " ")

    feature_count_pendant = {i: classifier.feature_count_[1][i] for i in range(len(classifier.feature_count_[1]))}
    feature_count_pendant = Counter(feature_count_pendant)
    pendant_most_common = feature_count_pendant.most_common(num_results)
    print("most common pendants")
    for i in pendant_most_common:
        print(vocabulary[i[0]], " :", i[1], " ")

    # map arrays with [id] = probability to dict id -> probability
    feature_probability_mk = {i: classifier.feature_log_prob_[0][i] for i in
                              range(len(classifier.feature_log_prob_[0]))}
    feature_probability_mk = Counter(feature_probability_mk)
    mk_most_probable = feature_probability_mk.most_common(num_results)
    print("most probable mediacriticism")
    for i in mk_most_probable:
        print(vocabulary[i[0]], " :", i[1], " ")


    feature_probability_pendant = {i: classifier.feature_log_prob_[1][i] for i in
                                   range(len(classifier.feature_log_prob_[1]))}
    feature_probability_pendant = Counter(feature_probability_pendant)
    pendant_most_probable = feature_probability_pendant.most_common(num_results)
    print("most probable pendants")
    for i in pendant_most_probable:
        print(vocabulary[i[0]], " :", i[1], " ")
    # check with any sentence
    # change the following to a for loop that iterates over your data that you want to test
    return mk_most_common, pendant_most_common, mk_most_probable, pendant_most_probable


def calculate_metrics(label_predicted, label_test):
    accuracy = metrics.accuracy_score(label_test, label_predicted)
    print("accuracy:\t", accuracy)
    precision_mk = metrics.precision_score(label_test, label_predicted, pos_label='MK')
    print("precision:\t", precision_mk)
    recall = metrics.recall_score(label_test, label_predicted, pos_label='MK')
    print("recall:\t", recall)
    f1 = metrics.f1_score(label_test, label_predicted, pos_label='MK')
    print("f1:\t", f1)
    return accuracy, precision_mk, recall, f1


if __name__ == "__main__":
    from sylvester.data import load_training
    from pathlib import Path

    training_file = sorted([file for file in Path().glob("../training_sessions/*.pkl")])[0]

    print(f"Analyzing {training_file}")
    classifier, vectorizer, label_predicted, label_test = load_training(training_file)

    accuracy, precision_mk, recall, f1 = calculate_metrics(label_predicted, label_test)
    mk_most_common, pendant_most_common, mk_most_probable, pendant_most_probable = analyze(classifier, vectorizer)
