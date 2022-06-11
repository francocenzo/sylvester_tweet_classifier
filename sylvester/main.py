# -*- coding:utf-8 -*-
import time

from sylvester.analyzer import analyze
from sylvester.data import load_training, find_training_file
from sylvester.classify import classify_item
from sylvester.training import train_on_corpus


def train_corpus():
    corpus = input("Full path to corpus: ")

    if len(corpus) == 0:
        corpus = "../data/Tweets_1-1200.csv"

    runs = input("How man times: ")
    try:
        runs = int(runs)
    except:
        runs = 1

    save = input("Save trainings? (Y/n)")
    if save.lower() != "n":
        save = True
    else:
        save = False

    save_feature = input("Save features? (Y/n)")
    if save_feature.lower() != "n":
        save_feature = True
    else:
        save_feature = False

    results = []
    for run in range(runs):
        time.sleep(1)
        classifier, vectorizer, label_predicted, label_test, accuracy, precision_mk, recall, f1, timestamp, file_name = train_on_corpus(corpus, save=save, save_feature=save_feature)
        results.append({
            "classifier": classifier,
            "vectorizer": vectorizer,
            "label_predicted": label_predicted,
            "label_test": label_test,
            "accuracy": accuracy,
            "precision_mk": precision_mk,
            "recall": recall,
            "f1": f1,
            "timestamp": timestamp,
            "file_name": file_name
        })

    print(f"\nTraining complete. Saved:")
    for idx, item in enumerate(results):
        print(f'Training run {idx + 1}: {item["file_name"]}')


    avg_accuracy = sum(item["accuracy"] for item in results) / runs
    avg_precision_mk = sum(item["precision_mk"] for item in results) / runs
    avg_recall = sum(item["recall"] for item in results) / runs
    avg_f1 = sum(item["f1"] for item in results) / runs
    print(f"\nAverages of {runs} runs:")
    print(f"avg accuracy: {avg_accuracy}")
    print(f"avg precision_mk: {avg_precision_mk}")
    print(f"avg recall: {avg_recall}")
    print(f"avg f1: {avg_f1}")


def analyze_training():
    file_path = find_training_file()
    classifier, vectorizer, label_predicted, label_test = load_training(file_path)
    analyze(classifier, vectorizer)


def print_results(results):
    print("Summary:")
    for result in results:
        user_sentence, prediction, probability = result
        print(f"\n{user_sentence}\n\tPrediction: {prediction}\n\tProbability: {probability}\n")


def classify_user_input():
    classifier, vectorizer, label_predicted, label_test = load_training(find_training_file())

    results = []
    user_input = input("\n Classify a sentence? (Y/n)")

    while user_input.lower() != "n":
        user_sentence = input("Enter Sentence: ")
        prediction, probability = classify_item(user_sentence, classifier, vectorizer)
        results.append([user_sentence, prediction, probability])
        print(f"Prediction: {prediction}, Probability: {probability}")
        user_input = input("\n Classify another sentence? (Y/n)")

    print_results(results)


def classify_txt():
    classifier, vectorizer, label_predicted, label_test = load_training(find_training_file())
    user_input = input("Full path to corpus txt file (one item per line): ")

    with open(user_input, encoding="utf-8") as corpus:
        items = corpus.readlines()

    results = []
    for item in items:
        prediction, probability = classify_item(item, classifier, vectorizer)
        results.append([item, prediction, probability])
    print_results(results)


def run():
    options = [
        ["Train Corpus", train_corpus],
        ["Analyze Training", analyze_training],
        ["Classify input", classify_user_input],
        ["Classify TXT-File", classify_txt]
    ]

    for idx, item in enumerate(options):
        print(f"[{idx}] {item[0]}")

    task_idx = int(input("Select task: "))
    task = options[task_idx][1]
    task()


if __name__ == "__main__":
    # training-2022-06-10_19-06-38.pkl
    # ../data/Tweets_1-1200.csv
    run()
