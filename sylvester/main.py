# -*- coding:utf-8 -*-
import time
from datetime import datetime

from sylvester.analyzer import analyze
from sylvester.data import load_training, find_training_file
from sylvester.classify import classify_item
from sylvester.menu import ask_snowball, ask_spacy, ask_save_feature, ask_save_trainings, ask_run_number, \
    ask_use_stopwords, ask_corpus_path
from sylvester.training import train_on_corpus
from pathlib import Path
import csv

loaded_spacy_model = None
loaded_snowball_stemmer = None
default_corpus = r"../data/Tweets_list2_1-1120.csv"


def train_corpus():
    global loaded_spacy_model
    global loaded_snowball_stemmer

    corpus = ask_corpus_path()
    if len(corpus) == 0:
        corpus = default_corpus
    runs = ask_run_number()

    save = ask_save_trainings()
    save_feature = ask_save_feature()
    spacy_model = ask_spacy(loaded_spacy_model)
    snowball_stemmer = ask_snowball(loaded_snowball_stemmer)
    use_stowords = ask_use_stopwords()

    results = run_training(corpus, runs, save, save_feature, snowball_stemmer, spacy_model, use_stopwords=use_stowords, ngram_range=[1, 1], test_size=0.25, rows=500)

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


def run_training(corpus, runs, save, save_feature, snowball_stemmer, spacy_model, use_stopwords=False, ngram_range=None, test_size=None, rows=None):

    results = []
    for run in range(runs):
        time.sleep(1)
        classifier, vectorizer, label_predicted, label_test, accuracy, \
            precision_mk, recall, f1, timestamp, file_name = train_on_corpus(
            corpus,
            save=save,
            save_feature=save_feature,
            spacy_model=spacy_model,
            snowball_stemmer=snowball_stemmer,
            use_stopwords=use_stopwords,
            ngram_range=ngram_range,
            test_size=test_size,
            rows=rows
        )

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
    return results


def analyze_training():
    file_path = find_training_file()
    classifier, vectorizer, label_predicted, label_test = load_training(file_path)
    analyze(classifier, vectorizer)


def print_results(results):
    print("Summary:")
    save_output = input("Save output (Y/n): ")
    default_output_path = Path(r"C:\Users\vince\Documents\Vincenzo\Diss\Python classifier\Python_code\sylvester_tweet_classifier\output")
    output_file = default_output_path / f"output_{str(datetime.now()).replace(':', '-')}.csv"

    for result in results:
        user_sentence, prediction, probability = result
        print_string = f"\n{user_sentence}\n\tPrediction: {prediction}\n\tProbability: {probability}\n"
        csv_string = str(user_sentence).replace("\n", " ")
        output_string = f'"{csv_string}",{prediction},{probability}\n'
        print(print_string)
        if save_output.lower() == "y":
            with open(output_file, "a+", encoding="utf8") as open_file:
                open_file.write(output_string)


def classify_user_input():
    classifier, vectorizer, label_predicted, label_test = load_training(find_training_file())

    results = []
    user_input = input("\n Classify a sentence? (Y/n)")
    spacy_model = ask_spacy()
    snowball_stemmer = ask_snowball()

    while user_input.lower() != "n":
        user_sentence = input("Enter Sentence: ")
        prediction, probability = classify_item(user_sentence, classifier, vectorizer, spacy_model=spacy_model, snowball_stemmer=snowball_stemmer)
        results.append([user_sentence, prediction, probability])
        print(f"Prediction: {prediction}, Probability: {probability}")
        user_input = input("\n Classify another sentence? (Y/n)")

    print_results(results)


def classify_txt():
    classifier, vectorizer, label_predicted, label_test = load_training(find_training_file())
    user_input = input("Full path to corpus txt file (one item per line): ")

    with open(user_input, encoding="utf-8") as corpus:
        items = corpus.readlines()

    spacy_model = ask_spacy()
    snowball_stemmer = ask_snowball()

    results = []
    for item in items:
        prediction, probability = classify_item(item, classifier, vectorizer, spacy_model=spacy_model, snowball_stemmer=snowball_stemmer)
        results.append([item, prediction, probability])
    print_results(results)


def run():
    options = [
        ["Train Corpus", train_corpus],
        ["Analyze Training", analyze_training],
        ["Classify0 input", classify_user_input],
        ["Classify TXT-File", classify_txt],
        ["Bulk Analyze"]
    ]

    for idx, item in enumerate(options):
        print(f"[{idx}] {item[0]}")

    task_idx = int(input("Select task: "))
    task = options[task_idx][1]
    task()


def bulk_run():
    global loaded_spacy_model
    global loaded_snowball_stemmer

    corpus = ask_corpus_path()
    if len(corpus) == 0:
        corpus = default_corpus

    runs = ask_run_number()
    save = ask_save_trainings()
    save_feature = ask_save_feature()
    spacy_model = ask_spacy(loaded_spacy_model)

    snowball_stemmer = False
    if not spacy_model:
        snowball_stemmer = ask_snowball(loaded_snowball_stemmer)

    print("Starting\n")

    stopword_variations = [False, True]
    test_size_variations = [0.1, 0.2, 0.25, 0.3]
    test_size_variations = [0.1]
    ngram_variations = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]
    ngram_variations = [(1, 1)]
    rows_variations = list(range(50, 550, 50))
    rows_variations = [100, 200, 300]

    run_datetime = str(datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
    header = ["run_timestamp", "corpus", "use_stopwords", "test_size", "ngram", "rows", "runs", "stem_lemma", "avg_accuracy", "avg_precision_mk", "avg_recall", "avg_f1"]
    csv_output = []
    csv_output.append(header)

    if spacy_model:
        stem_lem = "Spacy Lemmatizer"
    elif snowball_stemmer:
        stem_lem = "NLTK Snowball Stemmer"
    else:
        stem_lem = "None"

    overall_start = datetime.now()
    for use_stopwords in stopword_variations:
        for test_size in test_size_variations:
            for ngram in ngram_variations:
                timestamp = str(datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
                print("\n")
                print(f"Timestamp: {timestamp}")
                print(f"Stopwords: {use_stopwords}")
                print(f"Stem or Lem: {stem_lem}")
                print(f"NGram: {ngram[0]}-{ngram[1]}-gram")
                print(f"Corpus/Test: {int(100 - (100 * test_size))}/{int(100 * test_size)}")

                for rows in rows_variations:

                    results = run_training(
                        corpus,
                        runs,
                        save,
                        save_feature,
                        snowball_stemmer,
                        spacy_model,
                        use_stopwords=use_stopwords,
                        ngram_range=ngram,
                        test_size=test_size,
                        rows=rows)

                    avg_accuracy = sum(item["accuracy"] for item in results) / runs
                    avg_precision_mk = sum(item["precision_mk"] for item in results) / runs
                    avg_recall = sum(item["recall"] for item in results) / runs
                    avg_f1 = sum(item["f1"] for item in results) / runs

                    run_output = [
                        str(run_datetime), str(corpus), str(use_stopwords), str(test_size), str(ngram), str(rows), str(runs),
                        str(type(stem_lem)), str(avg_accuracy), str(avg_precision_mk), str(avg_recall), str(avg_f1)
                    ]
                    csv_output.append(run_output)

                    print(f"{rows}\t{avg_accuracy}\t{avg_precision_mk}\t{avg_recall}\t{avg_f1}")

    file_name = f"Results_{run_datetime}.csv"
    with open(file_name, "w", encoding="utf8") as fo:
        csv_writer = csv.writer(fo, delimiter=";", quotechar='"')
        csv_writer.writerows(csv_output)

    print(f"Finished. Runtime: {datetime.now() - overall_start}")


if __name__ == "__main__":
    # training-2022-06-10_19-06-38.pkl
    # ../data/Tweets_1-1200.csv
    # run()
    bulk_run()
