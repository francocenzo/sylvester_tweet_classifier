# -*- coding:utf-8 -*-
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from sylvester.preprocessing import preprocess_sentence


def get_data(file_path):

    # header = 0 -> zero-based index
    data_file = pd.read_csv(file_path, header=0, sep=';', nrows=100)
    # debug info to show that the data is imported:
    print(data_file.shape)
    print(data_file.head())

    mk = []
    pendant = []

    value_counter = 0
    # sort into to arrays for each label
    for value in data_file.values:
        mk_sentence = preprocess_sentence(value[0])
        mk.append(mk_sentence)
        pendant_sentence = preprocess_sentence(value[1])
        pendant.append(pendant_sentence)
        if value_counter % 50:  # modulo is true if / 60 has no rest
            print("mk: ", mk_sentence)
            print("pendant: ", pendant_sentence)
        value_counter += 1

    # combine all data -> mk, mk, mk ... pendant, pendant, pendant
    data = mk + pendant
    # create a same sized array for the data labels
    label = len(mk) * ['MK'] + len(pendant) * ['PE']

    return data, label


def generate_stopwords(file_path) -> list:

    with open(file_path, encoding="utf-8") as open_file:
        content = open_file.readlines()

    stop_words = set()
    for word in content:
        stop_words.add(word.strip())
    return stop_words


def save_training(classifier, vectorizer, label_predicted, label_test, folder=False, file_name=False):

    folder = folder or "../training_sessions"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if file_name:
        file_name = f"{folder}/{file_name.rstrip('.pkl')}.pkl"
    else:
        file_name = f"{folder}/training-{timestamp}.pkl"

    data = [classifier, vectorizer, label_predicted, label_test]
    with open(file_name, "wb") as open_file:
        pickle.dump(data, open_file)

    return timestamp, file_name


def load_training(location):
    file_path = find_training_file(location)

    with open(file_path, 'rb') as open_file:
        classifier, vectorizer, label_predicted, label_test = pickle.load(open_file)

    return classifier, vectorizer, label_predicted, label_test


def find_training_file(user_input=False):
    timestamp = False
    file_path = False

    user_input = user_input or input("file path or timestamp in default location: ")

    if Path(user_input).is_dir():
        folder = Path(user_input) / f"training-{timestamp}.pkl"
        if folder.is_file():
            file_path = folder

    elif Path(user_input).is_file():
        file_path = user_input

    elif Path(f"../training_sessions/{user_input}").is_file():
        file_path = Path(f"../training_sessions/{user_input}")

    if not file_path:
        raise FileNotFoundError(f"Could not find file. (user input: {user_input})")

    return file_path


if __name__ == "__main__":
    from pathlib import Path

    file_path = sorted([file for file in Path().glob("../data/*.csv")])[0]
    data, label = get_data(file_path)

    file_path = sorted([file for file in Path().glob("../data/stopwords-de.txt")])[0]
    stop_words = generate_stopwords(file_path)

    training_file = sorted([file for file in Path().glob("../training_sessions/*.pkl")])[0]
    classifier, vectorizer, label_predicted, label_test = load_training(training_file)

    save_training(classifier, vectorizer, label_predicted, label_test, file_name="test.pkl")

