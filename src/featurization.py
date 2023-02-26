import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from mlem.api import save


def get_df(data):
    df = pd.read_csv(
        data,
        encoding="utf-8",
        header=None,
        delimiter="\t",
        names=["id", "label", "text"],
    )
    sys.stderr.write(f"The input data frame {data} size is {df.shape}\n")
    return df


if __name__ == "__main__":

    params = yaml.safe_load(open("params.yaml"))["featurize"]

    np.set_printoptions(suppress=True)

    if len(sys.argv) != 3 and len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython featurization.py data-dir-path preprocessor-path\n")
        sys.exit(1)

    train_input = os.path.join(sys.argv[1], "train.tsv")
    test_input = os.path.join(sys.argv[1], "test.tsv")

    max_features = params["max_features"]
    ngrams = params["ngrams"]

    # Generate train feature matrix
    df_train = get_df(train_input)
    train_words = np.array(df_train.text.str.lower().values.astype("U"))

    bag_of_words = CountVectorizer(
        stop_words="english", max_features=max_features, ngram_range=(1, ngrams)
    )

    bag_of_words.fit(train_words)
    train_words_binary_matrix = bag_of_words.transform(train_words)
    feature_names = bag_of_words.get_feature_names_out()
    tfidf = TfidfTransformer(smooth_idf=False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

    def preprocess_texts(texts):
        test_words = np.array([t.lower() for t in texts])
        test_words_binary_matrix = bag_of_words.transform(test_words)
        test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)
        return test_words_tfidf_matrix

    with open("feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

    # we save preprocessing 
    save(preprocess_texts, sys.argv[2], sample_data=df_train.text)
