import os
import sys

import yaml
from sklearn.ensemble import RandomForestClassifier
from mlem.api import save, load

from featurization import get_df


params = yaml.safe_load(open("params.yaml"))["train"]

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py data preprocessor model\n")
    sys.exit(1)

input = sys.argv[1]
preprocessor_path = sys.argv[2]
output = sys.argv[3]
seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]

preprocessing = load(preprocessor_path)

df_train = get_df(os.path.join(input, "train.tsv"))
labels = df_train.label

sys.stderr.write("Input dataset len {}\n".format(len(df_train)))

clf = RandomForestClassifier(
    n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
)

x = preprocessing(df_train.text)
sys.stderr.write("X matrix size {}\n".format(x.shape))
sys.stderr.write("Y matrix size {}\n".format(labels.shape))

clf.fit(x, labels)

save(
    clf,
    output,
    preprocess=preprocessing,
    sample_data=["This is a sample text."]
)
