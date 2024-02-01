from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn import svm
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import time
import numpy as np

GROUND_TRUTH_PATH = 'data/DBLPTestGroundTruth.txt'
TRAIN_PATH = 'data/DBLPTrainset.txt'
TEST_PATH = 'data/DBLPTestset.txt'
PERFORMANCE_DF_HEADERS = ['Vectorizer', 'Pre-processing', 'Model', 'Total Time', 'Model Time', 'Precision',
                          'Recall', 'F1 Score', 'Confusion Matrix']
NUMBER_OF_FOLDS = 5
RANDOM_STATE = 42
UNDERSAMPLE = True
OVERSAMPLE = False


def load_file(path: str) -> list:
    with open(path, 'r') as f:
        return [[y.strip() for y in x.split('\t')] for x in f.readlines()]


def pre_process_data(data: list, progress: bool = True) -> tuple:
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    st = []
    lt = []
    ns = []
    stns = []
    ltns = []
    l = []

    # Loop through each title in the training set
    iterable = tqdm(data, desc="Pre-processing data") if progress else data
    for title in iterable:
        if len(title) == 2:
            conference, paper_title = title
        elif len(title) == 3:
            conference, paper_title = title[1:]
        else:
            raise ValueError(f'Expected 2 or 3 columns, got {len(title)}')
        tokens = word_tokenize(paper_title)
        stemmed_title = " ".join([stemmer.stem(token) for token in tokens])
        lemmatized_title = " ".join([lemmatizer.lemmatize(token) for token in tokens])
        no_stop_words_title = " ".join([token for token in tokens if token not in stop_words])
        stemmed_no_stop_words_title = " ".join([stemmer.stem(token) for token in tokens if token not in stop_words])
        lemmatized_no_stop_words_title = " ".join(
            [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words])
        st.append(stemmed_title)
        lt.append(lemmatized_title)
        ns.append(no_stop_words_title)
        stns.append(stemmed_no_stop_words_title)
        ltns.append(lemmatized_no_stop_words_title)
        l.append(conference)

    return st, lt, ns, stns, ltns, l


def calculate_tp_fp_fn_tn_fractions(conf_matrix):
    total = np.sum(conf_matrix)
    TP = np.diag(conf_matrix) / total
    FP = (np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)) / total
    FN = (np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)) / total
    TN = total - (FP + FN + np.diag(conf_matrix)) / total

    return TP, FP, FN, TN


def train_model(training_data: list, training_labels: list, training_model) -> tuple:
    start_time = time.time()
    kf = KFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    precision_scores, recall_scores, f1_scores, conf_matrices = [], [], [], []

    # Ensure training_data is a 2D array or a sparse matrix
    if isinstance(training_data, list):
        training_data = np.array(training_data)
    if isinstance(training_labels, list):
        training_labels = np.array(training_labels)

    for train_index, test_index in kf.split(training_data):
        X_train, X_test = training_data[train_index], training_data[test_index]
        y_train, y_test = training_labels[train_index], training_labels[test_index]

        training_model.fit(X_train, y_train)
        y_pred = training_model.predict(X_test)

        precision_scores.append(precision_score(y_test, y_pred, average='micro'))
        recall_scores.append(recall_score(y_test, y_pred, average='micro'))
        f1_scores.append(f1_score(y_test, y_pred, average='micro'))

        conf_matrices.append(confusion_matrix(y_test, y_pred))

    end_time = time.time()
    duration = end_time - start_time
    average_duration = duration / NUMBER_OF_FOLDS

    aggregated_conf_matrix = np.sum(conf_matrices, axis=0)

    return (
        np.mean(precision_scores),
        np.mean(recall_scores),
        np.mean(f1_scores),
        aggregated_conf_matrix,
        average_duration,
        duration
    )


def oversample_data(data: list) -> list:
    # Convert data to DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=['Category', 'Text'])

    # Getting the maximum category size to match other categories' size
    max_size = df['Category'].value_counts().max()

    # Empty list to hold oversampled data
    oversampled_data = []

    with tqdm(total=len(df.groupby('Category')), desc="Oversampling data") as pbar:
        for category, group in df.groupby('Category'):
            oversampled_group = resample(group,
                                         replace=True,  # Sample with replacement
                                         n_samples=max_size,  # Match the majority class
                                         random_state=RANDOM_STATE)
            oversampled_data.append(oversampled_group)
            pbar.update(1)

    # Concatenate the oversampled dataframes
    oversampled_df = pd.concat(oversampled_data)

    # Convert the DataFrame back to a list of lists
    oversampled_list = list(oversampled_df.values)

    return oversampled_list


def undersample_data(data: list) -> list:
    # Convert data to DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=['Category', 'Text'])

    # Getting the minimum category size to match other categories' size
    min_size = df['Category'].value_counts().min()

    # Empty list to hold undersampled data
    undersampled_data = []

    with tqdm(total=len(df.groupby('Category')), desc="Undersampling data") as pbar:
        for category, group in df.groupby('Category'):
            undersampled_group = resample(group,
                                          replace=False,
                                          n_samples=min_size,
                                          random_state=RANDOM_STATE)
            undersampled_data.append(undersampled_group)
            pbar.update(1)

    # Concatenate the undersampled dataframes
    undersampled_df = pd.concat(undersampled_data)

    # Convert the DataFrame back to a list of lists
    undersampled_list = list(undersampled_df.values)

    return undersampled_list



if __name__ == '__main__':

    with tqdm(total=4, desc="Loading data") as pbar:
        test_file = load_file(TEST_PATH)
        test_data = [x[1] for x in test_file]
        pbar.update(1)
        train_file = load_file(TRAIN_PATH)
        train_data = [[x[1], x[2]] for x in train_file]
        pbar.update(1)
        ground_truth_file = load_file(GROUND_TRUTH_PATH)
        ground_truth_data = [x[1] for x in ground_truth_file]
        pbar.update(1)
        zipped_test_data = [[x, y] for x, y in zip(ground_truth_data, test_data)]
        train_data.extend(zipped_test_data)
        pbar.update(1)

    if OVERSAMPLE:
        train_data = oversample_data(train_data)
    elif UNDERSAMPLE:
        train_data = undersample_data(train_data)

    stemmed_titles, lemmatized_titles, no_stop_words_titles, stemmed_no_stop_words, lemmatized_no_stop_words, labels = \
        pre_process_data(train_data)

    # Total number of iterations for the progress bar
    performance_df = pd.DataFrame(columns=PERFORMANCE_DF_HEADERS)
    total_iterations = 3 * 5 * 6
    trained_models = []
    with tqdm(total=total_iterations, desc="Training models") as pbar:
        start_time = time.time()
        vectorizers = {
            "term-frequency": TfidfVectorizer(),
            'count': CountVectorizer(),
            'hashing': HashingVectorizer(n_features=10000, norm=None, alternate_sign=False)
        }
        for vectorizer_name in vectorizers:
            vectorizer = vectorizers[vectorizer_name]
            a, b, c, d, e, _ = pre_process_data(test_file, progress=False)
            pre_processing_methods = {
                "stemmed": [vectorizer.fit_transform(stemmed_titles), vectorizer.transform(a)],
                "lemmatized": [vectorizer.fit_transform(lemmatized_titles), vectorizer.transform(b)],
                "no-stop-words": [vectorizer.fit_transform(no_stop_words_titles), vectorizer.transform(b)],
                "stemmed-no-stop-words": [vectorizer.fit_transform(stemmed_no_stop_words), vectorizer.transform(b)],
                "lemmatized-no-stop-words": [vectorizer.fit_transform(lemmatized_no_stop_words),
                                             vectorizer.transform(b)],
            }
            for pre_processing_method_name in pre_processing_methods:
                pre_processing_method = pre_processing_methods[pre_processing_method_name][0]
                models = {
                    "multi-nomial": MultinomialNB(),
                    'logistic-regression-lbsfg': LogisticRegression(solver='lbfgs', max_iter=3000),
                    'logistic-regression-liblinear': LogisticRegression(solver='liblinear', max_iter=3000),
                    'logistic-regression-saga': LogisticRegression(solver='saga', max_iter=3000),
                    'svm': svm.SVC(),
                    'random-forest': RandomForestClassifier(),
                }
                for model_name in models:
                    model = models[model_name]
                    model_micro_averaged_precision, model_micro_averaged_recall, model_micro_f1_score, \
                    aggregated_conf_matrix, model_average_duration, model_duration = train_model(pre_processing_method,
                                                                                                 labels,
                                                                                                 model)
                    new_row = pd.DataFrame([[vectorizer_name, pre_processing_method_name, model_name, model_duration,
                                             model_average_duration, model_micro_averaged_precision,
                                             model_micro_averaged_recall, model_micro_f1_score,
                                             aggregated_conf_matrix]],
                                           columns=PERFORMANCE_DF_HEADERS)
                    # Concatenate the original DataFrame with the new row
                    performance_df = pd.concat([performance_df, new_row], ignore_index=True)
                    pbar.update(1)
    print("Finished!")
