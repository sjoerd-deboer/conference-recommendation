from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from tqdm import tqdm
import time

GROUND_TRUTH_PATH = 'data/DBLPTestGroundTruth.txt'
TRAIN_PATH = 'data/DBLPTrainset.txt'
TEST_PATH = 'data/DBLPTestset.txt'
PERFORMANCE_DF_HEADERS = ['Vectorizer', 'Pre-processing', 'Model', 'Total Time', 'Model Time', 'Precision',
                          'Recall', 'F1 Score']
NUMBER_OF_FOLDS = 5


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


def train_model(training_data: list, training_labels: list, training_model) -> tuple:
    start_time = time.time()
    kf = KFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=42)

    # Define the scoring metrics
    scoring = {
        'precision': make_scorer(precision_score, average='micro'),
        'recall': make_scorer(recall_score, average='micro'),
        'f1_score': make_scorer(f1_score, average='micro')
    }

    # Perform cross validation
    cv_results = cross_validate(training_model, training_data, training_labels, cv=kf, scoring=scoring)

    precision, recall, score = cv_results['test_precision'], cv_results['test_recall'], cv_results['test_f1_score']

    end_time = time.time()
    duration = end_time - start_time
    average_duration = duration / NUMBER_OF_FOLDS

    # training_model.fit(training_data, training_labels)
    return sum(precision) / NUMBER_OF_FOLDS, sum(recall) / NUMBER_OF_FOLDS, sum(
        score) / NUMBER_OF_FOLDS, average_duration, duration


def evaluate_model(evaluating_model, x: list, y: list) -> tuple:
    start_time = time.time()
    y_pred = evaluating_model.predict(x)
    end_time = time.time()
    duration = end_time - start_time

    # Calculate the micro averaged precision
    micro_averaged_precision = precision_score(y, y_pred, average='micro')
    # Calculate the micro averaged recall
    micro_averaged_recall = recall_score(y, y_pred, average='micro')
    # Calculate the f1 score
    micro_f1_score = f1_score(y, y_pred, average='micro')

    return micro_averaged_precision, micro_averaged_recall, micro_f1_score, duration


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

    stemmed_titles, lemmatized_titles, no_stop_words_titles, stemmed_no_stop_words, lemmatized_no_stop_words, labels = \
        pre_process_data(train_file)

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
                        model_average_duration, model_duration = train_model(pre_processing_method, labels, model)
                    new_row = pd.DataFrame([[vectorizer_name, pre_processing_method_name, model_name, model_duration,
                                             model_average_duration, model_micro_averaged_precision,
                                             model_micro_averaged_recall, model_micro_f1_score]],
                                           columns=PERFORMANCE_DF_HEADERS)
                    # Concatenate the original DataFrame with the new row
                    performance_df = pd.concat([performance_df, new_row], ignore_index=True)
                    pbar.update(1)
    print("Finished!")
