import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Common Preprocessing
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stop words
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in set(stopwords.words('english'))]

    # Lemmatization
    words = [WordNetLemmatizer().lemmatize(word) for word in words]
    return ' '.join(words)


# 5 folds cross validation to tune parameter
def parameterTuning(classifier, param_grid, X_train, y_train):
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("-----------------------------------------")
    print("Best parameter:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    return grid_search.best_estimator_


# Classify the corpus using three linear classifiers
def classification(X, y):
    print("===================================================")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    # Use force_alpha=False to prevent Underflow
    best_nb = parameterTuning(MultinomialNB(),
                              {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}, X_train, y_train)
    y_pred_nb = best_nb.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

    # Logistic Regression classifier
    best_lr = parameterTuning(LogisticRegression(max_iter=100000),
                              {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}, X_train, y_train)
    y_pred_lr = best_lr.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

    # SVM classifier
    best_svm = parameterTuning(SVC(kernel='linear'),
                               {'C': [0.1, 1, 10]}, X_train, y_train)
    y_pred_svm = best_svm.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("===================================================")
    print("")


with open("fakes.txt", "r") as file:
    fakes = file.readlines()

with open("facts.txt", "r") as file:
    facts = file.readlines()

fakes_processed = [preprocess(fake) for fake in fakes]
facts_processed = [preprocess(fact) for fact in facts]
corpus_processed = fakes_processed + facts_processed
y = [0] * len(fakes_processed) + [1] * len(facts_processed)

# Special Preprocessing: Feature extraction using N-gram
# N=1
vectorized0 = CountVectorizer(ngram_range=(1, 1), analyzer='word')
X0 = vectorized0.fit_transform(corpus_processed)
print("Count of 1-gram: ", len(vectorized0.get_feature_names_out()))
classification(X0, y)

# N=2
vectorized1 = CountVectorizer(ngram_range=(2, 2), analyzer='word')
X1 = vectorized1.fit_transform(corpus_processed)
print("Count of 2-gram: ", len(vectorized1.get_feature_names_out()))
classification(X1, y)

# N=1 and N=2
vectorized2 = CountVectorizer(ngram_range=(1, 2), analyzer='word')
X2 = vectorized2.fit_transform(corpus_processed)
print("Count of 1-gram and 2-gram: ", len(vectorized2.get_feature_names_out()))
classification(X2, y)
