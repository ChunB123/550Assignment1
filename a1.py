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

with open("fakes.txt", "r") as file:
    fakes = file.readlines()

with open("facts.txt", "r") as file:
    facts = file.readlines()


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


# Classify the corpus using three linear classifiers
def classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    # Use force_alpha=False to prevent Underflow
    nb_classifier = MultinomialNB(force_alpha=False)
    nb_classifier.fit(X_train, y_train)
    y_pred_nb = nb_classifier.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

    # Logistic Regression classifier
    lr_classifier = LogisticRegression(max_iter=100000)
    lr_classifier.fit(X_train, y_train)
    y_pred_lr = lr_classifier.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

    # SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


fakes_processed = [preprocess(fake) for fake in fakes]
facts_processed = [preprocess(fact) for fact in facts]
corpus_processed = fakes_processed + facts_processed
y = [0] * len(fakes_processed) + [1] * len(facts_processed)

# Feature extraction using N-gram
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