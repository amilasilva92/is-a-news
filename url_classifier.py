import streamlit as st
import numpy as np
import pickle
import time
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from newspaper import Article
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

nltk.download('stopwords')
nltk.download('punkt')

class URL_Classifier:
    def __init__(self):
        self.negative_domains = []
        self.positive_domains = []
        self.load_predefined_domains()

        self.classifier = None
        self.model_filename = 'pretrained_classifier.sav'
        # self.classifier = self.train_classifier()
        self.load_classifier()

    def load_predefined_domains(self):
        f = open('negative_domains.txt', 'r')
        for line in f:
            self.negative_domains.append(line.strip())
        f.close()
        # print("{} negative domains are loaded".format(len(self.negative_domains)))

        f = open('positive_domains.txt', 'r')
        for line in f:
            self.positive_domains.append(line.strip())
        f.close()
        # print("{} positive domains are loaded".format(len(self.positive_domains)))

    def check_in_existing_domains(self, url):
        for negative_domain in self.negative_domains:
            if negative_domain in url:
                return 0

        for positive_domain in self.positive_domains:
            if positive_domain in url:
                return 1
        return -1

    def classify_url(self, url):
        result = self.check_in_existing_domains(url)
        if result == 1:
            return True
        elif result == 0:
            return False

        # print('url domain is not found')
        url_features = self.extract_url_features(url)
        if url_features is None:
            return False

        result = self.check_in_existing_domains(url_features[-1])
        if result == 1:
            return True
        elif result == 0:
            return False

        predicted_label = self.classifier.predict(np.array(url_features[:-1]).reshape(1,-1))
        return False

    def train_classifier(self):
        # load features and labels
        X = np.load('url_featuers.npy')
        y = np.load('url_labels.npy')

        # generate a balanced dataset
        balanced_X = X[y==1]
        balanced_y = y[y==1]
        no_samples = balanced_y.shape[0]
        balanced_X = np.concatenate((balanced_X, X[y==0][:no_samples]), axis = 0)
        balanced_y = np.concatenate((balanced_y, y[y==0][:no_samples]), axis = 0)

        # learn the machine learning model
        X_train, X_test, y_train, y_test = train_test_split(balanced_X, balanced_y, test_size = 0.2)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        a = model.score(X_test, y_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, model.predict(X_test))
        print("evaluation results:")
        print("acc: {}, pre: {}, rec: {}, f1: {}".format(a, p, r, f))

        # save the model
        pickle.dump(model, open(self.model_filename, 'wb'))
        return model

    def load_classifier(self):
        self.classifier = pickle.load(open(self.model_filename, 'rb'))

    def extract_url_features(self, url):
        article = self.crawl_article(url)
        if article is None:
            return None

        # print('url is successfully crawled')
        # generate link count
        soup = BeautifulSoup(article.html, 'lxml')
        link_count = len(soup.find_all('a'))

        # generate word count
        text = article.text
        text = text.replace('\t', ' ').replace('\n', ' ')
        word_count = len(word_tokenize(text))
        sent_count = len(sent_tokenize(text))

        title = article.title
        title = title.replace('\t', ' ').replace('\n', ' ')
        title_word_count = len(word_tokenize(title))

        # if url is shortened
        try:
            r = requests.get(url)
            url_l = r.url
        except:
            url_l = url

        # print(word_count, sent_count, title_word_count, link_count, len(url_l), url_l)
        return [word_count, sent_count, title_word_count, link_count, len(url_l), url_l]

    def crawl_article(self, url):
        try:
            article = Article(url)
            article.download()
            time.sleep(2)

            article.parse()
            article.nlp()
            return article
        except:
            return None


classifier = URL_Classifier()
st.write(""" #  Is a News?
This application aims to ***automatically discover news URLs based on their content and a predefined URL database***, which could be useful for various downstream applications such as online misinformation detection and news domain identification""")

url = st.text_input("Enter URL", "Type Here...")
result = classifier.classify_url(url)

if url != "Type Here..." and url != "":
    if result:
        st.info("This is a News URL")
    else:
        st.info("This is not a News URL")

from PIL import Image
img = Image.open("news_background.png")
st.image(img,width=700)
