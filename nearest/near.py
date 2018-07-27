import gensim
from nltk.tokenize import word_tokenize
import nltk
import csv
import numpy as np
nltk.download('punkt')
print(dir(gensim))

def generate_data(data_train, data_out):
    docs = []
    header = np.array(['InputSentence1','InputSentence2','InputSentence3',
        'InputSentence4','AnswerRightEnding','FakeEnding1','FakeEnding2','FakeEnding3','FakeEnding4','FakeEnding5','FakeEnding6'])
    with open(data_train, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            sentences = row[1:7]
            doc = []
            for sentence in sentences:
                for w in word_tokenize(sentence):
                    w_l = w.lower()
                    doc.append(w_l)
            docs.append(doc)

    dictionary = gensim.corpora.Dictionary(docs)

    print("Number of words in dictionary:", len(dictionary))

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    # print(corpus)

    tf_idf = gensim.models.TfidfModel(corpus)
#    print(tf_idf)
    # s = 0
    # for i in corpus:
    #     s += len(i)
    # print(s)

    sims = gensim.similarities.Similarity('/Users/liukangning/Downloads/raw_data/',tf_idf[corpus],
                                          num_features=len(dictionary))
    # print(sims)
    # print(type(sims))
    #
    # query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
    # print(query_doc)
    # query_doc_bow = dictionary.doc2bow(query_doc)
    # print(query_doc_bow)
    # query_doc_tf_idf = tf_idf[query_doc_bow]
    # print(query_doc_tf_idf)
    #
    # sims[query_doc_tf_idf]


    with open(data_train, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        doc_sent6 = []
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            sentence = row[6]
            doc_sent6.append(sentence)

    with open(data_train, 'r', encoding="ISO-8859-1") as inputfile, open(data_out, 'w', encoding="ISO-8859-1") as outputfile:
        reader = csv.reader(inputfile)
        writer = csv.writer(outputfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            sentences = row[1:7]
            doc = []
            for sentence in sentences:
                for w in word_tokenize(sentence):
                    w_l = w.lower()
                    doc.append(w_l)
            query_doc_bow = dictionary.doc2bow(doc)
            query_doc_tf_idf = tf_idf[query_doc_bow]
            likely = sims[query_doc_tf_idf]
            likely[idx-1] = 0
#            print(likely.shape)
            max_likely1 = np.argmax(likely)
            likely[max_likely1] = 0
            max_likely2 = np.argmax(likely)
            likely[max_likely2] = 0
            max_likely3 = np.argmax(likely)
            likely[max_likely3] = 0
            max_likely4 = np.argmax(likely)
            likely[max_likely4] = 0
            max_likely5 = np.argmax(likely)
            likely[max_likely5] = 0
            max_likely6 = np.argmax(likely)

#            print(max_likely)
            back = []
            back[0:5] = row[2:7]
            back.append(doc_sent6[max_likely1])
            back.append(doc_sent6[max_likely2])
            back.append(doc_sent6[max_likely3])
            back.append(doc_sent6[max_likely4])
            back.append(doc_sent6[max_likely5])
            back.append(doc_sent6[max_likely6])
            writer.writerow(back)

def append_data(data_train, data_out):
    docs = []
    header = np.array(['InputSentence1','InputSentence2','InputSentence3',
        'InputSentence4','AnswerRightEnding','FakeEnding1','FakeEnding2','FakeEnding3','FakeEnding4','FakeEnding5','FakeEnding6'])
    with open(data_train, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            sentences = row[1:7]
            doc = []
            for sentence in sentences:
                for w in word_tokenize(sentence):
                    w_l = w.lower()
                    doc.append(w_l)
            docs.append(doc)

    dictionary = gensim.corpora.Dictionary(docs)

    print("Number of words in dictionary:", len(dictionary))

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    # print(corpus)

    tf_idf = gensim.models.TfidfModel(corpus)
#    print(tf_idf)
    # s = 0
    # for i in corpus:
    #     s += len(i)
    # print(s)

    sims = gensim.similarities.Similarity('./',tf_idf[corpus],
                                          num_features=len(dictionary))
    # print(sims)
    # print(type(sims))
    #
    # query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
    # print(query_doc)
    # query_doc_bow = dictionary.doc2bow(query_doc)
    # print(query_doc_bow)
    # query_doc_tf_idf = tf_idf[query_doc_bow]
    # print(query_doc_tf_idf)
    #
    # sims[query_doc_tf_idf]


    with open(data_train, 'r') as f:
        reader = csv.reader(f)
        doc_sent6 = []
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            sentence = row[6]
            doc_sent6.append(sentence)

    with open(data_train, 'r') as inputfile, open(data_out, 'a') as outputfile:
        reader = csv.reader(inputfile)
        writer = csv.writer(outputfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            sentences = row[1:7]
            doc = []
            for sentence in sentences:
                for w in word_tokenize(sentence):
                    w_l = w.lower()
                    doc.append(w_l)
            query_doc_bow = dictionary.doc2bow(doc)
            query_doc_tf_idf = tf_idf[query_doc_bow]
            likely = sims[query_doc_tf_idf]
            likely[idx-1] = 0
#            print(likely.shape)
            max_likely1 = np.argmax(likely)
            likely[max_likely1] = 0
            max_likely2 = np.argmax(likely)
            likely[max_likely2] = 0
            max_likely3 = np.argmax(likely)
            likely[max_likely3] = 0
            max_likely4 = np.argmax(likely)
            likely[max_likely4] = 0
            max_likely5 = np.argmax(likely)
            likely[max_likely5] = 0
            max_likely6 = np.argmax(likely)

#            print(max_likely)
            back = []
            back[0:5] = row[2:7]
            back.append(doc_sent6[max_likely1])
            back.append(doc_sent6[max_likely2])
            back.append(doc_sent6[max_likely3])
            back.append(doc_sent6[max_likely4])
            back.append(doc_sent6[max_likely5])
            back.append(doc_sent6[max_likely6])
            writer.writerow(back)

if __name__ == "__main__":
    data_train1 = 'train_stories.csv'
    data_out   = "near.csv"
    generate_data(data_train1, data_out)
