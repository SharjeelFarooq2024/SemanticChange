import math
import numpy as np

class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size if self.corpus_size > 0 else 0.0
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        if tokenizer:
            corpus = [self.tokenizer(doc) for doc in corpus]
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.f.append(frequencies)
            for word in frequencies:
                self.df[word] = self.df.get(word, 0) + 1
        for word, freq in self.df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.f])
            score += (self.idf.get(q, 0) * (q_freq * (self.k1 + 1)) /
                      (q_freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))))
        return score

    def get_top_n(self, query, documents, n=5):
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]
