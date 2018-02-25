import json
import requests
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer, GermanStemmer, EnglishStemmer

stemmer_es = SpanishStemmer()
stemmer_en = EnglishStemmer()
stemmer_de = GermanStemmer()

from invert import Matrix
from scipy.spatial.distance import cosine
import csv

#define sets of stopwords
stop_de = set(stopwords.words('german'))
stop_en = set(stopwords.words('english'))
stop_es = set(stopwords.words('spanish'))

#extract mutual concepts
titles = pd.read_csv('data/mutual.csv', encoding='latin')
concepts_de = titles['de'].values
concepts_en = titles['en'].values
concepts_es = titles['es'].values


def get_url(lang, concept):
    return 'https://'+lang+'.wikipedia.org/w/api.php?action=query&prop=extracts&rvprop=content&format=json&titles='+concept

def get_info(url):
    response = requests.get(url)
    #print(str(response.content))
    if response.status_code == 200:
        return json.loads(response.content.decode())
    else:
        return None

def get_text(js, lang):
    article = next(iter(js['query']['pages'].values()))
    text = (article['extract'])
    text = [t.lower() for t in text.split() if t.isalpha()]
    if lang == 'de':
        text = [stemmer_de.stem(t) for t in text if t not in stop_de]
    elif lang == 'en':
        text = [stemmer_en.stem(t) for t in text if t not in stop_en]
    else:
        text = [stemmer_es.stem(t) for t in text if t not in stop_es]
    return text

def build_corpus(concepts, lang):
    corpus = dict()
    for i in range(len(concepts)):
        url = get_url(lang, concepts[i])
        content = get_info(url)
        if concepts[i] in corpus.keys():
            corpus[concepts[i]] += get_text(content,lang)
        else:
            corpus[concepts[i]] = get_text(content, lang)
    return corpus



'''Experiment 1'''
result1 = open('experiment1.csv','a', encoding='utf-8')

de_data = Matrix(build_corpus(concepts_de,'de'),concepts_de)
de_data.fill_matrix()

en_data = Matrix(build_corpus(concepts_en,'en'),concepts_en)
en_data.fill_matrix()

es_data = Matrix(build_corpus(concepts_es,'es'),concepts_es)
es_data.fill_matrix()

for n in range(len(concepts_de)):
    result1.write(concepts_de[n]+'\t'+concepts_en[n]+'\t'+concepts_es[n]+'\n')
    words_de = [w[0] for w in de_data.get_words(n)]
    words_en = [w[0] for w in en_data.get_words(n)]
    words_es = [w[0] for w in es_data.get_words(n)]
    for k in range(len(words_de)):
        result1.write(words_de[k]+'\t'+words_en[k]+'\t'+words_es[k]+'\n')
    result1.write('\n')
    #fields = [concepts_de[n], concepts_en[n],concepts_es[n]]
    #writer = csv.DictWriter(result1, fieldnames=fields)
    #writer.writeheader()
    #writer.writerow(results)




# m_es = Matrix(sp,concepts_en)
# m_es.fill_matrix()
#
# result1= (m_es.get_concepts('matrix'))
# print(result1)
# m_de = Matrix(gr,concepts_de)
# m_de.fill_matrix()
#
# result2 = (m_de.get_concepts('matrix'))
# print(result2)
# mutual = len(set(result1.keys()&set(result2.keys())))
# all = (len(result1)+len(result2))-mutual
# print(all)
# print(mutual)
# print(mutual/all)

#print(m.build_matrix())

# i1 = list(m_es.voc).index('matrix')
#
# i2 = list(m_de.voc).index('matrix')
# print(m_es.matrix[i1,:])
# print(m_de.matrix[i2,:])
# print(cosine(m_es.matrix[i1,:], m_de.matrix[i2,:]))





