import json
import requests
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer, GermanStemmer, EnglishStemmer
from invert import Matrix
from scipy.spatial.distance import cosine, euclidean, sqeuclidean, seuclidean
from nltk.tag import pos_tag
import treetaggerwrapper
#import csv

#load stemmers
#stemmer_es = SpanishStemmer()
#stemmer_en = EnglishStemmer()
#stemmer_de = GermanStemmer()

class Embeddings:
    def __init__(self, smooth):
        # extract mutual concepts
        titles = pd.read_csv('data/mutual.csv', encoding='latin')
        self.concepts_de = titles['de'].values
        self.concepts_en = titles['en'].values
        self.concepts_es = titles['es'].values
        self.de_data = None
        self.en_data = None
        self.es_data = None
        self.smooth = smooth

    def get_url(self, lang, concept):
        return 'https://'+lang+'.wikipedia.org/w/api.php?action=query&prop=extracts&rvprop=content&format=json&titles='+concept

    def get_info(self,url):
        response = requests.get(url)
        #print(str(response.content))
        if response.status_code == 200:
            return json.loads(response.content.decode())
        else:
            return None

    def get_text(self,js, lang):
        #define sets of stopwords
        stop_de = set(stopwords.words('german'))
        stop_en = set(stopwords.words('english'))
        stop_es = set(stopwords.words('spanish'))
        tagger_de = treetaggerwrapper.TreeTagger(TAGLANG='de', TAGDIR='/home/polina/Dokumente/semparse/overnight/treeTagger')
        tagger_es = treetaggerwrapper.TreeTagger(TAGLANG='es', TAGDIR='/home/polina/Dokumente/semparse/overnight/treeTagger')
        tagger_en = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='/home/polina/Dokumente/semparse/overnight/treeTagger')

        article = next(iter(js['query']['pages'].values()))
        text = (article['extract'])
        text = [t.lower() for t in text.split() if t.isalpha()]

        if lang == 'de':
            #text = [stemmer_de.stem(t) for t in text if t not in stop_de]
            text = [t for t in text if t not in stop_de]
            tags= tagger_de.tag_text(text)
            text = [tag[0] for tag in treetaggerwrapper.make_tags(tags) if tag[1] == 'NN' or tag[1] == 'NE' or tag[1] == 'ADJA']
        elif lang == 'en':
            #text = [stemmer_en.stem(t) for t in text if t not in stop_en]
            text = [t for t in text if t not in stop_en]
            tags = tagger_en.tag_text(text)
            text = [tag[0] for tag in treetaggerwrapper.make_tags(tags) if tag[1] == 'NN' or tag[1] == 'NP' or tag[1] == 'NNS' or tag[1] == 'NPS' or tag[1] == 'JJ']
        else:
            #text = [stemmer_es.stem(t) for t in text if t not in stop_es]
            text = [t for t in text if t not in stop_es]
            tags= tagger_es.tag_text(text)
            text = [tag[0] for tag in treetaggerwrapper.make_tags(tags) if tag[1] == 'NC' or tag[1] == 'NMEA' or tag[1] == 'NP' or tag[1] == 'ADJ' or 'VL' in tag[1]]

        return text

    def build_corpus(self,concepts, lang):
        corpus = dict()
        for i in range(len(concepts)):
            url = self.get_url(lang, concepts[i])
            content = self.get_info(url)
            if concepts[i] in corpus.keys():
                corpus[concepts[i]] += self.get_text(content,lang)
            else:
                corpus[concepts[i]] = self.get_text(content, lang)
        return corpus

    def train(self):
        self.de_data = Matrix(self.build_corpus(self.concepts_de,'de'),self.concepts_de)
        if self.smooth:
            self.de_data.smooth(0.8)
        else:
            self.de_data.fill_counts()
        self.de_data.fill_matrix()

        self.en_data = Matrix(self.build_corpus(self.concepts_en,'en'),self.concepts_en)
        if self.smooth:
            self.en_data.smooth(0.8)
        else:
            self.en_data.fill_counts()
        self.en_data.fill_matrix()

        self.es_data = Matrix(self.build_corpus(self.concepts_es,'es'),self.concepts_es)
        if self.smooth:
            self.es_data.smooth(0.8)
        else:
            self.es_data.fill_counts()
        self.es_data.fill_matrix()


'''Experiment 1'''
def compare_concepts(concepts_de, concepts_en, concepts_es, de_data, en_data, es_data):
    result1 = open('experiment1smooth.csv','w', encoding='utf-8')
    for n in range(len(concepts_de)):
        result1.write(concepts_de[n]+'\t'+concepts_en[n]+'\t'+concepts_es[n]+'\n')
        result1.write('\n')
        words_de = [w[0] for w in de_data.get_words(n)]
        words_en = [w[0] for w in en_data.get_words(n)]
        words_es = [w[0] for w in es_data.get_words(n)]
        for k in range(len(words_de)):
            result1.write(words_de[k]+'\t'+words_en[k]+'\t'+words_es[k]+'\n')
        result1.write('\n')
        result1.write('\n')
    #fields = [concepts_de[n], concepts_en[n],concepts_es[n]]
    #writer = csv.DictWriter(result1, fieldnames=fields)
    #writer.writeheader()
    #writer.writerow(results)

'''Experiment 2'''

#for words of one language
def compare_same(word1, word2, data):
    w1=data.voc.index(word1)
    w2=data.voc.index(word2)
    return (cosine(data.matrix[w1,:], data.matrix[w2,:]),euclidean(data.matrix[w1,:], data.matrix[w2,:]))

'''Experiment 3'''
#for words of two different languages
def compare_diff(word1, word2, data1, data2):
    w1=data1.voc.index(word1)
    w2=data2.voc.index(word2)
    return (cosine(data1.matrix[w1,:], data2.matrix[w2,:]),euclidean(data1.matrix[w1,:], data2.matrix[w2,:]))

if __name__ == '__main__':
    #decide whether to do smoothing or not
    smooth = input('Do you want to use a smoothing technique?(y/n) ')
    print ('Loading embeddings...')
    if smooth == 'y':
        emb=Embeddings(True)
    else:
        emb=Embeddings(False)
    emb.train()
    proceed = True
    while proceed:
        exp_number=input('Which experiment do you choose?(1,2,3) ')
        if exp_number == '1':
            print ('Comparing concepts...')
            compare_concepts(emb.concepts_de,emb.concepts_en,emb.concepts_es, emb.de_data, emb.en_data, emb.es_data)
            print('Done. You can find the result in "experiment1.csv"')
            prc = input('Do you want to do another experiment?(y/n) ')
            if prc == 'y':
                continue
            else:
                proceed=False
        elif exp_number == '2':
            print ('Comparing two words of the same language...')
            proceed1 = True
            while proceed1:
                lang = input('Give the language (en, de, es): ')
                word1 = input('Give the first word: ')
                word2 = input('Give the second word: ')
                if lang == 'en':
                    result=compare_same(word1, word2, emb.en_data)
                    print('Cosine similarity: '+str(result[0])+' Euclidean distance: ' + str(result[1]))
                elif lang == 'de':
                    result = compare_same(word1, word2,emb.de_data)
                    print('Cosine similarity: ' + str(result[0]) + ' Euclidean distance: ' + str(result[1]))
                else:
                    result = compare_same(word1, word2, emb.es_data)
                    print('Cosine similarity: ' + str(result[0]) + ' Euclidean distance: ' + str(result[1]))
                nxt = input('Do you want to proceed with this experiment?(y/n) ')
                if nxt == 'y':
                    continue
                else:
                    proceed1 = False
            prc = input('Do you want to do another experiment?(y/n) ')
            if prc == 'y':
                continue
            else:
                proceed=False
        elif exp_number == '3':
            print('Comparing two words of different languages...')
            proceed1 = True
            while proceed1:
                lang1 = input('Give the first language (en, de, es): ')
                word1 = input('Give the first word: ')
                lang2 = input('Give the second language (en, de, es): ')
                word2 = input('Give the second word: ')
                if lang1 == 'en' and lang2 == 'de':
                    result = compare_diff(word1, word2, emb.en_data, emb.de_data)
                    print('Cosine similarity: ' + str(result[0]) + ' Euclidean distance: ' + str(result[1]))
                elif lang1 == 'en' and lang2 == 'es':
                    result=compare_diff(word1, word2, emb.en_data, emb.es_data)
                    print('Cosine similarity: ' + str(result[0]) + ' Euclidean distance: ' + str(result[1]))
                elif lang1 == 'es' and lang2 == 'de':
                    result=compare_diff(word1, word2, emb.es_data, emb.de_data)
                    print('Cosine similarity: ' + str(result[0]) + ' Euclidean distance: ' + str(result[1]))
                elif lang1 == 'de' and lang2 == 'en':
                    result=compare_diff(word1, word2, emb.de_data, emb.en_data)
                    print('Cosine similarity: ' + str(result[0]) + ' Euclidean distance: ' + str(result[1]))
                elif lang1 == 'de' and lang2 == 'es':
                    result=compare_diff(word1, word2, emb.de_data, emb.es_data)
                    print('Cosine similarity: ' + str(result[0]) + ' Euclidean distance: ' + str(result[1]))
                elif lang1 == 'es' and lang2 == 'en':
                    result = compare_diff(word1, word2, emb.es_data, emb.en_data)
                    print('Cosine similarity: ' + str(result[0]) + ' Euclidean distance: ' + str(result[1]))
                nxt = input('Do you want to proceed with this experiment?(y/n) ')
                if nxt == 'y':
                    continue
                else:
                    proceed1 = False
            prc = input('Do you want to do another experiment?(y/n) ')
            if prc == 'y':
                continue
            else:
                proceed=False



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





