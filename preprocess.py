import json
import requests
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
stop_de = set(stopwords.words('german'))
stop_en = set(stopwords.words('english'))
stop_es = set(stopwords.words('spanish'))

titles = pd.read_csv('data/mutual.csv', encoding='latin')

concepts_de = titles['de'].values
concepts_en = titles['en'].values
concepts_es = titles['es'].values


#api_token = 'english_wiki'
#api_url_base = 'https://es.wikipedia.org/w/api.php?action=query&format=json&titles=Pan|Biología&indexpageids= '
#api_url_base = 'https://de.wikipedia.org/w/api.php?action=query&prop=extracts&rvprop=content&titles=Ableitungl&format=json'


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
        text = [t for t in text if t not in stop_de]
    elif lang == 'en':
        text = [t for t in text if t not in stop_en]
    else:
        text = [t for t in text if t not in stop_es]
    return text


for i in range(len(concepts_es)):
    url = get_url('es',concepts_es[i])
    content = get_info(url)
    print(Counter(get_text(content, 'es')))






# account_info = get_account_info()
# print(account_info)
# #account_info = {"batchcomplete":'true',"query":{"pages":[{"pageid":15580374,"ns":0,"title":"Main Page","revisions":[{"contentformat":"text/x-wiki","contentmodel":"wikitext","content":"<!--        BANNER ACROSS TOP OF PAGE         -->\n<div id=\"mp-topbanner\" style=\"clear:both; position:relative; box-sizing:border-box; width:100%; margin:1.2em 0 6px; min-width:47em; border:1px solid #ddd; background-color:#f9f9f9; color:#000; white-space:nowrap;\">\n<!--        \"WELCOME TO WIKIPEDIA\" AND ARTICLE COUNT        -->\n<div style=\"margin:0.4em; width:22em; text-align:center;\">\n<div style=\"font-size:162%; padding:.1em;\">Welcome to [[Wikipedia]],</div>\n<div style=\"font-size:95%;\">the [[free content|free]] [[encyclopedia]] that [[Wikipedia:Introduction|anyone can edit]].</div>\n<div id=\"articlecount\" style=\"font-size:85%;\">[[Special:Statistics|{{NUMBEROFARTICLES}}]] articles in [[English language|English]]</div>\n</div>\n<!--        PORTAL LIST ON RIGHT-HAND SIDE        -->\n<ul style=\"position:absolute; right:-1em; top:50%; margin-top:-2.4em; width:38%; min-width:25em; font-size:95%;\">\n<li style=\"position:absolute; left:0; top:0;\">[[Portal:Arts|Arts]]</li>\n<li style=\"position:absolute; left:0; top:1.6em;\">[[Portal:Biography|Biography]]</li>\n<li style=\"position:absolute; left:0; top:3.2em;\">[[Portal:Geography|Geography]]</li>\n<li style=\"position:absolute; left:33%; top:0;\">[[Portal:History|History]]</li>\n<li style=\"position:absolute; left:33%; top:1.6em;\">[[Portal:Mathematics|Mathematics]]</li>\n<li style=\"position:absolute; left:33%; top:3.2em;\">[[Portal:Science|Science]]</li>\n<li style=\"position:absolute; left:66%; top:0;\">[[Portal:Society|Society]]</li>\n<li style=\"position:absolute; left:66%; top:1.6em;\">[[Portal:Technology|Technology]]</li>\n<li style=\"position:absolute; left:66%; top:3.2em;\"><strong>[[Portal:Contents/Portals|All portals]]</strong></li>\n</ul>\n</div>\n<!--        MAIN PAGE BANNER        -->\n{{#if:{{Main Page banner}}|\n<div id=\"mp-banner\" class=\"MainPageBG\" style=\"margin-top:4px; padding:0.5em; background-color:#fffaf5; border:1px solid #f2e0ce;\">\n{{Main Page banner}}\n</div>\n}}\n<!--        TODAY'S FEATURED CONTENT        -->\n{| role=\"presentation\" id=\"mp-upper\" style=\"width: 100%; margin-top:4px; border-spacing: 0px;\"\n<!--        TODAY'S FEATURED ARTICLE; DID YOU KNOW        -->\n| id=\"mp-left\" class=\"MainPageBG\" style=\"width:55%; border:1px solid #cef2e0; padding:0; background:#f5fffa; vertical-align:top; color:#000;\" |\n<h2 id=\"mp-tfa-h2\" style=\"margin:0.5em; background:#cef2e0; font-family:inherit; font-size:120%; font-weight:bold; border:1px solid #a3bfb1; color:#000; padding:0.2em 0.4em;\">{{#ifexpr:{{formatnum:{{PAGESIZE:Wikipedia:Today's featured article/{{#time:F j, Y}}}}|R}}>150|From today's featured article|Featured article <span style=\"font-size:85%; font-weight:normal;\">(Check back later for today's.)</span>}}</h2>\n<div id=\"mp-tfa\" style=\"padding:0.1em 0.6em;\">{{#ifexpr:{{formatnum:{{PAGESIZE:Wikipedia:Today's featured article/{{#time:F j, Y}}}}|R}}>150|{{Wikipedia:Today's featured article/{{#time:F j, Y}}}}|{{Wikipedia:Today's featured article/{{#time:F j, Y|-1 day}}}}}}</div>\n<h2 id=\"mp-dyk-h2\" style=\"clear:both; margin:0.5em; background:#cef2e0; font-family:inherit; font-size:120%; font-weight:bold; border:1px solid #a3bfb1; color:#000; padding:0.2em 0.4em;\">Did you know...</h2>\n<div id=\"mp-dyk\" style=\"padding:0.1em 0.6em 0.5em;\">{{Did you know}}</div>\n| style=\"border:1px solid transparent;\" |\n<!--        IN THE NEWS and ON THIS DAY        -->\n| id=\"mp-right\" class=\"MainPageBG\" style=\"width:45%; border:1px solid #cedff2; padding:0; background:#f5faff; vertical-align:top;\"|\n<h2 id=\"mp-itn-h2\" style=\"margin:0.5em; background:#cedff2; font-family:inherit; font-size:120%; font-weight:bold; border:1px solid #a3b0bf; color:#000; padding:0.2em 0.4em;\">In the news</h2>\n<div id=\"mp-itn\" style=\"padding:0.1em 0.6em;\">{{In the news}}</div>\n<h2 id=\"mp-otd-h2\" style=\"clear:both; margin:0.5em; background:#cedff2; font-family:inherit; font-size:120%; font-weight:bold; border:1px solid #a3b0bf; color:#000; padding:0.2em 0.4em;\">On this day...</h2>\n<div id=\"mp-otd\" style=\"padding:0.1em 0.6em 0.5em;\">{{Wikipedia:Selected anniversaries/{{#time:F j}}}}</div>\n|}\n<!--        TODAY'S FEATURED LIST        --><!-- CONDITIONAL SHOW -->{{#switch:{{CURRENTDAYNAME}}|Monday|Friday=\n<div id=\"mp-middle\" class=\"MainPageBG\" style=\"margin-top:4px; border:1px solid #f2cedd; background:#fff5fa; overflow:auto;\">\n<div id=\"mp-center\">\n<h2 id=\"mp-tfl-h2\" style=\"margin:0.5em; background:#f2cedd; font-family:inherit; font-size:120%; font-weight:bold; border:1px solid #bfa3af; color:#000; padding:0.2em 0.4em\">From today's featured list</h2>\n<div id=\"mp-tfl\" style=\"padding:0.3em 0.7em;\">{{#ifexist:Wikipedia:Today's featured list/{{#time:F j, Y}}|{{Wikipedia:Today's featured list/{{#time:F j, Y}}}}|{{TFLempty}}}}</div>\n</div>\n</div>|}}<!-- END CONDITIONAL SHOW -->\n<!--        TODAY'S FEATURED PICTURE        -->\n<div id=\"mp-lower\" class=\"MainPageBG\" style=\"margin-top:4px; border:1px solid #ddcef2; background:#faf5ff; overflow:auto;\">\n<div id=\"mp-bottom\">\n<h2 id=\"mp-tfp-h2\" style=\"margin:0.5em; background:#ddcef2; font-family:inherit; font-size:120%; font-weight:bold; border:1px solid #afa3bf; color:#000; padding:0.2em 0.4em\">{{#ifexist:Template:POTD protected/{{#time:Y-m-d}}|Today's featured picture | Featured picture&ensp;<span style=\"font-size:85%; font-weight:normal;\">(Check back later for today's.)</span>}}</h2>\n<div id=\"mp-tfp\" style=\"margin:0.1em 0.4em 0.6em;\">{{#ifexist:Template:POTD protected/{{#time:Y-m-d}}|{{POTD protected/{{#time:Y-m-d}}}}|{{POTD protected/{{#time:Y-m-d|-1 day}}}}}}</div>\n</div>\n</div>\n<!--        SECTIONS AT BOTTOM OF PAGE        -->\n<div id=\"mp-other\" style=\"padding-top:4px; padding-bottom:2px;\">\n== Other areas of Wikipedia ==\n{{Other areas of Wikipedia}}\n</div><div id=\"mp-sister\">\n== Wikipedia's sister projects ==\n{{Wikipedia's sister projects}}\n</div><div id=\"mp-lang\">\n== Wikipedia languages ==\n{{Wikipedia languages}}\n</div>\n<!--        INTERWIKI STRAPLINE        -->\n<noinclude>{{Main Page interwikis}}{{noexternallanglinks}}{{#if:{{Wikipedia:Main_Page/Tomorrow}}||}}</noinclude>__NOTOC____NOEDITSECTION__"}]}]}}
# text = next(iter(account_info['query']['pages'].values()))
# #print(text)
# article=(text['extract'])
# article=[t.lower() for t in article.split() if t.isalpha()]
# article=[t for t in article if t not in stop_de]

#print(Counter(article))










# de = open('data/de/full.txt').readlines()
# es = open('data/es/full.txt').readlines()
# #sep = es.read().split('\n\n')
# #for lin in de.readlines():
#  #   print(lin)
#
# def get_concepts(corpus):
#     #print("initialize")
#     #print("start")
#     all = dict()
#     art = dict()
#     #print(len(corpus))
#     for i in range(len(corpus)):
#         #print(i)
#         if corpus[i] != '\n':
#             if '[[' in corpus[i] and ']]' in corpus[i] and len(corpus[i]) < 20:
#                 key = corpus[i][2:-3]
#                 #print(key)
#             else:
#                 if key in all.keys():
#                     pass
#                     #all[key] += corpus[i].replace('\n', ' ')
#                 else:
#                     all[key] = True
#                     #all[key] = corpus[i].replace('\n', ' ')
#     return all
#
#
# de_concepts = get_concepts(de)
# es_concepts = get_concepts(es)
#
# mutual=[]
# for k in de_concepts.keys():
#     if k in es_concepts.keys():
#         #print(de_concepts[k])
#         mutual.append(k)
# print(len(mutual))
# print(mutual[:20])
#
# # test = open('data/test_es.txt','w')
# # test2 = open('data/test_de.txt','w')
# # part = mutual[:20]
# # for m in range(len(part)):
# #     for i in range(len(es)):
# #         if '[['+part[m]+']]' in es[i]:
# #             line =
# #             while line
#
#


