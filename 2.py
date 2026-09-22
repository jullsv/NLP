import os
import urllib.request
import gensim
import re

MODEL_URL = "https://marigostra.ru/static/nlp/cbow.txt"
MODEL_PATH = "cbow.txt"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

word2vec = gensim.models.KeyedVectors.load_word2vec_format(
    MODEL_PATH,
    binary=False
)

pos = ["папирус_NOUN","полистирол_NOUN"]
neg = []

dist = word2vec.most_similar(
    positive=pos,
    negative=neg,
    topn=100
)

pat = re.compile("(.*)_NOUN")
result = []

for i in dist:
    e = pat.match(i[0])

    if e is not None:
        result.append(i)

    if len(result) == 10:
        break

for i in result:
    e = pat.match(i[0])
    print(e.group(1), i[1])

words = [i[0] for i in result]
