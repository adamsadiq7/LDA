import gensim.corpora as corpora
from nltk.corpus import stopwords
import nltk
from gensim.utils import simple_preprocess
import gensim
import pandas as pd
import re
from pprint import pprint 

ngrams = []

def clean(data):
    data = data.lower()
    data = data.replace(",", "")
    data = data.replace(".", "")
    data = data.replace("-", "")
    data = data.replace(u"\xa0", u"")
    data = data.replace("\xa0", "")
    data = data.replace(u"\xa0", "")
    data = data.replace("(", "")
    data = data.replace(")", "")
    data = data.replace("/", "")
    data = data.replace("!", "")
    data = data.replace("`", "")
    data = data.replace("  ", " ")
    data = data.replace(":", "")
    data = data.replace(";", "")
    data = data.replace("?", "")
    data = data.replace("'", "")
    data = data.replace("\n", "")
    data = data.replace("\\\n", "")
    return data

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def find_double_words(description):
    new_description = description
    for double_word in ngrams:
        if double_word in new_description:
            holder = double_word.split(' ')
            holder = '_'.join(holder)
            new_description = new_description.replace(double_word, holder)
    return new_description

if __name__ == '__main__':

    filename = ""

    nltk.download('stopwords')

    stop_words = stopwords.words('english')
    stop_words.extend(['pardeftab', 'cf', 'sa', 'fs'])
    
    df = pd.read_csv(f'../{filename}.txt', delimiter="\t")
    
    # # Remove punctuation
    df['Sun020122'] = \
    df['Sun020122'].map(lambda x: re.sub('[,\.!?:"*\']', '', x))

    # Convert titles to lowercase
    df['Sun020122'] = \
    df['Sun020122'].map(lambda x: x.lower())    

    # # Join words together
    long_sermon = ','.join(list(df['Sun020122'].values))
    long_sermon = [find_double_words(long_sermon)]
    long_sermon = remove_stopwords(long_sermon)

    id2word = corpora.Dictionary(long_sermon)

    texts = long_sermon

    corpus = [id2word.doc2bow(text) for text in texts]

    num_topics = 1

    lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
    pprint(lda_model.print_topics())
