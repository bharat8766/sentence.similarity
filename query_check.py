import time
start = time.time()
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import pandas as pd
import nltk
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#===============================================================================================
from colorama import Fore, Back, Style


def yellowtext(s):
    """Yellow text"""
    return Fore.YELLOW + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def greentext(s):
    """Green text"""
    return Fore.GREEN + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def redtext(s):
    """Red text"""
    return Fore.RED + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def redback(s):
    """Red background"""
    return Back.RED + s + Back.RESET

def magentaback(s):
    """Magenta background"""
    return Back.MAGENTA + s + Back.RESET
#======================================================================================================
input_filename = input("Input text filename : ")
filename = input_filename+'.txt'
with open(filename, 'r') as data:
  tr = data.read()
print("\n")
print("Loading model...")
# tensroflow hub module for Universal sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" 
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", 
#"https://tfhub.dev/google/universal-sentence-encoder-large/3"]

embed = hub.Module(module_url)


query1 = "Tata Group Mediprime mediclaim policy tata group axis bank account verification department Medical Health Policy "
query2 = "insured Person suffers an Illness or Accident during the Policy period that requires Insured Person’s Hospitalization as an inpatient,"
query3 = "Day Care Procedure hospital beds Day Care procedures are surgeries or procedures which are undertaken by insured person as an inpatient for less than 24 hours in a hospital."
query4 = "Critical Illness Cover Cancer heart attack kidney failure bone marrow transplantation paralysis stroke"
# query5 = "Pre post hospitalization benefit includes medical Expenses incurred within 30 days immediately before the Insured Person was hospitalized  60 days immediately"


# keyword1 = "Tata Group Mediprime mediclaim policy"
# keyword2 = "In-patient treatment"
# keyword3 = "Pre post hospitalization"
# keyword4 = "Day Care Procedure"
# keyword5 = "Critical Illness Cover"


tr = tr.replace('\n','')


def splitParagraphIntoSentences(paragraph):
    ''' break a paragraph into sentences
        and return a list '''
    import re
    # to split by multile characters

    #   regular expressions are easiest (and fastest)
    sentenceEnders = re.compile('[!.?]')
    sentenceList = sentenceEnders.split(paragraph)
    return sentenceList

if __name__ == '__main__':
    p = tr
    nlst = []
    from IPython.display import HTML, display
    import tabulate
    sentences = splitParagraphIntoSentences(p)
    i = 0
    for s in sentences:
        k = str(i) + ") " + s.strip()
        i = i+1

#for text embeddings : 
def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return sess.run(embed(texts))

#preprocessing the text via removing stop words and some other symbols
def remove_stopwords(stop_words, tokens):
    res = []
    for token in tokens:
        if not token in stop_words:
            res.append(token)
    return res

def process_text(text):
    text = text.encode('ascii', errors='ignore').decode()
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'#+', ' ', text )
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
    text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)
    #text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text

def lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemma_list = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, 'v')
        if lemma == token:
            lemma = lemmatizer.lemmatize(token)
        lemma_list.append(lemma)
        
    return lemma_list

def process_all(text):
    text = process_text(text)
    return ' '.join(remove_stopwords(stop_words, text.split()))

data = sentences
data_processed = list(map(process_text, data))

#creating embedding vectors of the data
BASE_VECTORS = get_features(data)
BASE_VECTORS.shape


#cosine similarity metric for finding the semantic similarity between the splitted sentences and the phrase/query 
# that is needed to be checked.
def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)


def test_similarity(text1, text2):
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]
    print(vec1.shape)
    return cosine_similarity(vec1, vec2)



def semantic_search(query, data, vectors):
    from IPython.display import HTML, display
    import tabulate
    
    query = process_text(query)
    print("Extracting features...")
    query_vec = get_features(query)[0].ravel()
    res = []
    for i, d in enumerate(data):

        qvec = vectors[i].ravel()
        sim = round(cosine_similarity(query_vec, qvec)*100, 3)
        res.append((sim, d[:100]))

    sr  = sorted(res, key=lambda x : x[0], reverse=True)
    table = sr

    df = pd.DataFrame(data=np.array(table), columns=['confidence', 'sentence'])
    
    return df.head()




print("Query: ", end='')
print(greentext(query1))
print(semantic_search(query1, data_processed, BASE_VECTORS))
print("\n")
print("Query: ", end='')
print(greentext(query2))
print(semantic_search(query2, data_processed, BASE_VECTORS))
print("\n")
print("Query: ", end='')
print(greentext(query3))
print(semantic_search(query3, data_processed, BASE_VECTORS))
print("\n")
print("Query: ", end='')
print(greentext(query4))
print(semantic_search(query4, data_processed, BASE_VECTORS))
print("\n")
# print("\n")
# print("Query: ", end='')
# print(greentext(query5))
# print(semantic_search(query5, data_processed, BASE_VECTORS))
end = time.time()
print("Time taken : ",end='')
print(round(end-start,3))


# print("\n")
# print("Keyword: ", end='')
# print(greentext(keyword1))
# print(semantic_search(keyword1, data_processed, BASE_VECTORS))
# print("\n")
# print("Keyword: ", end='')
# print(greentext(keyword2))
# print(semantic_search(keyword2, data_processed, BASE_VECTORS))
# print("\n")
# print("Keyword: ", end='')
# print(greentext(keyword3))
# print(semantic_search(keyword3, data_processed, BASE_VECTORS))
# print("\n")
# print("Keyword: ", end='')
# print(greentext(keyword4))
# print(semantic_search(keyword4, data_processed, BASE_VECTORS))
# print("\n")
# print("Keyword: ", end='')
# print(greentext(keyword5))
# print(semantic_search(keyword5, data_processed, BASE_VECTORS))

