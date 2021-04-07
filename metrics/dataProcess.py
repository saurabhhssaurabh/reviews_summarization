# import glob
# import json 
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import nltk
# from nltk.corpus import stopwords
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# import string
 
# nltk.download('stopwords')

import glob
import json 
import gzip 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_ROOT = './data/Electronics.json.gz'
DATA_ROOT_JSON = './data/Electronics.json'
# DATA_ROOT_JSON = './sample_data/Gift_Cards.json'
X_train_fp = './preprocessed/X_train.npy'
X_test_fp = './preprocessed/X_test.npy'
y_train_fp = './preprocessed/y_train.npy'
y_test_fp = './preprocessed/y_test.npy'

MAX = 0
MAX_STRING = ''
IDX = 0
FILES =  glob.glob(DATA_ROOT)

# turn a doc into clean tokens
def clean_doc(doc):
  # split into tokens by white space
  tokens = doc.split()
  # remove punctuation from each token
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
  # remove remaining tokens that are not alphabetic
  tokens = [word for word in tokens if word.isalpha()]
  # filter out stop words
  stop_words = set(stopwords.words('english'))
  tokens = [w for w in tokens if not w in stop_words]
  # filter out short tokens
  tokens = [word for word in tokens if len(word) > 1]
  return tokens

# sanity check
# doc = """
#     Then I cant even find my card number. None of the examples on there website show my card. Absolutely horrible experience and would never recommend this to anyone. You have been warned!!!!!!
#     Bought as a gift for my dad but should have read the small print about how to use the card as it was a little difficult since my dad is older.  Good deal but cant use at some of the better courses.
#     This gift card, works nothing like a gift card. Very difficult to redeem. Causes your gift recipient quite a bit of work.
#     I purchased and gave the Go Play Golf Card to a good friend, who is an avid golfer.
#     It is a great gift. The smile you will get is priceless.
# """
# a = clean_doc(doc)
# print(a)

def find_max_string():   
    # the goal of this function is to find the max token among all the json files in directory
    for file in tqdm(FILES):
        print('Processing : {0}'.format(file))
        data = pd.read_json(file, lines=True)
        for idx, d in enumerate(data['reviewText']):
            if len(str(d)) > MAX:
                MAX = len(str(d))
                IDX = idx
        print('String max: {0}'.format(MAX))
        MAX_STRING = data['reviewText'][IDX]

    print('Global MAX: {0}'.format(MAX))
    print('Global MAX_STRING: {0}'.format(MAX_STRING))

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def prepareData(fp):
  data = getDF(fp)
  train_text = data['reviewText'].to_numpy(dtype='str')
  train_label = data['overall'].to_numpy()
  train_label = train_label.reshape((len(train_label), 1))
  train_label = tf.keras.utils.to_categorical(train_label, num_classes=6)
  train_label = np.asarray(train_label).astype(np.int)
  X_train, X_test, y_train, y_test = train_test_split(train_text, train_label, test_size=0.2, shuffle=True)
  return X_train, X_test, y_train, y_test

def prepareDataJSON(fp):
  train_text = []
  train_label = []
  reader = pd.read_json(fp, lines=True, chunksize=10000)
  i = 0
  for data in reader:
    train_text = train_text + list(data['reviewText'])
    train_label = train_label + list(data['overall'])
    i += 1
    if(i > 30):
      break
  del data
  train_text = np.array(train_text, dtype='str')
  train_label = np.array(train_label)
  print('loaded data')
#   train_text = data['reviewText'].to_numpy(dtype='str')
#   print('train_text')
#   train_label = data['overall'].to_numpy()
#   print('train_label')
  train_label = train_label.reshape((len(train_label), 1))
  print('train_label 2')
  train_label = tf.keras.utils.to_categorical(train_label, num_classes=6)
  print('train_label 3')
  train_label = np.asarray(train_label).astype(np.int)
  print('train_label 4')
  X_train, X_test, y_train, y_test = train_test_split(train_text, train_label, test_size=0.2, shuffle=True)
  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepareDataJSON(DATA_ROOT_JSON)
np.save(X_train_fp, X_train)
np.save(X_test_fp, X_test)
np.save(y_train_fp, y_train)
np.save(y_test_fp, y_test)

# find_max_string()