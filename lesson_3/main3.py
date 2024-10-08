import pandas as pd # Library for Dataframes
import numpy as np # Library for math functions
import pickle # Python object serialization library. Not secure
from google.colab import drive
drive.mount('/content/drive')


word_embeddings = pickle.load( open('/content/drive/MyDrive/!Kafedra/GOIT/woolf/numprog/mod2_1/word_embeddings_subset.p', "rb"))

len(word_embeddings) # there should be 243 words that will be used in this assignment
