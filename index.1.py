import json, csv
import pandas as pd
from pandas import DataFrame
import numpy as np
import time, os
os.system('cls')
######
import collections
from collections import defaultdict
from textblob import TextBlob
import re
import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
#Finding Similar words from defined Array of Features Extraction 
from nltk.corpus import wordnet as wn
from itertools import chain
import warnings
warnings.filterwarnings("ignore")
#####################################################
print('\n');print("Displaying Product Reviews data");print('----------------------------------')
fdata = pd.read_csv('dataset/final_dataset.csv', sep=',')
fddata = pd.DataFrame(fdata)
print(fddata)
#####################################################
print('\n');print("Calculating mean for all rated products");print('----------------------------------')
# time.sleep(3)
dd = fddata.groupby('p_id')['rating'].mean().sort_values(ascending=False).head(15) 
print(dd)
#####################################################
print('\n');print("Counting number of rating per product");print('----------------------------------')
ratings = pd.DataFrame(fddata.groupby('p_id')['rating'].mean().head(15))
ratings['num of ratings'] = pd.DataFrame(fddata.groupby('p_id')['rating'].count())
print(ratings)
# time.sleep(3)
print('\n');print("Listing Mosted rated product first");print('----------------------------------')
tr = ratings.sort_values('num of ratings', ascending = False).head(15)
print(tr)
#####################################################
print('\n');print("Sorting Highest (5 Star) rated product first");print('----------------------------------')
tr0 = ratings.sort_values('rating',ascending = False).head(15)
print(tr0)
#####################################################
print('\n');print("Recommending the similar product");print('----------------------------------')
correlation_matrix = fddata.pivot_table(index ='user_id', columns ='p_name', values ='rating') 
user_p456_ratings = correlation_matrix['iPhone 4'] 
user_p456_ratings.head(20)
######################################################
p456 = correlation_matrix.corrwith(user_p456_ratings) 
corr_p456 = pd.DataFrame(p456, columns=['Correlation']) 
corr_p456.dropna(inplace = True) 
corr_p456.head(10) 
cm = correlation_matrix.head(10)
cm.fillna('-',inplace=True)
print(cm)
print('\n');print("Recommended products");print('----------------------------------')
# time.sleep(5)
print(corr_p456)
print('\n');print("Sorting Similar products");print('----------------------------------')
# time.sleep(5)
sorts = corr_p456.sort_values('Correlation',ascending = False).head(5)
print(sorts)
print('\n');print("Mean for products");print('----------------------------------')
# time.sleep(3)
mean_all_prod = fddata.groupby('p_id')['rating'].mean().sort_values(ascending=False).head(20) 
print(mean_all_prod)



############ Test #################

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine

data = pd.read_csv('dataset/final_dataset.csv')

# # Create a new dataframe without the user ids.
# data_items = data.drop('p_id', 1)
# print (data_items)


# # magnitude = sqrt(x2 + y2 + z2 + ...)
# magnitude = np.sqrt(np.square(data_items).sum(axis=1))

# # unitvector = (x / magnitude, y / magnitude, z / magnitude, ...)
# data_items = data_items.divide(magnitude, axis='index')

# def calculate_similarity(data_items):
#     """Calculate the column-wise cosine similarity for a sparse
#     matrix. Return a new dataframe matrix with similarities.
#     """
#     data_sparse = sparse.csr_matrix(data_items)
#     similarities = cosine_similarity(data_sparse.transpose())
#     sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
#     return sim

# # Build the similarity matrix
# data_matrix = calculate_similarity(data_items)

# # Lets get the top 11 similar artists for Beyonce
# data_matrix.loc['iPhone'].nlargest(11)

############# TEst 2
