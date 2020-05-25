# -*- coding: utf-8 -*-
"""index_creation.ipynb

Automatically generated by Colaboratory.

"""

#from google.colab import drive
#drive.mount('/gdrive')

#pip install hashedindex

import nltk
import re
# import os
import bs4 as bs 
from nltk.tokenize import word_tokenize
import string  
import pandas as pd
import hashedindex
import json
import sys
# nltk.download('punkt')

# Commented out IPython magic to ensure Python compatibility.


def get_docs(filename):
  ''' 
  Function to parse <doc id > tags and store extract all the 
  documents id along with text for index creation 
  '''
  docs_clean = {}
  with  open(filename, "r") as file:
    data  = file.read()
    docs  = { doc.group(1).strip():doc.group(2).strip() for doc in re.finditer(r'<doc id="([0-9]*)".*?>(.*?)</doc>',data,flags = re.M|re.S)}
    docs_clean = { k: bs.BeautifulSoup(v,features="lxml").get_text() for k,v in docs.items()}
  # print(len(docs_clean))
  
  return docs_clean




def create_postinglist(docs):
  ''' Function to create inverted index to facilitate 
  vector space model for document retreival.
  '''
  index  =  hashedindex.HashedIndex()
  for (k,v) in docs.items():
    for tokens in nltk.word_tokenize(v):
      if tokens not in string.punctuation:
        index.add_term_occurrence(tokens,k)
  return index




def create_bigram_index(docs):
  ''' Function to create bigram-inverted index to facilitate 
  phrasal queries
  '''
  index  =  hashedindex.HashedIndex()
  for (k,v) in docs.items():
    tokens = nltk.word_tokenize(v)
    for bigrams in nltk.ngrams(tokens,2):
      index.add_term_occurrence(bigrams,k)
  return index




def save_index(filename, index):
  with open(filename, "w+") as f:
    temp  = { str(k):str(json.dumps(index[k])) for k in index.items()}
    j  = json.dumps(temp)
    f.write(j)
    f.close()



def store_docids(docs):
  f = open('docids.txt',"w")
  f.write(str(list(docs.keys()))) 
  f.close()




def create_indices(filenames):
  '''
  input parameter is list of filenames 
  to create combined index for than one 
  files.
  '''
  docs = get_docs(filenames)
  index = create_postinglist(docs)
  bigram_index = create_bigram_index(docs)
  save_index("index.txt", index)
  save_index("bigram_index.txt",bigram_index)

  # Need to save doc ids to docsids.txt
  print("No of documents: ") 
  print(len(docs))
  store_docids(docs)




def main():
  if(len(sys.argv) > 1):
    filename = sys.argv[1]
  else:
    filename = "./Wikipedia/AD/wiki_22"
    
  create_indices(filename)



if __name__ == "__main__":
  main()