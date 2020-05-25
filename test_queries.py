#!/usr/bin/env python
# coding: utf-8



import nltk
import re
import bs4 as bs 
import string  
import numpy as np
import json
import ast
import math
import pandas as pd
from scipy import spatial
import sys
import time
import os


def read_postinglist(filename):

    '''
    Function to read dictionary stored in json format in
    index.txt.
    input parameters : filepath
    output : posting list in form od a dictinary 
    ''' 

    fp = open(filename, 'r')
    ipStr = fp.read()
    posting_dict = json.loads(ipStr)
    posting_dict = {k:json.loads(posting_dict[k]) for k in posting_dict}
    fp.close()
    return posting_dict




def get_docids(filename):

    '''
    Function to get list of  
    document ids from the text file storing document ids as a list.
    input parameters : filepath for docids.txt(file storing doc ids).
    ouput: list of doc ids.
    '''

    docfile = open(filename, 'r')
    doc_id_list = [n.strip() for n in ast.literal_eval(docfile.read())]
    return doc_id_list




def cal_tfidf(doc_ids , posting_dict):
    ''' 
    Function to populate tf score matrix for the
    documents.

    Scheme used 'lnc' for matrix population.
    normalization incorporated while calculating 
    cosine similarity.

    input parameters: list of doc ids, posting list dictionary

    output: tf_score_matrix(dataframe)
    '''
    # if(os.path.exists("tf.csv")):
    #     return pd.read_csv("tf.csv")
    total_docs = len(doc_ids)
    row_name = doc_ids
    col_name = posting_dict.keys()
    tf_score_matrix = pd.DataFrame(0, index = row_name, columns = col_name)
    for word in posting_dict:
        for docno in posting_dict[word]:
            temp = posting_dict[word][docno]
            tf_score_matrix[word][docno] = 1 + np.log10(temp)
    # tf_score_matrix.to_csv("tf.csv")
    return tf_score_matrix




def cal_query_vector(query_doc,posting_dict,total_docs):

    '''
    Function to calculate query vector using scheme
    'ltc'
    normalization incorporated while calculating 
    cosine similarity.

    input parameters: vector of query words, posting list dictionary,
    total number of docum

    output: query vector with idf weigthing

    '''
    query_idf =  pd.DataFrame(columns = posting_dict.keys())
    query_idf.loc[0] = np.zeros(len(posting_dict))
    for word in query_doc:
        if(word not in posting_dict.keys()):
            continue
        query_idf[word] = np.log10((total_docs/len(posting_dict[word])))
    return query_idf




def get_top_k(doc_ids, tf_score_matrix,query_idf,k,query,posting_dict):

    ''' 
    Function returns top k documents based on
    cosine similarity score using basic vector space model
    '''
    cosine_sim = {}

    for docs in doc_ids:
        cosine_sim[docs] = 1 - spatial.distance.cosine(tf_score_matrix.loc[docs], query_idf.loc[0])
    sorted_cosine = sorted(cosine_sim.items(),  key= lambda x: x[1])[::-1]
    return sorted_cosine[:k]




def champion_list(query_doc, posting_dict, tf_score_matrix,query_idf,top_k):
    '''
    Function to incorporate improvement 1: champion list
    Return top k based on cosine similarit.
    '''
    query_championlist = {}
    query_doc  = [ word for word in query_doc if word in posting_dict.keys()]
    for word in query_doc:
        query_championlist[word] = sorted(posting_dict[word].items(), key=lambda x:x[1])[::-1]
    pruned_champion_list = {k:query_championlist[k][:15] for k in query_doc}

    doc_id_cl = set()

    for word in query_doc:
        for k in pruned_champion_list[word]:
            doc_id_cl.add(k[0])

    cosine_sim_cl = {}

    for docs in doc_id_cl:
        cosine_sim_cl[docs] = 1 - spatial.distance.cosine(tf_score_matrix.loc[docs], query_idf)

    sorted_cosine_cl = sorted(cosine_sim_cl.items(), key=lambda x:x[1])[::-1]

    return sorted_cosine_cl[:top_k]




def phrase_query(raw_query,bigram_posting):
    ''' 
    Function to run phrasal query on bigram inverted 
    index and return documents retrieved assigning them highest
    score.
    '''
    query_tokens =  nltk.word_tokenize(raw_query)
    query_bigrams = nltk.ngrams(query_tokens,2)
    temp = set()
    doc_set = set()
    
    for bigrams in query_bigrams:
        if(str(bigrams) not in bigram_posting.keys()):
            continue
        temp = set((bigram_posting[str(bigrams)]).keys())
        if(len(doc_set) == 0):
            doc_set = temp
        else:
            doc_set =doc_set.intersection(temp)
    return doc_set




def phrase_query_parser(k,raw_query,bigram_posting, posting_dict, tf_score_matrix,doc_id,query_idf):
    '''
    Function to extend phrasal query retrieval if documents retrieved
    are less than k run vector space model to retrieve more 
    documents.
    input parameters : k(top k documents), bigram_posting list, posting list,  tf_matrix, doc ids,
    query_idf vector.
    '''
    doc_set = phrase_query(raw_query,bigram_posting)
    if(len(doc_set) >= k):
        return [(x,1) for x in list(doc_set)[0:k]]
    else:
        doc_ranked_list = [(x,1) for x in list(doc_set)[0:k]]
        doc_list  = [x[0] for x in doc_ranked_list]
        next_set = get_top_k(doc_id, tf_score_matrix,query_idf,2*k ,raw_query.split(' '),posting_dict)
        index  = 0
        while(len(doc_ranked_list) <k):
            if(next_set[index][0] not in doc_list):
                doc_ranked_list.append(next_set[index])
            index =  index + 1
    return doc_ranked_list
        



def read_query(filename):
    '''
    Function to read query from textfile
    '''
    query = []
    try:
        f = open(filename, 'r')
    except IOError:
        print("Error: File does not appear to exist.")
        exit()
    query  = f.read()
    return query 





def main():
    option = 0 
    if(len(sys.argv) > 4):
        query_file = sys.argv[1]
        indexfile = sys.argv[2]
        bigram_indexfile = sys.argv[3]
        docid_file  =  sys.argv[4]
        if(len(sys.argv) > 5):
        	option = int(sys.argv[5])
            # if option == 1 , champion list model returns the answer for the query
            # if option == 2 , phrasal query + vector space model 
            # otherwise simple vector space model
    else:
        print("Insufficient Arguments")

    query = read_query(query_file)
    posting_dict = read_postinglist(indexfile)
    doc_id_list = get_docids(docid_file)
    tf_score_matrix = cal_tfidf(doc_id_list, posting_dict)
    
    print("\n\n\n Query: " + query + "\n")
    query_idf  = cal_query_vector(query.split(' '),posting_dict,len(posting_dict))

    if(option == 1):
        start =  time.process_time()
        res = champion_list(query.split(' '), posting_dict, tf_score_matrix, query_idf,10)
        end  =  time.process_time()
        print("\n\nUsing Champion list: \n")
        for docs in res:
            print("Doc id: " + docs[0] + " Score: " + str(docs[1]))
        print("\n\n Time taken to process query = " + str(end-start) + "\n\n")

    elif(option == 2):
        bigram_posting_list = read_postinglist(bigram_indexfile)
        start = time.process_time()
        res = phrase_query_parser(10,query,bigram_posting_list,posting_dict,tf_score_matrix,doc_id_list,query_idf)
        end  = time.process_time()
        print("\n\nUsing phrasal query plus vector space model: \n")
        for docs in res:
            print("Doc id: " + docs[0] + " Score: " + str(docs[1]))
        print("\n\n Time taken to process query = " + str(end-start) + "\n\n")
    else:
        start = time.process_time()
        res = get_top_k(doc_id_list, tf_score_matrix,query_idf,10,query,posting_dict)
        end  = time.process_time()
        print("\nUsing vector space model:\n")
        for docs in res:
            print("Doc id: " + docs[0] + " Score: " + str(docs[1]))
        print("\n\n Time taken to process query = " + str(end-start) + "\n\n")

if __name__ == "__main__":
    main()






