#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import sys
import json
import math
import re
import string
import time
from operator import add

# preprocess the input text to list of words (remove punctuation and numbers) 
def preprocess_text(text, stopword_list):
    pre_text = text.lower()
    removed = pre_text.translate(str.maketrans('', '', string.punctuation + string.digits)).split()
    result = [word for word in removed if word not in stopword_list and not re.findall("[^\u0061-\u007A]+", word)] # remove non-english characters
    return result

# count word tf scores per document
# filter out rare words
def compute_wordtf(iterator):
    words_count_dict = {}
    # list of word_lists
    for ls in list(iterator):
        for key in ls:
            if key not in words_count_dict:
                words_count_dict[key] = 1
            else:
                words_count_dict[key] += 1
    # filter out rare words with frequency less than 4
    newDict = {key: value for (key, value) in words_count_dict.items() if value >= 2}
    sorted_wordlist = sorted(newDict.items(), key = lambda x: (-x[1], x[0]))
    output_list = [(pair[0], pair[1]/sorted_wordlist[0][1]) for pair in sorted_wordlist]
    return output_list
    
# compute idf for each word
def compute_idf(iterator):
    # input: list of docs
    docs = list(iterator)
    count = len(docs)
    return [(i, math.log(len(business_dict)/count)) for i in docs]

def dict_to_list(a_dict, name):
    return [{'data': name, 'key': key, 'value': value} for key, value in a_dict.items()]

if __name__ == '__main__':
    t1 = time.time()
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    stopwords_file = sys.argv[3]

# train_file = 'train_review.json'
# model_file = 'out.json'
# stopwords_file = 'stopwords'
    stopword_list = []

    with open(stopwords_file) as file:
        for words in file:
            stopword_list.append(words.strip())

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    data = sc.textFile(train_file).map(lambda line: json.loads(line))

    # create a user_id: index dict
    user_dict = data.map(lambda item: item['user_id']).distinct().zipWithIndex().collectAsMap()

    # create a business_id: index dict
    business_dict = data.map(lambda item: item['business_id']).distinct().zipWithIndex().collectAsMap()

    # Not used because it removed too many words!
    # compute rare word frequency and create frequent word:index dict
    # word_count_pairs = data.flatMap(lambda item: preprocess_text(item['text'], stopword_list)).reduceByKey(add)
    # total_count = word_count_pairs.map(lambda x : x[1]).sum()
    # frequent_words_dict = word_count_pairs.filter(lambda word_cnt_pair: word_cnt_pair[1] >= math.floor(0.000001*total_count)).map(lambda x : x[0]).zipWithIndex().collectAsMap()
    # frequent_words_dict 

    # create business_wordtf_rdd
    # preprocess text
    # e.g., [((0, 'tanning'), 1.0), ((0, 'month'), 0.875), ...]
    business_wordtf_rdd = data.map(lambda item: (business_dict[item['business_id']], preprocess_text(item['text'], stopword_list)))                     .groupByKey().flatMapValues(compute_wordtf).map(lambda pair: ((pair[0], pair[1][0]), pair[1][1]))

    # create business_wordidf_rdd
    # e.g., [((0, 'customer'), 1.1984288519850217),
    #        ((2240, 'customer'), 1.1984288519850217)]
    business_wordidf_rdd = business_wordtf_rdd.map(lambda pair: (pair[0][1], pair[0][0])).groupByKey()                             .flatMapValues(compute_idf).map(lambda pair: ((pair[1][0], pair[0]), pair[1][1]))

    # create business_tfidf_rdd  (business_index: [list of words])
    business_tfidf_rdd = business_wordtf_rdd.join(business_wordidf_rdd)                             .map(lambda pair: (pair[0][0], (pair[0][1], pair[1][0]*pair[1][1]))).groupByKey()                             .mapValues(lambda x: sorted(list(x), key = lambda x: -x[1])[:200])                             .mapValues(lambda x: [i[0] for i in x])

    # create a word: index dict
    word_dict = business_tfidf_rdd.flatMap(lambda pair: [(i, 1) for i in pair[1]])                     .map(lambda pair: pair[0]).distinct().zipWithIndex().collectAsMap()

    # create a business(index): profile(list of word index) dict
    business_profile = business_tfidf_rdd.mapValues(lambda wordlist: [word_dict[i] for i in wordlist]).collectAsMap()

    # create a user(index): profile(list of word index) dict
    user_profile = data.map(lambda item: (user_dict[item['user_id']], business_dict[item['business_id']])).groupByKey()             .mapValues(set).flatMapValues(lambda business_ids: [business_profile[i] for i in business_ids if i in business_profile])             .mapValues(set).mapValues(list).collectAsMap()

    output = dict_to_list(user_dict, 'user_dict')
    output.extend(dict_to_list(business_dict, 'business_dict'))
    output.extend(dict_to_list(user_profile, 'user_profile'))
    output.extend(dict_to_list(business_profile, 'business_profile'))

    with open(model_file, 'w+') as out:
        for row in output:
            out.writelines(json.dumps(row) + "\n")

    t2 = time.time()
    print("Duration: " + str(t2 - t1))


# In[ ]:




