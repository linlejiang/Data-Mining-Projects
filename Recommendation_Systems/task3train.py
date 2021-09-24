#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pyspark
import sys
import json
from itertools import combinations
import math
import time

# item-based prediction & user-based prediction: 
# for each business_id, get the average rating per user
def compute_average(x):
    # input a iterator of list of tuples
    # return a user: star dict
    ls = list(x)
    temp = {}
    for i in ls:
        if i[0] not in temp:
            temp[i[0]] = [i[1], 1]
        else:
            temp[i[0]][0] += i[1]
            temp[i[0]][1] += 1
                        
    return {i: temp[i][0]/temp[i][1] for i in temp}

# item-based prediction & user-based prediction: 
# to filter out pairs with less than co-rating threshold of 3
def check_co_rating(dict1, dict2):
    if len(set(dict1.keys()).intersection(set(dict2.keys()))) >= 3:
        return True
    else:
        return False
    
# item-based prediction & user-based prediction:
def compute_pearson_sim(dict1, dict2):
    
    co_rated = set(dict1.keys()).intersection(set(dict2.keys()))
    mean1 = sum([dict1[i] for i in co_rated])/len(co_rated)
    mean2 = sum([dict2[i] for i in co_rated])/len(co_rated)
    numerator = sum([(dict1[i] - mean1)*(dict2[i] - mean2) for i in co_rated])
    
    if numerator == 0:
        sim = 0
        return sim
    
    denominator_dict1 = math.sqrt(sum([(dict1[i] - mean1)**2 for i in co_rated]))
    denominator_dict2 = math.sqrt(sum([(dict2[i] - mean2)**2 for i in co_rated]))
    
    if denominator_dict1*denominator_dict2 == 0:
        sim = 0
    else:
        sim = numerator/(denominator_dict1*denominator_dict2)
    
    return sim

# user-based prediction:
# create hash functions
def minhash(x):
    hashed_value = []
    prime_num = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 
                 197, 199, 211, 223, 227, 229]
    i = 0
    while i <= 49:
        hashed_value.append((prime_num[i]*x + prime_num[-(i+1)]) % m)
        i += 1
    return hashed_value

# user-based prediction:
# generate signatures given list_of_minhash_lists
def generate_signature(ls):
    signature = []
    i = 0
    while i <= 49:
        value = [999999999999999]
        for j in ls:
            if j[i] < value[0]:
                value[0] = j[i]
        signature.append(value[0])
        i += 1
    return signature

# user-based prediction:
# LSH function
# input: one row of the signature matrix
# output: ((chunk_index, band_list), user_id) for groupByKey in LSH
def LSH(user_id, signature):
    output_list = []
    chunk_index = 0
    
    while chunk_index <= 49:
        output_list.append(((chunk_index, signature[chunk_index]), user_id))
        chunk_index += 1
        
    return output_list

# user-based prediction:
def compute_jaccard_sim(dict1, dict2):
    co_rated = set(dict1.keys()).intersection(set(dict2.keys()))
    if len(co_rated) >= 3:
        return float(len(co_rated)/len(set(dict1.keys()).union(set(dict2.keys()))))
    else:
        return 0

if __name__ == '__main__':

    t1 = time.time()
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    cf_type = sys.argv[3]

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    data = sc.textFile(train_file).map(lambda line: json.loads(line))                 .map(lambda item: (item['user_id'], item['business_id'], item['stars']))

    # create a user_id: index dict
    user_dict = data.map(lambda item: item[0]).distinct().zipWithIndex().collectAsMap()

    # create a business_id: index dict
    business_dict_rdd = data.map(lambda item: item[1]).distinct().zipWithIndex()
    business_dict = business_dict_rdd.collectAsMap()

    if cf_type == 'item_based':

        # create business_user_dict
        # return {business_id1: {{userid1, star}, {userid2, star}, ...}, business_id2: {{}, {}, ...}} dict
        business_user_dict = data.map(lambda pair: (business_dict[pair[1]], (user_dict[pair[0]], pair[2])))                 .groupByKey().mapValues(compute_average).filter(lambda pair: len(pair[1]) >= 3).collectAsMap()

        # create index_business_id dict
        index_business_id = {index: business_id for business_id, index in business_dict.items()}

        # generate all possible comparison pairs # 1171857 pairs with at least 3 co-ratings, generate pairwise pearson similarity, sorted, and filtered
        business_count = len(business_dict)
        comparison_pairs = business_dict_rdd.map(lambda pair: (pair[1], pair[1]))                 .flatMapValues(lambda index: [i for i in range(index+1, business_count+1)])                 .filter(lambda pair: pair[0] in business_user_dict and pair[1] in business_user_dict)                 .filter(lambda pair: check_co_rating(business_user_dict[pair[0]], business_user_dict[pair[1]]))                 .map(lambda pair: (pair, compute_pearson_sim(business_user_dict[pair[0]], business_user_dict[pair[1]])))                 .filter(lambda pair: pair[1] > 0)                 .map(lambda pair: {"b1": index_business_id[pair[0][0]], "b2": index_business_id[pair[0][1]], "sim": pair[1]}) # 567147 resulting pairs

    elif cf_type == 'user_based':
        # number of buckets
        m = business_dict_rdd.count()*2

        # create user_business_signature
        user_business_signature = data.map(lambda pair: (user_dict[pair[0]], business_dict[pair[1]]))                                 .groupByKey().mapValues(set).filter(lambda pair: len(pair[1]) >= 3)                                 .flatMap(lambda pair: [(pair[0], i) for i in pair[1]])                                 .mapValues(minhash).groupByKey().mapValues(list).mapValues(generate_signature)

        # create user_business_dict
        # return {user_id1: {{businessid1, star}, {businessid2, star}, ...}, user_id2: {{}, {}, ...}} dict
        user_business_dict = data.map(lambda pair: (user_dict[pair[0]], (business_dict[pair[1]], pair[2])))                 .groupByKey().mapValues(compute_average).filter(lambda pair: len(pair[1]) >= 3).collectAsMap()

        # create index_user_id dict
        index_user_id = {index: user_id for user_id, index in user_dict.items()}

        # LSH algorithm to find pairs
        comparison_pairs = user_business_signature.flatMap(lambda pair: LSH(pair[0], pair[1])).groupByKey()             .filter(lambda pair: len(pair[1]) >= 2)             .flatMap(lambda pair: [i for i in combinations(pair[1], 2)]).distinct()             .map(lambda pair: (pair, compute_jaccard_sim(user_business_dict[pair[0]], user_business_dict[pair[1]])))             .filter(lambda pair: pair[1] >= 0.01)             .filter(lambda pair: check_co_rating(user_business_dict[pair[0][0]], user_business_dict[pair[0][1]]))             .map(lambda pair: pair[0])             .map(lambda pair: (pair, compute_pearson_sim(user_business_dict[pair[0]], user_business_dict[pair[1]])))             .filter(lambda pair: pair[1] > 0)             .map(lambda pair: {"u1": index_user_id[pair[0][0]], "u2": index_user_id[pair[0][1]], "sim": pair[1]}) # 497189 resulting pairs

    with open(model_file, 'w+') as out:
        for row in comparison_pairs.collect():
            out.writelines(json.dumps(row) + "\n")
            
    t2 = time.time()
    print("Duration: " + str(t2 - t1))


# In[ ]:




