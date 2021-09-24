#!/usr/bin/env python
# coding: utf-8

# In[168]:


import pyspark
import sys
import json
from itertools import combinations
import time

# create hash functions
def minhash(x):
    hashed_value = []
    prime_num = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 
                 197, 199, 211, 223, 227, 229]
    i = 0
    while i <= 29:
        hashed_value.append((prime_num[i]*x + prime_num[-(i+1)]) % m)
        i += 1
    return hashed_value

# generate signatures given list_of_minhash_lists
def generate_signature(ls):
    signature = []
    i = 0
    while i <= 29:
        value = [999999999999999]
        for j in ls:
            if j[i] < value[0]:
                value[0] = j[i]
        signature.append(value[0])
        i += 1
    return signature

# LSH function
# input: one row of the signature matrix
# output: ((chunk_index, band_list), business_id) for groupByKey in LSH
def LSH(business_id, signature):
    output_list = []
    chunk_index = 0
    
    while chunk_index <= 29:
        output_list.append(((chunk_index, signature[chunk_index]), business_id))
        chunk_index += 1
        
    return output_list

def compute_jaccard_sim(set1, set2):
    return float(len(set1.intersection(set2))/len(set1.union(set2)))

if __name__ == '__main__':
    t1 = time.time()
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    data = sc.textFile(input_file).map(lambda line: json.loads(line))                 .map(lambda item: (item['user_id'], item['business_id'])).distinct()

    # create a user_id: index rdd
    user_rdd = data.map(lambda pair: pair[0]).distinct().zipWithIndex()
    m = user_rdd.count()*2

    user_dict = user_rdd.collectAsMap()

    # create a business_id: index dict
    business_dict = data.map(lambda pair: pair[1]).distinct().zipWithIndex().collectAsMap()

    # create business_user_signature
    business_user_signature = data.map(lambda pair: (business_dict[pair[1]], user_dict[pair[0]]))                                 .mapValues(minhash).groupByKey().mapValues(list).mapValues(generate_signature)

    # create business_user_dict
    business_user_dict = data.map(lambda pair: (business_dict[pair[1]], user_dict[pair[0]])).groupByKey()                             .mapValues(set).collectAsMap()

    # create index_business_id dict
    index_business_id = {index: business_id for business_id, index in business_dict.items()}

    # LSH algorithm to find pairs
    result = business_user_signature.flatMap(lambda pair: LSH(pair[0], pair[1])).groupByKey()         .filter(lambda pair: len(pair[1]) >= 2)         .flatMap(lambda pair: [i for i in combinations(pair[1], 2)]).distinct()         .map(lambda pair: (pair, compute_jaccard_sim(business_user_dict[pair[0]], business_user_dict[pair[1]])))         .filter(lambda pair: pair[1] >= 0.05)         .map(lambda pair: ((index_business_id[pair[0][0]], index_business_id[pair[0][1]]), pair[1])).collect()

    output = [{"b1": i[0][0], "b2": i[0][1], "sim": i[1]} for i in result]

    with open(output_file, 'w+') as out:
        for row in output:
            out.writelines(json.dumps(row) + "\n")
    
    t2 = time.time()
    print("Duration: " + str(t2 - t1))


# In[ ]:




