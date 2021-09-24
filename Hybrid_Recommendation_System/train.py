#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pyspark
import sys
import json
from itertools import combinations
import math
import time
import os

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

# to filter out pairs with less than co-rating threshold of 3
def check_co_rating(dict1, dict2):
    if len(set(dict1.keys()).intersection(set(dict2.keys()))) >= 3:
        return True
    else:
        return False
    
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
    
    return sim * abs(sim ** 2.5)

if __name__ == '__main__':

    t1 = time.time()
    train_file = '../resource/asnlib/publicdata/train_review.json'
    user_average = '../resource/asnlib/publicdata/user_avg.json'
    business_average = '../resource/asnlib/publicdata/business_avg.json'
    cf_model_file = './cf.model'
    user_avg_model_file = './user_avg.model'
    business_avg_model_file = './business_avg.model'
    
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    data = sc.textFile(train_file).map(lambda line: json.loads(line))                 .map(lambda item: (item['user_id'], item['business_id'], item['stars']))

    user_avg_dict = sc.textFile(user_average).map(lambda line: json.loads(line)).collect()[0]
    business_avg_dict = sc.textFile(business_average).map(lambda line: json.loads(line)).collect()[0]

    # compute user average that are not in the user_avg file
    computed_user_avg = data.map(lambda item: (item[0], item[2])).groupByKey().mapValues(lambda stars: sum(stars)/len(stars))                             .collectAsMap()

    # compute business average that are not in the business_avg file
    computed_business_avg = data.map(lambda item: (item[1], item[2])).groupByKey().mapValues(lambda stars: sum(stars)/len(stars))                             .collectAsMap()

    computed_user_avg.update(user_avg_dict)
    computed_business_avg.update(business_avg_dict)

    # create a user_id: index dict
    user_dict = data.map(lambda item: item[0]).distinct().zipWithIndex().collectAsMap()

    # create a business_id: index dict
    business_dict_rdd = data.map(lambda item: item[1]).distinct().zipWithIndex()
    business_dict = business_dict_rdd.collectAsMap()

    # create business_user_dict
    # return {business_id1: {{userid1, star}, {userid2, star}, ...}, business_id2: {{}, {}, ...}} dict
    business_user_dict = data.map(lambda pair: (business_dict[pair[1]], (user_dict[pair[0]], pair[2])))                 .groupByKey().mapValues(compute_average).filter(lambda pair: len(pair[1]) >= 3).collectAsMap()

    # create index_user_id dict
    index_user_id = {index: user_id for user_id, index in user_dict.items()}

    # create index_business_id dict
    index_business_id = {index: business_id for business_id, index in business_dict.items()}

    # generate all possible comparison pairs with at least 3 co-ratings, generate pairwise pearson similarity, sorted, and filtered
    business_count = len(business_dict)

    comparison_pairs = business_dict_rdd.map(lambda pair: (pair[1], pair[1]))                 .flatMapValues(lambda index: [i for i in range(index+1, business_count+1)])                 .filter(lambda pair: pair[0] in business_user_dict and pair[1] in business_user_dict)                 .filter(lambda pair: check_co_rating(business_user_dict[pair[0]], business_user_dict[pair[1]]))                 .map(lambda pair: (pair, compute_pearson_sim(business_user_dict[pair[0]], business_user_dict[pair[1]])))                 .filter(lambda pair: pair[1] > 0)                 .map(lambda pair: {"b1": index_business_id[pair[0][0]], "b2": index_business_id[pair[0][1]], "sim": pair[1]})

    with open(user_avg_model_file, 'w+') as out:
        out.write(json.dumps(computed_user_avg) + "\n")

    with open(business_avg_model_file, 'w+') as out:
        out.write(json.dumps(computed_business_avg) + "\n")

    with open(cf_model_file, 'w+') as out:
        for row in comparison_pairs.collect():
            out.writelines(json.dumps(row) + "\n")

    t2 = time.time()
    print("Duration: " + str(t2 - t1))


# In[ ]:




