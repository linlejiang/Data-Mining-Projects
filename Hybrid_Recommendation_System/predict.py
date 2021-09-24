#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import sys
import json
import time
import os

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
                        
    return [(i, temp[i][0]/temp[i][1]) for i in temp]

# item_based prediction: compute weighted average
def item_based_predict_score(ls):
    # input: ls of tuples (star, weight)
    numerator = sum([i[0]*i[1] for i in ls])
    denominator = sum([i[1] for i in ls])

    return numerator/denominator

def get_avg(business, user):
    
    if business in index_business_id:
        business_id = index_business_id[business]
    if user in index_user_id:
        user_id = index_user_id[user]

    a = 0.625 * business_avg.get(business_id, grand_avg)
    b = (1 - 0.625) * user_avg.get(user_id, grand_avg)
    return (a + b)

def process_predictive_score(predicted, avg):

    if avg > 4.5:
        return 5.0
    elif avg > 1.5:
        return avg
    else:
        a = 0.6 * predicted
        b = 0.4 * avg
        return round(a + b)
    
if __name__ == '__main__':
    t1 = time.time()

    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    test_file = sys.argv[1]
    output_file = sys.argv[2]
    
    train_file = '../resource/asnlib/publicdata/train_review.json'
    cf_model_file = './cf.model'
    user_avg_model_file = './user_avg.model'
    business_avg_model_file = './business_avg.model'
    N_neighbor = 9
    grand_avg = 3.7961611526341503

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    data = sc.textFile(train_file).map(lambda line: json.loads(line))                 .map(lambda item: (item['user_id'], item['business_id'], item['stars']))

    # create a user_id: index dict
    user_dict = data.map(lambda item: item[0]).distinct().zipWithIndex().collectAsMap()

    # create index_user_id dict
    index_user_id = {index: user_id for user_id, index in user_dict.items()}

    # create a business_id: index dict
    business_dict = data.map(lambda item: item[1]).distinct().zipWithIndex().collectAsMap()

    # create index_business_id dict
    index_business_id = {index: business_id for business_id, index in business_dict.items()}

    # read in user and business average models
    user_avg = sc.textFile(user_avg_model_file).map(lambda line: json.loads(line)).collect()[0]
    business_avg = sc.textFile(business_avg_model_file).map(lambda line: json.loads(line)).collect()[0]

    # item-based CF prediction 
    # some business_id are cold start problem and being filtered
    # return user, business pairs
    test_pair = sc.textFile(test_file).map(lambda line: json.loads(line))                     .map(lambda item: (item['user_id'], item['business_id']))                     .filter(lambda item: item[0] in user_dict).filter(lambda item: item[1] in business_dict)                     .map(lambda pair: (user_dict[pair[0]], business_dict[pair[1]]))

    user_business_rdd = data.map(lambda pair: (user_dict[pair[0]], (business_dict[pair[1]], pair[2]))).groupByKey().mapValues(compute_average)

    # create a user_business_dict {(user, rated_business_id): stars}
    user_business_dict = user_business_rdd.flatMap(lambda pair: [((pair[0], i[0]), i[1]) for i in pair[1]]).collectAsMap()

    # sorted ((business,business), sim) dict from training model
    item_data = sc.textFile(cf_model_file).map(lambda line: json.loads(line))                     .map(lambda item: (tuple(sorted((business_dict[item['b1']], business_dict[item['b2']]))), item['sim'])).collectAsMap()

    # for dd in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]: # p = 0.625, z = 0.4, j = 1, 1.1806339136929032, 
    test_result = test_pair.leftOuterJoin(user_business_rdd)                     .flatMapValues(lambda pair: [((pair[0], i[0]), tuple(sorted((pair[0], i[0])))) for i in pair[1]])                     .map(lambda pair: ((pair[0], pair[1][0]), pair[1]))                     .mapValues(lambda key: item_data.get(key[1], 0.2)).filter(lambda pair: pair[1] > 0)                     .map(lambda pair: ((pair[0][0], pair[0][1][0]), (pair[1], pair[0][1][1])))                     .groupByKey().mapValues(lambda pair: sorted(list(pair), key = lambda x: -x[0])[:9])                     .flatMap(lambda pair: [(pair[0], (user_business_dict[(pair[0][0], i[1])], i[0])) for i in pair[1]])                     .groupByKey().mapValues(item_based_predict_score)                     .map(lambda pair: (pair[0], (pair[1], get_avg(pair[0][1], pair[0][0]))))                     .mapValues(lambda scores: process_predictive_score(scores[0], scores[1]))                     .map(lambda pair: {"user_id": index_user_id[pair[0][0]], "business_id": index_business_id[pair[0][1]], "stars": pair[1]})

    # cold business and cold user
    cold_pair = sc.textFile(test_file).map(lambda line: json.loads(line))                     .map(lambda item: (item['user_id'], item['business_id']))                     .filter(lambda item: item[0] not in user_dict or item[1] not in business_dict)                     .map(lambda pair: {"user_id": pair[0], "business_id": pair[1], "stars": grand_avg})

    result = test_result.collect()
    test2 = cold_pair.collect()
    result.extend(test2)

    with open(output_file, 'w+') as out:
        for row in result:
            out.writelines(json.dumps(row) + "\n")

    t2 = time.time()
    print("Duration: " + str(t2 - t1))


# In[ ]:




