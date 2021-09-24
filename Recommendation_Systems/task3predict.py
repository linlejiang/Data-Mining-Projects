#!/usr/bin/env python
# coding: utf-8

# In[179]:


import pyspark
import sys
import json
import time

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

def compute_user_average(x):
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
    
    pair = [(i, temp[i][0]/temp[i][1]) for i in temp]
    numerator = sum([i[1] for i in pair])
    denominator = len(pair)
                        
    return numerator/denominator

# user_based prediction: compute weighted average
def user_based_predict_score(ls, user_x):
    # input: ls of tuples (user_id, (star, weight))
    numerator = sum([(i[1][0] - user_average_dict[i[0]])*i[1][1] for i in ls])
    denominator = sum([i[1][1] for i in ls])
    output = user_average_dict[user_x] + numerator/denominator
    
    if output > 5:
        return 5
    elif output < 0:
        return 0
    else:
        return output

if __name__ == '__main__':

    t1 = time.time()
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    output_file = sys.argv[4]
    cf_type = sys.argv[5]
    N_neighbor = 9

# train_file = 'train_review.json'
# test_file = 'test_review_ratings.json'
# model_file = 'task3user.model'
# output_file = 'task3user.predict'
# cf_type = 'user_based'

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    data = sc.textFile(train_file).map(lambda line: json.loads(line))             .map(lambda item: (item['user_id'], item['business_id'], item['stars']))

    # create a user_id: index dict
    user_dict = data.map(lambda item: item[0]).distinct().zipWithIndex().collectAsMap()

    # create index_user_id dict
    index_user_id = {index: user_id for user_id, index in user_dict.items()}

    # create a business_id: index dict
    business_dict = data.map(lambda item: item[1]).distinct().zipWithIndex().collectAsMap()

    # create index_business_id dict
    index_business_id = {index: business_id for business_id, index in business_dict.items()}

    if cf_type == 'item_based':

        # some business_id are cold start problem and being filtered
        # return user, business pairs
        test_pair = sc.textFile(test_file).map(lambda line: json.loads(line))                 .map(lambda item: (item['user_id'], item['business_id']))                 .filter(lambda item: item[0] in user_dict).filter(lambda item: item[1] in business_dict)                 .map(lambda pair: (user_dict[pair[0]], business_dict[pair[1]]))

        user_business_rdd = data.map(lambda pair: (user_dict[pair[0]], (business_dict[pair[1]], pair[2])))                 .groupByKey().mapValues(compute_average)

        # create a user_business_dict {(user, rated_business_id): stars}
        user_business_dict = user_business_rdd.flatMap(lambda pair: [((pair[0], i[0]), i[1]) for i in pair[1]]).collectAsMap()

        # sorted ((business,business), sim) dict from training model
        item_data = sc.textFile(model_file).map(lambda line: json.loads(line))                 .map(lambda item: (tuple(sorted((business_dict[item['b1']], business_dict[item['b2']]))), item['sim'])).collectAsMap()

        # item-based prediction
        # join the test_pairs with the user_business_rating
        # for the test user, use flatMapValues to generate and sort all business comparison pairs (b?, b1), (b?, b2) ...
        # filter out business_ids that are not in the model file
        # match, ranked the business pairs by similarity
        # compute predicted scores
        result = test_pair.leftOuterJoin(user_business_rdd)                 .flatMapValues(lambda pair: [((pair[0], i[0]), tuple(sorted((pair[0], i[0])))) for i in pair[1]])                 .map(lambda pair: ((pair[0], pair[1][0]), pair[1]))                 .mapValues(lambda key: item_data.get(key[1], -1)).filter(lambda pair: pair[1] > 0)                 .map(lambda pair: ((pair[0][0], pair[0][1][0]), (pair[1], pair[0][1][1])))                 .groupByKey().mapValues(lambda pair: sorted(list(pair), key = lambda x: -x[0])[:N_neighbor])                 .flatMap(lambda pair: [(pair[0], (user_business_dict[(pair[0][0], i[1])], i[0])) for i in pair[1]])                 .groupByKey().mapValues(item_based_predict_score)                 .map(lambda pair: {"user_id": index_user_id[pair[0][0]], "business_id": index_business_id[pair[0][1]], "stars": pair[1]})

    elif cf_type == 'user_based':

        # some business_id are cold start problem and being filtered
        # return business, user pairs
        test_pair = sc.textFile(test_file).map(lambda line: json.loads(line))                 .map(lambda item: (item['business_id'], item['user_id']))                 .filter(lambda item: item[0] in business_dict).filter(lambda item: item[1] in user_dict)                 .map(lambda pair: (business_dict[pair[0]], user_dict[pair[1]]))

        business_user_rdd = data.map(lambda pair: (business_dict[pair[1]], (user_dict[pair[0]], pair[2])))                 .groupByKey().mapValues(compute_average)

        # create a business_user_dict {(business_id, rated_user): stars}
        business_user_dict = business_user_rdd.flatMap(lambda pair: [((pair[0], i[0]), i[1]) for i in pair[1]]).collectAsMap()

        # sorted ((user, user), sim) dict from training model
        user_data = sc.textFile(model_file).map(lambda line: json.loads(line))                 .map(lambda item: (tuple(sorted((user_dict[item['u1']], user_dict[item['u2']]))), item['sim'])).collectAsMap()

        # compute user_average dict
        user_average_dict = data.map(lambda pair: (user_dict[pair[0]], (business_dict[pair[1]], pair[2])))                     .groupByKey().mapValues(compute_user_average).collectAsMap()

        # user-based prediction
        # join the test_pairs with the business_user_rating
        # for the test business, use flatMapValues to generate and sort all user comparison pairs (u?, u1), (u?, u2) ...
        # filter out user_ids that are not in the model file
        # match, ranked the user pairs by similarity
        # compute predicted scores
        result = test_pair.leftOuterJoin(business_user_rdd)                 .flatMapValues(lambda pair: [((pair[0], i[0]), tuple(sorted((pair[0], i[0])))) for i in pair[1]])                 .map(lambda pair: ((pair[0], pair[1][0]), pair[1]))                 .mapValues(lambda key: user_data.get(key[1], -1)).filter(lambda pair: pair[1] > 0)                 .map(lambda pair: ((pair[0][0], pair[0][1][0]), (pair[1], pair[0][1][1])))                 .groupByKey().mapValues(lambda pair: sorted(list(pair), key = lambda x: -x[0])[:N_neighbor])                 .flatMap(lambda pair: [(pair[0], (i[1], (business_user_dict[(pair[0][0], i[1])], i[0]))) for i in pair[1]])                 .groupByKey().mapValues(list).map(lambda pair: (pair[0], user_based_predict_score(pair[1], pair[0][1])))                 .map(lambda pair: {"user_id": index_user_id[pair[0][1]], "business_id": index_business_id[pair[0][0]], "stars": pair[1]})

    with open(output_file, 'w+') as out:
        for row in result.collect():
            out.writelines(json.dumps(row) + "\n")

    t2 = time.time()
    print("Duration: " + str(t2 - t1))


# In[ ]:




