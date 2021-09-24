#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pyspark
import sys
import json
import math
import time

def compute_cosine_sim(set1, set2):
    if len(set1) >= 1 and len(set2) >= 1:
        sim = float(len(set(set1).intersection(set(set2)))/(math.sqrt(len(set1)) * math.sqrt(len(set2))))
    else:
        sim = 0
    return sim

if __name__ == '__main__':
# t1 = time.time()
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]

#     test_file = 'test_review.json'
#     model_file = 'task2.model'
#     output_file = 'task2.res'

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    data = sc.textFile(model_file).map(lambda line: json.loads(line))

    user_dict = data.filter(lambda item: item['data'] == 'user_dict')                     .map(lambda item: (item['key'], item['value'])).collectAsMap()
    business_dict = data.filter(lambda item: item['data'] == 'business_dict')                     .map(lambda item: (item['key'], item['value'])).collectAsMap()
    user_profile = data.filter(lambda item: item['data'] == 'user_profile')                     .map(lambda item: (item['key'], item['value'])).collectAsMap()
    business_profile = data.filter(lambda item: item['data'] == 'business_profile')                     .map(lambda item: (item['key'], item['value'])).collectAsMap()

    # create a index_user_id dict
    index_user_id = {index: user_id for user_id, index in user_dict.items()}

    # create a index_business_id dict
    index_business_id = {index: business_id for business_id, index in business_dict.items()}

    # prediction
    predict = sc.textFile(test_file).map(lambda line: json.loads(line))             .map(lambda item: (user_dict.get(item['user_id'], -1), business_dict.get(item['business_id'], -1)))             .filter(lambda pair: pair[0] != -1 and pair[1] != -1)             .map(lambda pair: ((pair), compute_cosine_sim(user_profile.get(pair[0], [-1]), business_profile.get(pair[1], [-2]))))             .filter(lambda pair: pair[1] > 0.01)             .map(lambda pair: ((index_user_id[pair[0][0]], index_business_id[pair[0][1]]), pair[1])).collect()

    output = [{"user_id": i[0][0], "business_id": i[0][1], "sim": i[1]} for i in predict]

    with open(output_file, 'w+') as out:
        for row in output:
            out.writelines(json.dumps(row) + "\n")

# t2 = time.time()
# print("Duration: " + str(t2 - t1))


# In[ ]:




