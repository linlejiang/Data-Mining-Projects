#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark import SparkConf, SparkContext, SQLContext
import sys
import os
import time
from itertools import combinations
from graphframes import GraphFrame

# os.environ["PYSPARK_SUBMIT_ARGS"] = (
#     "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

if __name__ == '__main__':
    t1 = time.time()

    filter_threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    conf = SparkConf().setMaster("local[*]")         .set("spark.executor.memory", "4g")         .set("spark.driver.memory", "4g")
    sc = SparkContext.getOrCreate(conf = conf)
    sqlContext = SQLContext(sc)
    sc.setLogLevel("ERROR")

    lines = sc.textFile(input_file)
    header = lines.take(1)[0]
    user_business_dict = lines.filter(lambda line: line != header)             .map(lambda line: (line.split(',')[0], line.split(',')[1])).groupByKey().mapValues(set).collectAsMap()
    l1 = list(user_business_dict.keys())
    l2 = l1[::-1]
    candidate_pairs = list(combinations(l1, 2))
    candidate_pairs.extend(list(combinations(l2, 2)))
    user_pairs = sc.parallelize(candidate_pairs)         .filter(lambda pair: len(user_business_dict[pair[0]].intersection(user_business_dict[pair[1]])) >= filter_threshold).distinct()
    a1 = user_pairs.map(lambda pair: (pair[0],)).distinct()
    a2 = user_pairs.map(lambda pair: (pair[1],)).distinct()
    vertices_list = a1.union(a2).distinct()

    vertices = sqlContext.createDataFrame(vertices_list, ["id"])
    edges = sqlContext.createDataFrame(user_pairs, ["src", "dst"])
    g = GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)

    out = result.rdd.map(lambda x: (x['label'], x['id'])).groupByKey()         .map(lambda pair: sorted(list(pair[1]))).sortBy(lambda user: (len(user), user))

    with open(output_file, 'w+') as f:
        for i in out.collect():
            f.write(str(i)[1:-1] + '\n')

    t2 = time.time()
    print("Duration:", str(round(t2 - t1, 2)))


# In[ ]:




