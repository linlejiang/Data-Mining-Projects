#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import sys
import math
import time
from pyspark.mllib.fpm import FPGrowth

if __name__ == '__main__':
        
    t1 = time.time()
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

#     input_file = 'user_business.csv'
#     filter_threshold = 70
#     support = 90

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    lines = sc.textFile(input_file)
    header = lines.take(1)[0]
    data = lines.filter(lambda line: line != header)

    baskets = data.map(lambda line: (line.split(',')[0], line.split(',')[1])).groupByKey()                         .mapValues(set).filter(lambda x: len(x[1]) > filter_threshold).map(lambda x: x[1])

    total_basket_size = baskets.count()
    minSupport = float(support / total_basket_size)
    numPartitions = lines.getNumPartitions()

    model = FPGrowth.train(baskets, minSupport, numPartitions)
    result = model.freqItemsets().collect()

    task3_frequent_itemsets = []
    for items, values in result:
        task3_frequent_itemsets.append(sorted(items))

    task3_frequent_itemsets

    h2 = []
    with open("out_t2.csv", "r") as f:
        for line in f:
            h2.append(line.strip())

    task2_frequent_itemsets = []

    for i in h2:
        task2_frequent_itemsets.append(sorted(i.split('?')))

    output = ['Task2,' + str(len(task2_frequent_itemsets)),
    'Task3,' + str(len(task3_frequent_itemsets)),
    'Intersection,' + str(len([i for i in task2_frequent_itemsets if i in task3_frequent_itemsets]))]

    with open(output_file, "w") as outf:
        outf.write("\n".join(item for item in output))

    t2 = time.time()
    print("Duration:", str(t2 - t1))


# In[ ]:




