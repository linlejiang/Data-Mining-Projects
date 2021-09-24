#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pyspark
import sys
import json
import datetime
import time
import binascii
import random
from pyspark.streaming import StreamingContext

def fm(rdd):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    ground_truth = rdd.distinct().collect()
    estimate = []
    i = 0
    while i <= 199:
        result = -1
        j = random.randint(0,29400000)
        k = random.randint(0,29400000)
        for city in ground_truth:
            index = int(binascii.hexlify(city.encode('utf8')), 16)
            hash_value = '{0:032b}'.format((j*index + k) % m)
            trailing_zeros = len(hash_value) - len(hash_value.rstrip('0'))
            if trailing_zeros > result:
                result = trailing_zeros
        estimate.append(result)
        i += 1

    estimates = []
    j = 0
    while j <= 19:
        avg = sum(estimate[10*j:10*(j+1)])/10
        estimates.append(avg)
        j += 1
    new_estimate = 2**((sorted(estimates)[9] + sorted(estimates)[10])*0.5)
    with open(output_file, "a") as csvfile:
        csvfile.write(timestamp + "," + str(len(ground_truth)) + "," + str(new_estimate) + "\n")
        print(timestamp + "," + str(len(ground_truth)) + "," + str(new_estimate) + "\n")
    # return estimate  ## cannot do return here, otherwise it will only produce one round

if __name__ == '__main__':
    port = int(sys.argv[1])
    output_file = sys.argv[2]
    m = 2333333333
    estimates = []
    with open(output_file, "w") as csvfile:
        csvfile.write("Time,Ground Truth,Estimation\n")
        
    sc = pyspark.SparkContext('local[*]', 'PySparkShell')
    ssc = StreamingContext(sc, 5)
    streaming = ssc.socketTextStream('localhost', port).window(30,10).map(lambda line: json.loads(line))                     .map(lambda item: item['city']).filter(lambda city: city != '').foreachRDD(fm)

    ssc.start()
    ssc.awaitTermination()


# In[ ]:




