#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pyspark
import sys
import json
import csv
import time
import binascii

def minhash(x):
    hashed_value = []
    prime_num = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 
                 197, 199, 211, 223, 227, 229]
    i = 0
    while i <= 6:
        hashed_value.append((prime_num[i]*x + prime_num[-(i+1)]) % m)
        i += 1
    return hashed_value

def predict(city, check_set):
    if city == "" or city is None:
        yield 0
    else:
        new_index = int(binascii.hexlify(city.encode('utf8')), 16)
        if set(minhash(new_index)).issubset(check_set):
            yield 1
        else: 
            yield 0
            
if __name__ == '__main__':
    t1 = time.time()
    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]
    output_file = sys.argv[3]
    m = 7000

    # input_file_1 = 'business_first.json'
    # input_file_2 = 'business_second.json'
    # output_file = 'result1.csv'
    
    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    data = sc.textFile(input_file_1).map(lambda line: json.loads(line)).map(lambda item: item['city']).distinct()                 .filter(lambda city: city != '').map(lambda city: int(binascii.hexlify(city.encode('utf8')), 16))                 .flatMap(lambda city_index: minhash(city_index)).collect()
    check_set = set(data)

    result = sc.textFile(input_file_2).map(lambda line: json.loads(line)).map(lambda item: item['city'])                 .flatMap(lambda city: predict(city, check_set)).collect()

    with open(output_file, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ' ')
        writer.writerow(result)

    t2 = time.time()
    print("Duration: " + str(t2 - t1))


# In[33]:




