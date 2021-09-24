#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyspark
import sys
import json
import math
import time
from itertools import combinations
from operator import add

# apriori for generating candidate itemsets in SON Phase 1
def apriori(basket, support, num_of_partition):
    
    baskets = list(basket)
    candidate_dict = {}
    threshold = math.ceil(support * len(baskets) / total_basket_size)
#     threshold = support/num_of_partition
    
    # find singleton, k = 1
    singleton_dict = {}
    for i in baskets:
        for j in i:
            if j not in singleton_dict:
                singleton_dict[j] = 1
            else:
                singleton_dict[j] += 1
    
                
    temp_singleton_list = [i for i in singleton_dict if singleton_dict[i] >= threshold]  # for generating pairs
    candidate_dict['1'] = [(i, ) for i in temp_singleton_list]
    candidate_dict['1'].sort() 
                
    # find candidate pairs, k = 2
    pair_dict = {}
    possible_pairs = combinations(temp_singleton_list, 2)
    for i in possible_pairs:
        pair = set(i)
        for j in baskets:
            if pair.issubset(j):
                pair_key = tuple(sorted(pair))
                if pair_key not in pair_dict:
                    pair_dict[pair_key] = 1
                else:
                    pair_dict[pair_key] += 1
    candidate_dict['2'] = [i for i in pair_dict if pair_dict[i] >= threshold]
    candidate_dict['2'].sort()
    
    # find candidate itemsets, k >= 3
    k = 2
    temp_dict = {}
    temp_candidate_dict = {}
    while len(candidate_dict[str(k)]) != 0:
        temp_dict[str(k)] = {}
        temp_candidate_dict[str(k)] = []

        for i in range(len(candidate_dict[str(k)]) - 1):
            for j in range(i + 1, len(candidate_dict[str(k)])):
                if len(tuple(sorted(set(candidate_dict[str(k)][i] + candidate_dict[str(k)][j])))) == k + 1:
                    candidate_tuple = tuple(sorted(set(candidate_dict[str(k)][i] + candidate_dict[str(k)][j])))
                    if candidate_tuple not in temp_candidate_dict[str(k)]:
                        temp_candidate_dict[str(k)].append(candidate_tuple)
                else:
                    break
                    
        for i in temp_candidate_dict[str(k)]:
            itemset = set(i)
            for j in baskets:
                if itemset.issubset(j):
                    itemset_key = tuple(sorted(itemset))
                    if itemset_key not in temp_dict[str(k)]:
                        temp_dict[str(k)][itemset_key] = 1
                    else:
                        temp_dict[str(k)][itemset_key] += 1
        candidate_dict[str(k + 1)] = [i for i in temp_dict[str(k)] if temp_dict[str(k)][i] >= threshold]
        candidate_dict[str(k + 1)].sort()
        k += 1
        
    candidate_list = []
    for i in range(1, k):
        candidate_list.extend(candidate_dict[str(i)])
        
    return candidate_list

# count the candidates generated from SON Phase 1
def candidate_counter(basket, Phase1):
    
    baskets = list(basket)
    itemset_dict = {}
    
    for i in Phase1:
        if len(i) == 1:  # singleton
            for j in baskets:
                if set(i).issubset(j):
                    if i not in itemset_dict:
                        itemset_dict[i] = 1
                    else:
                        itemset_dict[i] += 1
        else:
            for j in baskets:
                if set(i).issubset(j):
                    if tuple(sorted(i)) not in itemset_dict:
                        itemset_dict[tuple(sorted(i))] = 1
                    else:
                        itemset_dict[tuple(sorted(i))] += 1
                    
    candidate_count_list = [(k, v) for k, v in itemset_dict.items()]

    return candidate_count_list

# format the output
def formatter(Phase1, Phase2, output_file):

    new_str1 = 'Candidates:\n' + str(Phase1[0])[:-2] + ')'
    i = 1
    candidate_size_checker = 1
    while i < len(Phase1):
        if len(Phase1[i]) == 1:
            new_str1 += ',' + str(Phase1[i])[:-2] + ')'
        elif len(Phase1[i]) != candidate_size_checker:
            candidate_size_checker += 1
            new_str1 += '\n\n' + str(Phase1[i])
        else:
            new_str1 += ',' + str(Phase1[i])
        i += 1

    new_str1 += '\n\n' + 'Frequent Itemsets:\n' + str(Phase2[0])[:-2] + ')'

    j = 1
    frequent_size_checker = 1
    while j < len(Phase2):
        if len(Phase2[j]) == 1:
            new_str1 += ',' + str(Phase2[j])[:-2] + ')'
        elif len(Phase2[j]) != frequent_size_checker:
            frequent_size_checker += 1
            new_str1 += '\n\n' + str(Phase2[j])
        else:
            new_str1 += ',' + str(Phase2[j])
        j += 1
        
    new_str1 += '\n\n'
        
    with open(output_file, 'w+') as f:
        f.write(new_str1)

if __name__ == '__main__':
        
    t1 = time.time()
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    sc = pyspark.SparkContext('local[*]', 'PySparkShell')

    lines = sc.textFile(input_file)
    header = lines.take(1)[0]
    data = lines.filter(lambda line: line != header)
    num_of_partition = lines.getNumPartitions()

    if case_number == 1:
        baskets = data.map(lambda line: (line.split(',')[0], line.split(',')[1])).groupByKey()                         .mapValues(set).map(lambda x: x[1])
    else:
        baskets = data.map(lambda line: (line.split(',')[1], line.split(',')[0])).groupByKey()                         .mapValues(set).map(lambda x: x[1])

    total_basket_size = baskets.count()

    #SON Phase 1
    Phase1 = baskets.mapPartitions(lambda basket: apriori(basket, support, total_basket_size))                     .map(lambda item: (item, 1)).groupByKey().map(lambda pairs: pairs[0])                     .sortBy(lambda tuples: (len(tuples), tuples)).collect() 

    #SON Phase 2
    Phase2 = baskets.mapPartitions(lambda basket: candidate_counter(basket, Phase1)).reduceByKey(add)                     .filter(lambda pairs: pairs[1] >= support).map(lambda pairs: pairs[0])                     .sortBy(lambda tuples: (len(tuples), tuples)).collect()

    formatter(Phase1, Phase2, output_file)
    t2 = time.time()
    print("Duration:", str(round(t2 - t1, 2)))

