#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pyspark
import sys
import json
import csv
from itertools import combinations
import math
import random
import os
import time

def compute_euclidean_distance(list1, list2):
    return float(math.sqrt(sum([(i[0] - i[1])**2 for i in zip(list1, list2)])))

# compute Mahalanobis distance input: datapoint coordinates, DS centroid coordinates, DS cluster std list
def compute_mahalanobis_distance(list1, list2, std):
    return float(math.sqrt(sum([((i[0] - i[1])/i[2])**2 for i in zip(list1, list2, std)])))

def update_centroid(cluster_dict, data):
    temp_centroid_dict = {}
    for i in cluster_dict:
        cluster_list = []
        for j in cluster_dict[i]:
            cluster_list.append(data[j])
        temp_centroid_dict[i] = [sum(n)/len(n) for n in zip(*cluster_list)]
    return temp_centroid_dict

def compute_error(dict1, dict2):
    temp = 0
    for i in dict1:
        for j in zip(dict1[i], dict2[i]):
            temp += abs(j[0] - j[1])
    return temp

def compute_sum(data, lst_of_points, dimension):     
    temp_dict = {}
    num = len(lst_of_points)
    for i in range(dimension):
        cluster_list = []
        for j in lst_of_points:
            cluster_list.append(data[j])
        temp_dict['SUM'] = [sum(n) for n in zip(*cluster_list)]
        temp_dict['SUMSQ'] = [sum(m**2 for m in n) for n in zip(*cluster_list)]
        temp_dict['STD'] = [math.sqrt((i/num) - (j/num)**2) for i, j in zip(temp_dict['SUMSQ'], temp_dict['SUM'])]
    return temp_dict

def kmean(data, k, round_index):
    index = str(round_index)
    convergent = False
    counter = 0
    centroid_dict = {}
    cluster_dict = {}
    
    # init random centroid
    # random.seed(5)
    # adjust k to avoid sampling error
    if len(data) < k:
        k = len(data)
    random_centroid_id_list = random.sample(data.keys(), k)
    for i in range(len(random_centroid_id_list)):
        centroid_dict[index + str(i)] = data[random_centroid_id_list[i]]
    # 1st round
    distance_dict = {}
    for point in data:
        distance_dict[point] = {'centroid': 9999, 'distance': 999999999}
        for i in range(len(random_centroid_id_list)):
            distance = compute_euclidean_distance(centroid_dict[index + str(i)], data[point])
            if distance_dict[point]['distance'] > distance:
                distance_dict[point]['centroid'] = index + str(i)
                distance_dict[point]['distance'] = distance
        if distance_dict[point]['centroid'] not in cluster_dict:
            cluster_dict[distance_dict[point]['centroid']] = [point]
        else:
            cluster_dict[distance_dict[point]['centroid']].append(point)

    new_centroid_dict = update_centroid(cluster_dict, data)
    error = compute_error(new_centroid_dict, centroid_dict)

    # 2nd round and on...
    while convergent == False:
        # error can be adjusted
        if counter == 30 or error < 0.01:
            break
        counter += 1
        cluster_dict = {}
        new_centroid_dict_1 = new_centroid_dict  # assign new centroid dict
        distance_dict = {}
        for point in data:
            distance_dict[point] = {'centroid': 9999, 'distance': 999999999}
            for i in new_centroid_dict_1:
                distance = compute_euclidean_distance(new_centroid_dict_1[i], data[point])
                if distance_dict[point]['distance'] > distance:
                    distance_dict[point]['centroid'] = i
                    distance_dict[point]['distance'] = distance
            if distance_dict[point]['centroid'] not in cluster_dict:
                cluster_dict[distance_dict[point]['centroid']] = [point]
            else:
                cluster_dict[distance_dict[point]['centroid']].append(point)

        new_centroid_dict = update_centroid(cluster_dict, data)
        error = compute_error(new_centroid_dict, new_centroid_dict_1)
        
    DS_stat = {}
    for i in new_centroid_dict:
        DS_stat[i] = {}
        DS_stat[i]['N'] = len(cluster_dict[i])
        temp = compute_sum(data, cluster_dict[i], dimension)
        DS_stat[i]['SUM'] = temp['SUM']
        DS_stat[i]['SUMSQ'] = temp['SUMSQ']
        DS_stat[i]['STD'] = temp['STD']
        
    return new_centroid_dict, cluster_dict, DS_stat
    
def assign_cs_or_rs(centroid_dict, cluster_dict, cluster_stat, retain_dict):
    for i in list(cluster_stat.keys()):
        if cluster_stat[i]['N'] == 1:
            retain_dict.update({cluster_dict[i][0]: centroid_dict[i]})
            del centroid_dict[i]
            del cluster_dict[i]
            del cluster_stat[i]  
        elif cluster_stat[i]['N'] == 0:
            del centroid_dict[i]
            del cluster_dict[i]
            del cluster_stat[i]
        else:
            [retain_dict.pop(key) for key in cluster_dict[i]]
    return centroid_dict, cluster_dict, cluster_stat

# assign individual point to DS, CS, or RS (input datapoint: (id, [coordinates]) tuple)
# md_thre alpha can be adjusted (the pre-define alpha is 3, here all threshold can be adjusted by multiplying a num)
def point_to_set(datapoint):
    cluster_id = -1
    temp = 100
    # try to assign to DS
    for i in DS_centroid_dict:
        std1 = DS_cluster_stat[i]['STD']
        temp_distance = compute_mahalanobis_distance(datapoint[1], DS_centroid_dict[i], std1)
        if temp_distance <= md_thre and temp_distance < temp:
            cluster_id = i
            temp = temp_distance
    if cluster_id != -1:
        return ('DS', cluster_id)
    else:
        if CS_cluster_dict != {}: # after initialization, the compression set could be empty
            # try to assign to CS
            for i in CS_centroid_dict:
                std2 = CS_cluster_stat[i]['STD']
                temp_distance = compute_mahalanobis_distance(datapoint[1], CS_centroid_dict[i], std2)
                if temp_distance <= md_thre and temp_distance < temp:
                    cluster_id = i
                    temp = temp_distance
            if cluster_id != -1:
                return ('CS', cluster_id)
            else:
                return ('RS', cluster_id)
        else:
            return ('RS', cluster_id)

# update centroid after added new points
def update_cluster_centroid(centroid_list, new_points_dict, old_num):
    new_temp_num = len(list(new_points_dict.values()))
    new_temp_sum = [sum(i) for i in zip(*list(new_points_dict.values()))]
    new_centroid_list = [((x*old_num + y)/(new_temp_num + old_num)) for x, y in zip(centroid_list, new_temp_sum)]
    return new_centroid_list

# update centroid, dictionary and stat of the clusters
def update_cluster(post_md_dict):
    for i in post_md_dict:
        if i[0] == 'RS':
            retain_dict.update(post_md_dict[i])
        elif i[0] == 'DS':
            DS_cluster_dict[i[1]].extend(post_md_dict[i])
            old_num = DS_cluster_stat[i[1]]['N']
            DS_centroid_dict[i[1]] = update_cluster_centroid(DS_centroid_dict[i[1]], post_md_dict[i], old_num)
            new_n = len(post_md_dict[i])
            new_sum = [sum(i) for i in zip(*list(post_md_dict[i].values()))]
            new_sumsq = [sum(m**2 for m in n) for n in zip(*list(post_md_dict[i].values()))]
            DS_cluster_stat[i[1]]['N'] += new_n
            num = DS_cluster_stat[i[1]]['N']
            DS_cluster_stat[i[1]]['SUM'] = [x + y for x, y in zip(DS_cluster_stat[i[1]]['SUM'], new_sum)]
            DS_cluster_stat[i[1]]['SUMSQ'] = [x + y for x, y in zip(DS_cluster_stat[i[1]]['SUMSQ'], new_sumsq)]
            DS_cluster_stat[i[1]]['STD'] = [math.sqrt((i/num) - (j/num)**2) for i, j in zip(DS_cluster_stat[i[1]]['SUMSQ'], DS_cluster_stat[i[1]]['SUM'])]
        else:
            CS_cluster_dict[i[1]].extend(post_md_dict[i])
            old_num = CS_cluster_stat[i[1]]['N']
            CS_centroid_dict[i[1]] = update_cluster_centroid(CS_centroid_dict[i[1]], post_md_dict[i], old_num)
            new_n = len(post_md_dict[i])
            new_sum = [sum(i) for i in zip(*list(post_md_dict[i].values()))]
            new_sumsq = [sum(m**2 for m in n) for n in zip(*list(post_md_dict[i].values()))]
            CS_cluster_stat[i[1]]['N'] += new_n
            num = CS_cluster_stat[i[1]]['N']
            CS_cluster_stat[i[1]]['SUM'] = [x + y for x, y in zip(CS_cluster_stat[i[1]]['SUM'], new_sum)]
            CS_cluster_stat[i[1]]['SUMSQ'] = [x + y for x, y in zip(CS_cluster_stat[i[1]]['SUMSQ'], new_sumsq)]
            CS_cluster_stat[i[1]]['STD'] = [math.sqrt((i/num) - (j/num)**2) for i, j in zip(CS_cluster_stat[i[1]]['SUMSQ'], CS_cluster_stat[i[1]]['SUM'])]
            
# merge CS clusters
def merge_cluster(CS_centroid_dict, CS_cluster_dict, CS_cluster_stat, round_index):
    temp_index = 0
    if CS_centroid_dict != {}:
        for i, j in combinations(CS_centroid_dict.keys(),2):
            if i in CS_centroid_dict and j in CS_centroid_dict:
                if compute_mahalanobis_distance(CS_centroid_dict[i], CS_centroid_dict[j], CS_cluster_stat[i]['STD']) < md_thre:
                    # update cluster_centroid
                    CS_centroid_dict[('MS', str(round_index) + str(temp_index))] = update_cluster_centroid(CS_centroid_dict[i], {j: CS_centroid_dict[j]}, CS_cluster_stat[i]['N'])
                    del CS_centroid_dict[i]
                    del CS_centroid_dict[j]
                    # update cluster_dict
                    CS_cluster_dict[('MS', str(round_index) + str(temp_index))] = CS_cluster_dict[i]
                    CS_cluster_dict[('MS', str(round_index) + str(temp_index))].extend(CS_cluster_dict[j])
                    del CS_cluster_dict[i]
                    del CS_cluster_dict[j]
                    # update cluster_stat
                    CS_cluster_stat[('MS', str(round_index) + str(temp_index))] = {}
                    CS_cluster_stat[('MS', str(round_index) + str(temp_index))]['N'] = CS_cluster_stat[i]['N'] + CS_cluster_stat[j]['N']
                    num = CS_cluster_stat[('MS', str(round_index) + str(temp_index))]['N']
                    CS_cluster_stat[('MS', str(round_index) + str(temp_index))]['SUM'] = CS_cluster_stat[i]['SUM'] + CS_cluster_stat[j]['SUM']
                    CS_cluster_stat[('MS', str(round_index) + str(temp_index))]['SUMSQ'] = CS_cluster_stat[i]['SUMSQ'] + CS_cluster_stat[j]['SUMSQ']
                    CS_cluster_stat[('MS', str(round_index) + str(temp_index))]['STD'] = [math.sqrt((i/num) - (j/num)**2) for i, j in zip(CS_cluster_stat[('MS', str(round_index) + str(temp_index))]['SUMSQ'], CS_cluster_stat[('MS', str(round_index) + str(temp_index))]['SUM'])]
                    del CS_cluster_stat[i]
                    del CS_cluster_stat[j]
                    temp_index += 1

# merge CS to DS
# used euclidean distance between centroids as criteria
def merge_cs_rs_ds():
    for i in list(CS_centroid_dict.keys()):
        temp = 9999999999999
        to_ds_id = -1
        for j in list(DS_centroid_dict.keys()):
            distance = compute_euclidean_distance(CS_centroid_dict[i], DS_centroid_dict[j])
            if distance < temp:
                temp = distance
                to_ds_id = j
        # update DS cluster dict
        DS_cluster_dict[to_ds_id].extend(CS_cluster_dict[i])
        # update DS centroid
        DS_centroid_dict[to_ds_id] = update_cluster_centroid(DS_centroid_dict[to_ds_id], {to_ds_id: DS_cluster_dict[to_ds_id]}, DS_cluster_stat[to_ds_id]['N'])
        # update DS stat
        new_n = len(CS_cluster_dict[i])
        new_sum = CS_cluster_stat[i]['SUM']
        new_sumsq = CS_cluster_stat[i]['SUMSQ']
        DS_cluster_stat[to_ds_id]['N'] += new_n
        num = DS_cluster_stat[to_ds_id]['N']
        DS_cluster_stat[to_ds_id]['SUM'] = [x + y for x, y in zip(DS_cluster_stat[to_ds_id]['SUM'], new_sum)]
        DS_cluster_stat[to_ds_id]['SUMSQ'] = [x + y for x, y in zip(DS_cluster_stat[to_ds_id]['SUMSQ'], new_sumsq)]
        DS_cluster_stat[to_ds_id]['STD'] = [math.sqrt((i/num) - (j/num)**2) for i, j in zip(DS_cluster_stat[to_ds_id]['SUMSQ'], DS_cluster_stat[to_ds_id]['SUM'])]
        CS_cluster_dict[i] = {}

if __name__ == '__main__':
    t1 = time.time()
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file1 = sys.argv[3]
    output_file2 = sys.argv[4]

#     input_file = 'data/test1'
#     n_cluster = 10
#     output_file1 = 'cluster1.json'
#     output_file2 = 'intermediate1.csv'

    sc = pyspark.SparkContext('local[*]', 'PySparkShell')
    sc.setLogLevel("WARN")

    # get files
    file_path = []
    for file in sorted(os.listdir(input_file)):
        # Check whether file is in txt format
        if file.endswith(".txt"):
            file_path.append(f"{input_file}/{file}")
    round_index = 2

    # Step 1 Initialization
    data = sc.textFile(file_path[0]).map(lambda line: line.split(','))                 .map(lambda line: (int(line[0]), list(map(float, line[1:]))))
    initial_size = 0.2*data.count()

    # Step 2 
    # init_data can be adjusted
    # used first 20 percent of the datapoints as starting point
    init_data = data.filter(lambda pair: pair[0] <= initial_size).collectAsMap()
    dimension = len(list(init_data.values())[0])
    # Mahalanobis distance threshold
    md_thre = 2*math.sqrt(dimension)

    # Step 3
    # few_points can be adjusted (the 'very few' criteria)
    few_points = 2
    outlier_points = []

    # factor to number of n_cluster can be adjusted
    centroid_dict, cluster_dict, cluster_stat = kmean(init_data, 4*n_cluster, 0)
    for i in cluster_stat:
        if cluster_stat[i]['N'] <= few_points:
            outlier_points.extend(cluster_dict[i])
    inlier_points_data = {i:init_data[i] for i in list(set(init_data.keys()) - set(outlier_points))}
    outlier_points_data = {i:init_data[i] for i in outlier_points}

    # Step 4
    DS_centroid_dict, DS_cluster_dict, DS_cluster_stat = kmean(inlier_points_data, n_cluster, 0)
    retain_dict = {}

    # Step 5 & 6
    # factor to number of n_cluster can be adjusted
    temp_centroid_dict, temp_cluster_dict, temp_cluster_stat = kmean(outlier_points_data, 4*n_cluster, 0)
    # distinguish compression set and retain set
    CS_centroid_dict, CS_cluster_dict, CS_cluster_stat = assign_cs_or_rs(temp_centroid_dict, temp_cluster_dict, temp_cluster_stat, retain_dict)

    # Step 7 - 10
    # load the rest of the data from the first data file
    # assign individual point to DS, CS, or RS
    # output {(set, id): {datapoints...}}
    post_md_dict = data.filter(lambda pair: pair[0] > initial_size).map(lambda pair: (point_to_set(pair), pair))                 .groupByKey().mapValues(dict).collectAsMap()

    update_cluster(post_md_dict)

    # Step 11
    # factor to number of n_cluster can be adjusted

    temp_centroid_dict_1, temp_cluster_dict_1, temp_cluster_stat_1 = kmean(retain_dict, 4*n_cluster, 1)
    CS_centroid_dict_1, CS_cluster_dict_1, CS_cluster_stat_1 = assign_cs_or_rs(temp_centroid_dict_1, temp_cluster_dict_1, temp_cluster_stat_1, retain_dict)
    CS_centroid_dict.update(CS_centroid_dict_1)
    CS_cluster_dict.update(CS_cluster_dict_1)
    CS_cluster_stat.update(CS_cluster_stat_1)

    # Step 12
    merge_cluster(CS_centroid_dict, CS_cluster_dict, CS_cluster_stat, 1)
    intermediate_result = []
    header = ["round_id", "nof_cluster_discard", "nof_point_discard", "nof_cluster_compression", "nof_point_compression",
                "nof_point_retained"]
    intermediate_result.append({"round_id": 1,
                                "nof_cluster_discard": len(DS_cluster_dict),
                                "nof_point_discard": sum([len(DS_cluster_dict[i]) for i in DS_cluster_dict]),
                                "nof_cluster_compression": len(CS_cluster_dict),
                                "nof_point_compression": sum([len(CS_cluster_dict[i]) for i in CS_cluster_dict]),
                                "nof_point_retained": len(retain_dict)})    

    while round_index <= len(file_path):
        new_round_data = sc.textFile(file_path[round_index - 1]).map(lambda line: line.split(','))                 .map(lambda line: (int(line[0]), list(map(float, line[1:]))))
        new_round_md_dict = new_round_data.map(lambda pair: (point_to_set(pair), pair)).groupByKey()                                             .mapValues(dict).collectAsMap()

        update_cluster(new_round_md_dict)

        # factor to number of n_cluster can be adjusted
        new_round_centroid_dict_1, new_round_cluster_dict_1, new_round_cluster_stat_1 = kmean(retain_dict, 4*n_cluster, round_index)
        new_round_CS_centroid_dict, new_round_CS_cluster_dict, new_round_CS_cluster_stat = assign_cs_or_rs(new_round_centroid_dict_1, new_round_cluster_dict_1, new_round_cluster_stat_1, retain_dict)
        CS_centroid_dict.update(new_round_CS_centroid_dict)
        CS_cluster_dict.update(new_round_CS_cluster_dict)
        CS_cluster_stat.update(new_round_CS_cluster_stat)

        merge_cluster(CS_centroid_dict, CS_cluster_dict, CS_cluster_stat, round_index)

        # final round merge CS to DS
        if round_index == len(file_path):
            merge_cs_rs_ds()
            CS_cluster_dict = {}

        intermediate_result.append({"round_id": round_index,
                                "nof_cluster_discard": len(DS_cluster_dict),
                                "nof_point_discard": sum([len(DS_cluster_dict[i]) for i in DS_cluster_dict]),
                                "nof_cluster_compression": len(CS_cluster_dict),
                                "nof_point_compression": sum([len(CS_cluster_dict[i]) for i in CS_cluster_dict]),
                                "nof_point_retained": len(retain_dict)})    

        round_index += 1

    # write to immediate csv
    with open(output_file2, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = header)
        writer.writeheader()
        for i in intermediate_result:
            writer.writerow(i)

    # write final result to json
    final_result = {}
    for key, value in DS_cluster_dict.items():
        final_result.update({str(i): int(key[1:]) for i in value})
    final_result.update({str(i): -1 for i in retain_dict})

    with open(output_file1, 'w') as out:
        json.dump(final_result, out)

    t2 = time.time()
    print("Duration: " + str(t2 - t1))


# In[ ]:





# In[ ]:




