#!/usr/bin/env python
# coding: utf-8

# In[13]:


from pyspark import SparkConf, SparkContext, SQLContext
import sys
import os
import time
from itertools import combinations

def compute_single_tree_betweeness(root, edges):
    tree = {}
    tree[root] = {'level': 0, 'parent_list': []}
    
    # parent_dict to keep track of nodes checked and the corresponding level
    parent_dict = {}
    parent_dict[root] = 0
    
    # next_level nodes to be checked
    checklist = []
    
    # a dict to identify parent_node: [childen list] relationship
    # also for filling in edge scores later
    par_child_dict = {}
    par_child_dict[root] = {}
    
    # level 1:
    for child in edges[root]:
        tree[child] = {'level': 0, 'parent_list': []}
        tree[child]['level'] += 1
        tree[child]['parent_list'].append(root)
        parent_dict[child] = tree[child]['level']
        checklist.append(child)
        par_child_dict[root][child] = 0
        tree[child]['path_num'] = 1
        tree[child]['parent_edge'] = {root: 1}
        
    # checking processes:
    start_check_index = 0
    
    while start_check_index < len(checklist):
        end_check_index = len(checklist)
        for node in checklist[start_check_index:end_check_index]:
            start_check_index += 1
            par_child_dict[node] = {}
            
            for i in edges[node]:
                # new level child
                if i not in parent_dict:
                    tree[i] = {}
                    tree[i]['level'] = tree[node]['level'] + 1
                    tree[i]['parent_list'] = [node]
                    parent_dict[i] = tree[i]['level']
                    checklist.append(i)
                    par_child_dict[node][i] = 0
                    
                else:
                    # node already checked, see if this node is the parent of node i
                    if tree[i]['level'] == tree[node]['level'] + 1 and i not in tree[node]['parent_list']:
                        tree[i]['parent_list'].append(node)
                        par_child_dict[node][i] = 0
            
            # add path_num to the node just checked, since all parent nodes had been identified
            if 'path_num' not in tree[node]:
                tree[node]['path_num'] = 0
                tree[node]['parent_edge'] = {}
                for i in tree[node]['parent_list']: 
                    tree[node]['path_num'] += tree[i]['path_num']
                    tree[node]['parent_edge'][i] = tree[i]['path_num']
                    
    # assign edge scores
    # create a backward list to ensure the scores were assigned in a backward order
    backward_list = [key for key, value in sorted(tree.items(), key = lambda value: -value[1]['level'])]

    for i in backward_list:
        if 'parent_edge' in tree[i]: # exclude root
            if bool(par_child_dict[i]) == False: # lowest level node with no child
                temp = 1
                for j in tree[i]['parent_edge']:
                    par_child_dict[j][i] += float(temp * tree[i]['parent_edge'][j]/tree[i]['path_num'])
            else:
                child_wt = sum([par_child_dict[i][j] for j in par_child_dict[i]])
                temp = 1 + child_wt
                for j in tree[i]['parent_edge']:
                    par_child_dict[j][i] += float(temp * tree[i]['parent_edge'][j]/tree[i]['path_num'])
    
    return par_child_dict

def recompute_betweeness_score():
    result_list = [compute_single_tree_betweeness(i, edges) for i in node_list]

    betweeness_result_dict = {}

    for dicts in result_list:
        for k in dicts:
            for j in dicts[k]:
                if j < k:
                    key = (j, k)
                else:
                    key = (k, j)
                if key not in betweeness_result_dict:
                    betweeness_result_dict[key] = dicts[k][j]
                else:
                    betweeness_result_dict[key] += dicts[k][j]

    betweeness_result = sorted(betweeness_result_dict.items(), key = lambda item: (-item[1], item[0][0]))
    return betweeness_result

def extract_communities():

    checked_dict = {}
    communities = []

    while list(set(node_list) - set(checked_dict.keys())) != []:
        checklist = []
        start_check_index = 0
        checklist = [list(set(node_list) - set(checked_dict.keys()))[0]]

        checked_dict[list(set(node_list) - set(checked_dict.keys()))[0]] = 1

        while start_check_index < len(checklist):
            end_check_index = len(checklist)
            for node in checklist[start_check_index:end_check_index]:
                start_check_index += 1

                for i in edges[node]:
                    if i not in checked_dict:
                        checklist.append(i)
                        checked_dict[i] = 1

        communities.append(sorted(checklist))

    return communities

def compute_modularity(communities):

    Q = 0
    for community in communities:
        for node1 in community:
            for node2 in community:
                if (node1, node2) in adjacent_matrix:
                    Q += 1 - degree_matrix[node1]*degree_matrix[node2]/(2*m)
                else:
                    Q += 0 - degree_matrix[node1]*degree_matrix[node2]/(2*m)
    Q /= (2*m)

    return Q

if __name__ == '__main__':
    t1 = time.time()

    filter_threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    betweeness_file = sys.argv[3]
    community_file = sys.argv[4]

    # filter_threshold = 7
    # input_file = 'ub_sample_data.csv'
    # betweeness_file = 'out_betw.txt'
    # community_file = 'out_comm.txt'

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
    a1 = user_pairs.map(lambda pair: pair[0]).distinct()
    a2 = user_pairs.map(lambda pair: pair[1]).distinct()
    # list of vertices
    vertices = a1.union(a2).distinct()
    node_list = vertices.collect()
    # a node: [list of connected nodes] dict
    edges = user_pairs.groupByKey().mapValues(list).collectAsMap()

    # compute betweeness scores of the edges by generating a tree for each node as the root
    result_list = vertices.map(lambda x: compute_single_tree_betweeness(x, edges)).collect()

    # aggregate betweeness scores
    betweeness_result_dict = {}
    for dicts in result_list:
        for k in dicts:
            for j in dicts[k]:
                if j < k:
                    key = (j, k)
                else:
                    key = (k, j)
                if key not in betweeness_result_dict:
                    betweeness_result_dict[key] = dicts[k][j]
                else:
                    betweeness_result_dict[key] += dicts[k][j]

    betweeness_result = sorted(betweeness_result_dict.items(), key = lambda item: (-item[1], item[0][0]))

    with open(betweeness_file, 'w+') as f:
        for i in betweeness_result:
            f.write(str(i[0]) + ', ' + str(i[1]/2) + '\n')

    adjacent_matrix = {}

    for i in edges:
        for j in edges[i]:
            adjacent_matrix[(i, j)] = 1

    m = len(adjacent_matrix)/2

    degree_matrix = {i: len(edges[i]) for i in edges}

    communities = extract_communities()
    Q = compute_modularity(communities)
    removed_edge_num = 0

    while removed_edge_num < m:

        check_betweeness_index = 0
        high_value = [betweeness_result[0][1]]
        for i in betweeness_result[check_betweeness_index:]:

            if high_value[0] == i[1]:
                edges[i[0][0]].remove(i[0][1])
                edges[i[0][1]].remove(i[0][0])
                high_value[0] = i[1]
                check_betweeness_index += 1
                removed_edge_num += 1

            else:
                high_value[0] = betweeness_result[check_betweeness_index][1]
                break

        temp_communities = extract_communities()
        temp_Q = compute_modularity(temp_communities)

        if Q <= temp_Q:
            Q = temp_Q
            communities = temp_communities

        betweeness_result = recompute_betweeness_score()

    with open(community_file, 'w+') as f:
        for i in sorted(communities, key = lambda item: (len(item), item)):
            f.write(str(i)[1:-1] + '\n')

    t2 = time.time()
    print("Duration:", str(round(t2 - t1, 2)))


# In[ ]:




