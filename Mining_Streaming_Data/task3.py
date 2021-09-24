#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyspark
import sys
import csv
import random
import tweepy

class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        global sequence_num
        global hashtag_list
        global hashtag_dict
    
        hashtags = status.entities.get('hashtags')
        if len(hashtags) != 0:
            if sequence_num <= 100:
                temp = []
                for i in hashtags: # dict of hashtags in a tweet
                    temp.append(i['text'])
                    if i['text'] in hashtag_dict:
                        hashtag_dict[i['text']] += 1
                    else:
                        hashtag_dict[i['text']] = 1
                hashtag_list.append(temp)
                
            else:
                # decide whether to keep the new tweet
                if random.randint(1, sequence_num) <= 100:
                    temp = []
                    for i in hashtags:
                        temp.append(i['text'])
                        if i['text'] in hashtag_dict:
                            hashtag_dict[i['text']] += 1
                        else:
                            hashtag_dict[i['text']] = 1
                    remove_index = random.randint(0, 99)
                    for i in hashtag_list[remove_index]:
                        hashtag_dict[i] -= 1
                        if hashtag_dict[i] == 0:
                            del hashtag_dict[i]
                    hashtag_list[remove_index] = temp
                    
            sorted_hashtag_dict = sorted(hashtag_dict.items(), key=lambda pair: (-pair[1], pair[0]))
            print_criteria = sorted(set(hashtag_dict.values()), reverse=True)[:3]
            with open(output_file, 'a', encoding='utf-8') as csvfile:
                csvfile.write('The number of tweets with tags from the beginning: ' + str(sequence_num) + '\n')
                for key, value in sorted_hashtag_dict:
                    if value in print_criteria:
                        csvfile.write(key + ' : ' + str(value) + '\n')
                csvfile.write('\n')
            sequence_num += 1
    
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_error disconnects the stream
            return False

if __name__ == '__main__':
    API_KEY = ''
    API_SECRET = ''
    ACCESS_TOKEN = ''
    ACCESS_TOKEN_SECRET = ''
    topics = ['COVID19', 'Pandemic', 'SocialDistancing', 'StayAtHome', 'Trump', 'Quarantine', 'CoronaVirus', 'China', 'Wuhan']
    # port = sys.argv[1] # need to add this when submit
    output_file = sys.argv[2]
    sequence_num = 1
    hashtag_list = []
    hashtag_dict = {}

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    
    with open(output_file, 'w') as csvfile:
        csv.writer(csvfile)
            
    MyStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=MyStreamListener)
    myStream.filter(track=topics, languages=['en'])
    
    

