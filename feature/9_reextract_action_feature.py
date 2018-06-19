# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime

# from datetime import datetime

dire = '../../data/'
start = datetime.datetime.now()
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderFuture_train = pd.read_csv(dire + 'train/orderFuture_train.csv', encoding='utf-8')
userProfile_train = pd.read_csv(dire + 'train/userProfile_train.csv', encoding='utf-8')
userComment_train = pd.read_csv(dire + 'train/userComment_train.csv', encoding='utf-8')
action_train = pd.read_csv(dire + 'train/insert_action_train2.csv', encoding='utf-8')
city = pd.read_csv(dire + 'train/city.csv', encoding='utf-8')

orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
orderFuture_test = pd.read_csv(dire + 'test/orderFuture_test.csv', encoding='utf-8')
userProfile_test = pd.read_csv(dire + 'test/userProfile_test.csv', encoding='utf-8')
userComment_test = pd.read_csv(dire + 'test/userComment_test.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/insert_action_test2.csv', encoding='utf-8')
# """



############# 3.action feature_3   #############
"""
# 1. 全部浏览记录中0-9出现的次数
# 2. 对应浏览记录中0-9出现的次数
# 3. 全部浏览记录浏览时间
# 4. 对应浏览记录浏览时间
# 5. 对应浏览记录是否出现5 6

# """

# 全部浏览记录中0-9出现的次数
def count_56789(orderFuture, action):
    action_1 = action[action['actionType'] == 1]
    action_2 = action[action['actionType'] == 2]
    action_3 = action[action['actionType'] == 3]
    action_4 = action[action['actionType'] == 4]
    action_5 = action[action['actionType'] == 5]
    action_6 = action[action['actionType'] == 6]
    action_7 = action[action['actionType'] == 7]
    action_8 = action[action['actionType'] == 8]
    action_9 = action[action['actionType'] == 9]
    action_1 = action_1.groupby(action_1.userid)['actionType'].count().reset_index()  # 每个用户1操作的总数
    action_2 = action_2.groupby(action_2.userid)['actionType'].count().reset_index()  # 每个用户2操作的总数
    action_3 = action_3.groupby(action_3.userid)['actionType'].count().reset_index()  # 每个用户3操作的总数
    action_4 = action_4.groupby(action_4.userid)['actionType'].count().reset_index()  # 每个用户4操作的总数
    action_5 = action_5.groupby(action_5.userid)['actionType'].count().reset_index()  # 每个用户5操作的总数
    action_6 = action_6.groupby(action_6.userid)['actionType'].count().reset_index()  # 每个用户6操作的总数
    action_7 = action_7.groupby(action_7.userid)['actionType'].count().reset_index()  # 每个用户7操作的总数
    action_8 = action_8.groupby(action_8.userid)['actionType'].count().reset_index()  # 每个用户8操作的总数
    action_9 = action_9.groupby(action_9.userid)['actionType'].count().reset_index()  # 每个用户9操作的总数
    action_all = action.groupby(action.userid)['actionType'].count().reset_index()    # 每个用户 操作的总数
    action_1.rename(columns={'actionType': 'action_1'}, inplace=True)
    action_2.rename(columns={'actionType': 'action_2'}, inplace=True)
    action_3.rename(columns={'actionType': 'action_3'}, inplace=True)
    action_4.rename(columns={'actionType': 'action_4'}, inplace=True)
    action_5.rename(columns={'actionType': 'action_5'}, inplace=True)
    action_6.rename(columns={'actionType': 'action_6'}, inplace=True)
    action_7.rename(columns={'actionType': 'action_7'}, inplace=True)
    action_8.rename(columns={'actionType': 'action_8'}, inplace=True)
    action_9.rename(columns={'actionType': 'action_9'}, inplace=True)
    action_all.rename(columns={'actionType': 'action_all'}, inplace=True)
    orderFuture = pd.merge(orderFuture, action_1, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_2, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_3, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_4, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_5, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_6, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_7, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_8, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_9, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_all, on='userid', how='left')
    orderFuture['action_1_rate'] = orderFuture['action_1']/orderFuture['action_all']   # 每个用户1操作的次数占总数的比
    orderFuture['action_2_rate'] = orderFuture['action_2']/orderFuture['action_all']   # 每个用户2操作的次数占总数的比
    orderFuture['action_3_rate'] = orderFuture['action_3']/orderFuture['action_all']   # 每个用户3操作的次数占总数的比
    orderFuture['action_4_rate'] = orderFuture['action_4']/orderFuture['action_all']   # 每个用户4操作的次数占总数的比
    orderFuture['action_5_rate'] = orderFuture['action_5']/orderFuture['action_all']   # 每个用户5操作的次数占总数的比
    orderFuture['action_6_rate'] = orderFuture['action_6']/orderFuture['action_all']   # 每个用户6操作的次数占总数的比
    orderFuture['action_7_rate'] = orderFuture['action_7']/orderFuture['action_all']   # 每个用户7操作的次数占总数的比
    orderFuture['action_8_rate'] = orderFuture['action_8']/orderFuture['action_all']   # 每个用户8操作的次数占总数的比
    orderFuture['action_9_rate'] = orderFuture['action_9']/orderFuture['action_all']   # 每个用户9操作的次数占总数的比
    # print(orderFuture)
    return orderFuture

orderFuture_train = count_56789(orderFuture_train, action_train)
orderFuture_test = count_56789(orderFuture_test, action_test)


# 对应浏览记录中0-9出现的次数
def count_1_9(orderFuture, action):
    action_1 = action[(action['actionType'] == 1) & (action.orderid.isnull())]
    action_2 = action[(action['actionType'] == 2) & (action.orderid.isnull())]
    action_3 = action[(action['actionType'] == 3) & (action.orderid.isnull())]
    action_4 = action[(action['actionType'] == 4) & (action.orderid.isnull())]
    action_5 = action[(action['actionType'] == 5) & (action.orderid.isnull())]
    action_6 = action[(action['actionType'] == 6) & (action.orderid.isnull())]
    action_7 = action[(action['actionType'] == 7) & (action.orderid.isnull())]
    action_8 = action[(action['actionType'] == 8) & (action.orderid.isnull())]
    action_9 = action[(action['actionType'] == 9) & (action.orderid.isnull())]
    action_all = action[action.orderid.isnull()]
    action_1 = action_1.groupby(action_1.userid)['actionType'].count().reset_index()  # 每个用户1操作的总数
    action_2 = action_2.groupby(action_2.userid)['actionType'].count().reset_index()  # 每个用户2操作的总数
    action_3 = action_3.groupby(action_3.userid)['actionType'].count().reset_index()  # 每个用户3操作的总数
    action_4 = action_4.groupby(action_4.userid)['actionType'].count().reset_index()  # 每个用户4操作的总数
    action_5 = action_5.groupby(action_5.userid)['actionType'].count().reset_index()  # 每个用户5操作的总数
    action_6 = action_6.groupby(action_6.userid)['actionType'].count().reset_index()  # 每个用户6操作的总数
    action_7 = action_7.groupby(action_7.userid)['actionType'].count().reset_index()  # 每个用户7操作的总数
    action_8 = action_8.groupby(action_8.userid)['actionType'].count().reset_index()  # 每个用户8操作的总数
    action_9 = action_9.groupby(action_9.userid)['actionType'].count().reset_index()  # 每个用户9操作的总数
    action_all = action_all.groupby(action_all.userid)['actionType'].count().reset_index()    # 每个用户 操作的总数
    action_1.rename(columns={'actionType': 'action_1_c'}, inplace=True)
    action_2.rename(columns={'actionType': 'action_2_c'}, inplace=True)
    action_3.rename(columns={'actionType': 'action_3_c'}, inplace=True)
    action_4.rename(columns={'actionType': 'action_4_c'}, inplace=True)
    action_5.rename(columns={'actionType': 'action_5_c'}, inplace=True)
    action_6.rename(columns={'actionType': 'action_6_c'}, inplace=True)
    action_7.rename(columns={'actionType': 'action_7_c'}, inplace=True)
    action_8.rename(columns={'actionType': 'action_8_c'}, inplace=True)
    action_9.rename(columns={'actionType': 'action_9_c'}, inplace=True)
    action_all.rename(columns={'actionType': 'action_all_c'}, inplace=True)
    orderFuture = pd.merge(orderFuture, action_1, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_2, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_3, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_4, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_5, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_6, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_7, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_8, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_9, on='userid', how='left')
    orderFuture = pd.merge(orderFuture, action_all, on='userid', how='left')
    orderFuture['action_1_rate_c'] = orderFuture['action_1_c']/orderFuture['action_all_c']   # 每个用户1操作的次数占总数的比
    orderFuture['action_2_rate_c'] = orderFuture['action_2_c']/orderFuture['action_all_c']   # 每个用户2操作的次数占总数的比
    orderFuture['action_3_rate_c'] = orderFuture['action_3_c']/orderFuture['action_all_c']   # 每个用户3操作的次数占总数的比
    orderFuture['action_4_rate_c'] = orderFuture['action_4_c']/orderFuture['action_all_c']   # 每个用户4操作的次数占总数的比
    orderFuture['action_5_rate_c'] = orderFuture['action_5_c']/orderFuture['action_all_c']   # 每个用户5操作的次数占总数的比
    orderFuture['action_6_rate_c'] = orderFuture['action_6_c']/orderFuture['action_all_c']   # 每个用户6操作的次数占总数的比
    orderFuture['action_7_rate_c'] = orderFuture['action_7_c']/orderFuture['action_all_c']   # 每个用户7操作的次数占总数的比
    orderFuture['action_8_rate_c'] = orderFuture['action_8_c']/orderFuture['action_all_c']   # 每个用户8操作的次数占总数的比
    orderFuture['action_9_rate_c'] = orderFuture['action_9_c']/orderFuture['action_all_c']   # 每个用户9操作的次数占总数的比
    # print(orderFuture)
    return orderFuture

orderFuture_train = count_1_9(orderFuture_train, action_train)
orderFuture_test = count_1_9(orderFuture_test, action_test)


# 全部浏览记录浏览时间
def action_time(orderFuture, action):
    first_action = action[['userid', 'actionType', 'actionTime']].groupby(['userid']).first().reset_index()
    last_action = action[['userid', 'actionType', 'actionTime']].groupby(['userid']).last().reset_index()

    first_action['action_time'] = last_action['actionTime'] - first_action['actionTime']
    orderFuture = pd.merge(orderFuture, first_action[['userid', 'action_time']], on='userid', how='left')
    return orderFuture

orderFuture_train = action_time(orderFuture_train, action_train)
orderFuture_test = action_time(orderFuture_test, action_test)


# 对应浏览记录浏览时间
def action_time_c(orderFuture, action):
    action = action[action.orderid.isnull()]
    first_action = action[['userid', 'actionType', 'actionTime']].groupby(['userid']).first().reset_index()
    last_action = action[['userid', 'actionType', 'actionTime']].groupby(['userid']).last().reset_index()

    first_action['action_time_c'] = last_action['actionTime'] - first_action['actionTime']
    orderFuture = pd.merge(orderFuture, first_action[['userid', 'action_time_c']], on='userid', how='left')
    return orderFuture

orderFuture_train = action_time_c(orderFuture_train, action_train)
orderFuture_test = action_time_c(orderFuture_test, action_test)


# 全部浏览记录是否出现56 67 78 89
def appear_56(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'action_56_count', 'action_67_count', 'action_78_count', 'action_89_count'])
    userid = []
    action_56_count = []
    action_67_count = []
    action_78_count = []
    action_89_count = []
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count56 = 0
        count67 = 0
        count78 = 0
        count89 = 0
        for i in range(len(action1)):
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6)
                and (action1['actionType_time'][i] < 1800)):
                count56 = count56 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7)
                and (action1['actionType_time'][i] < 1800)):
                count67 = count67 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8)
                and (action1['actionType_time'][i] < 1800)):
                count78 = count78 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 8) and (action1['actionType'][i + 1] == 9)
                and (action1['actionType_time'][i] < 1800)):
                count89 = count89 + 1
        userid.append(row.userid)
        action_56_count.append(count56)
        action_67_count.append(count67)
        action_78_count.append(count78)
        action_89_count.append(count89)
    count['userid'] = userid
    count['action_56_count'] = action_56_count
    count['action_67_count'] = action_67_count
    count['action_78_count'] = action_78_count
    count['action_89_count'] = action_89_count
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_56_count', 'action_67_count', 'action_78_count', 'action_89_count']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_56(orderFuture_train, action_train)
orderFuture_test = appear_56(orderFuture_test, action_test)


# 对应浏览记录是否出现56 67 78 89
def appear_56_c(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'action_56_count_c', 'action_67_count_c', 'action_78_count_c', 'action_89_count_c'])
    userid = []
    action_56_count_c = []
    action_67_count_c = []
    action_78_count_c = []
    action_89_count_c = []
    action = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count56 = 0
        count67 = 0
        count78 = 0
        count89 = 0
        for i in range(len(action1)):
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6)
                and (action1['actionType_time'][i] < 1800)):
                count56 = count56 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7)
                and (action1['actionType_time'][i] < 1800)):
                count67 = count67 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8)
                and (action1['actionType_time'][i] < 1800)):
                count78 = count78 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 8) and (action1['actionType'][i + 1] == 9)
                and (action1['actionType_time'][i] < 1800)):
                count89 = count89 + 1
        userid.append(row.userid)
        action_56_count_c.append(count56)
        action_67_count_c.append(count67)
        action_78_count_c.append(count78)
        action_89_count_c.append(count89)
    count['userid'] = userid
    count['action_56_count_c'] = action_56_count_c
    count['action_67_count_c'] = action_67_count_c
    count['action_78_count_c'] = action_78_count_c
    count['action_89_count_c'] = action_89_count_c
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_56_count_c', 'action_67_count_c', 'action_78_count_c', 'action_89_count_c']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_56_c(orderFuture_train, action_train)
orderFuture_test = appear_56_c(orderFuture_test, action_test)


# 全部浏览记录是否出现567 678 789 566
def appear_567(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'action_567_count', 'action_678_count', 'action_789_count', 'action_566_count'])
    userid = []
    action_567_count = []
    action_678_count = []
    action_789_count = []
    action_566_count = []
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count567 = 0
        count678 = 0
        count789 = 0
        count566 = 0
        for i in range(len(action1)):
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800)):
                count567 = count567 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7) and (action1['actionType'][i + 2] == 8)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800)):
                count678 = count678 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8) and (action1['actionType'][i + 2] == 9)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800)):
                count789 = count789 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 6)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800)):
                count566 = count566 + 1
        userid.append(row.userid)
        action_567_count.append(count567)
        action_678_count.append(count678)
        action_789_count.append(count789)
        action_566_count.append(count566)
    count['userid'] = userid
    count['action_567_count'] = action_567_count
    count['action_678_count'] = action_678_count
    count['action_789_count'] = action_789_count
    count['action_566_count'] = action_566_count
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_567_count', 'action_678_count', 'action_789_count', 'action_566_count']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_567(orderFuture_train, action_train)
orderFuture_test = appear_567(orderFuture_test, action_test)


# 对应浏览记录是否出现567 678 789 566
def appear_567_c(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'action_567_count_c', 'action_678_count_c', 'action_789_count_c', 'action_566_count_c'])
    userid = []
    action_567_count_c = []
    action_678_count_c = []
    action_789_count_c = []
    action_566_count_c = []
    action = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count567 = 0
        count678 = 0
        count789 = 0
        count566 = 0
        for i in range(len(action1)):
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800)):
                count567 = count567 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7) and (action1['actionType'][i + 2] == 8)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800)):
                count678 = count678 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8) and (action1['actionType'][i + 2] == 9)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800)):
                count789 = count789 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 6)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800)):
                count566 = count566 + 1
        userid.append(row.userid)
        action_567_count_c.append(count567)
        action_678_count_c.append(count678)
        action_789_count_c.append(count789)
        action_566_count_c.append(count566)
    count['userid'] = userid
    count['action_567_count_c'] = action_567_count_c
    count['action_678_count_c'] = action_678_count_c
    count['action_789_count_c'] = action_789_count_c
    count['action_566_count_c'] = action_566_count_c
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_567_count_c', 'action_678_count_c', 'action_789_count_c', 'action_566_count_c']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_567_c(orderFuture_train, action_train)
orderFuture_test = appear_567_c(orderFuture_test, action_test)


# 全部浏览记录是否出现5678 6789
def appear_5678(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'action_5678_count', 'action_6789_count'])
    userid = []
    action_5678_count = []
    action_6789_count = []
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count5678 = 0
        count6789 = 0
        for i in range(len(action1)):
            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7) and (action1['actionType'][i + 3] == 8)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800) and (action1['actionType_time'][i + 2] < 1800)):
                count5678 = count5678 + 1
            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7) and (action1['actionType'][i + 2] == 8) and (action1['actionType'][i + 3] == 9)

                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800) and (action1['actionType_time'][i + 2] < 1800)):
                count6789 = count6789 + 1
        userid.append(row.userid)
        action_5678_count.append(count5678)
        action_6789_count.append(count6789)
    count['userid'] = userid
    count['action_5678_count'] = action_5678_count
    count['action_6789_count'] = action_6789_count
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_5678_count', 'action_6789_count']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_5678(orderFuture_train, action_train)
orderFuture_test = appear_5678(orderFuture_test, action_test)


# 对应浏览记录是否出现5678 6789
def appear_5678_c(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'action_5678_count_c', 'action_6789_count_c'])
    userid = []
    action_5678_count_c = []
    action_6789_count_c = []
    action = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count5678 = 0
        count6789 = 0
        for i in range(len(action1)):
            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7) and (action1['actionType'][i + 3] == 8)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800) and (action1['actionType_time'][i + 2] < 1800)):
                count5678 = count5678 + 1
            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7) and (action1['actionType'][i + 2] == 8) and (action1['actionType'][i + 3] == 9)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800) and (action1['actionType_time'][i + 2] < 1800)):
                count6789 = count6789 + 1
        userid.append(row.userid)
        action_5678_count_c.append(count5678)
        action_6789_count_c.append(count6789)
    count['userid'] = userid
    count['action_5678_count_c'] = action_5678_count_c
    count['action_6789_count_c'] = action_6789_count_c
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_5678_count_c', 'action_6789_count_c']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_5678_c(orderFuture_train, action_train)
orderFuture_test = appear_5678_c(orderFuture_test, action_test)


# 全部浏览记录是否出现56789
def appear_56789(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'action_56789_count'])
    userid = []
    action_56789_count = []
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count56789 = 0
        for i in range(len(action1)):
            if (((i + 4) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7) and (action1['actionType'][i + 3] == 8) and (action1['actionType'][i + 4] == 9)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800) and (action1['actionType_time'][i + 2] < 1800) and (action1['actionType_time'][i + 3] < 1800)):
                count56789 = count56789 + 1
        userid.append(row.userid)
        action_56789_count.append(count56789)
    count['userid'] = userid
    count['action_56789_count'] = action_56789_count
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_56789_count']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_56789(orderFuture_train, action_train)
orderFuture_test = appear_56789(orderFuture_test, action_test)


# 对应浏览记录是否出现56789
def appear_56789_c(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'action_56789_count_c'])
    userid = []
    action_56789_count_c = []
    action = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count56789 = 0
        for i in range(len(action1)):
            if (((i + 4) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7) and (action1['actionType'][i + 3] == 8) and (action1['actionType'][i + 4] == 9)
                and (action1['actionType_time'][i] < 1800) and (action1['actionType_time'][i + 1] < 1800) and ( action1['actionType_time'][i + 2] < 1800) and (action1['actionType_time'][i + 3] < 1800)):
                count56789 = count56789 + 1
        userid.append(row.userid)
        action_56789_count_c.append(count56789)
    count['userid'] = userid
    count['action_56789_count_c'] = action_56789_count_c
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_56789_count_c']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_56789_c(orderFuture_train, action_train)
orderFuture_test = appear_56789_c(orderFuture_test, action_test)


############# 3.action feature_5   #############
"""
# 1. action中大于6出现的次数
# 2. 对应点击2-4的和值 与 5-9 的比值
# 3. 全部点击2-4的和值 与 5-9 的比值
# 4. 对应浏览记录 1-9 操作所用平均时间
# 5. 全部浏览记录 1-9 操作所用平均时间

# """
# action中大于6出现的次数
def greater_6_c(orderFuture):
    action_7_c = orderFuture['action_7_c'].fillna(0)
    action_8_c = orderFuture['action_8_c'].fillna(0)
    action_9_c = orderFuture['action_9_c'].fillna(0)
    orderFuture['action_greater_7_c'] = action_7_c + action_8_c + action_9_c
    return orderFuture

orderFuture_train = greater_6_c(orderFuture_train)
orderFuture_test = greater_6_c(orderFuture_test)

# 对应点击2-4的和值 与 5-9 的比值
def rate_24_59_c(orderFuture):
    action = orderFuture.fillna(0)

    orderFuture['rate_1_59_c'] = (action['action_1_c'])/(action['action_5_c'] + action['action_6_c'] + action['action_7_c'] + action['action_8_c'] + action['action_9_c'])
    orderFuture['rate_24_59_c'] = (action['action_2_c'] + action['action_3_c'] + action['action_4_c'])/(action['action_5_c'] + action['action_6_c'] + action['action_7_c'] + action['action_8_c'] + action['action_9_c'])
    # orderFuture['rate_time_1_59_c'] = (action['time_1_c'])/(action['time_5_c'] + action['time_6_c'] + action['time_7_c'] + action['time_8_c'] + action['time_9_c'])
    return orderFuture

orderFuture_train = rate_24_59_c(orderFuture_train)
orderFuture_test = rate_24_59_c(orderFuture_test)

# 全部点击2-4的和值 与 5-9 的比值
def rate_24_59(orderFuture):
    action = orderFuture.fillna(0)

    orderFuture['rate_1_59'] = (action['action_1'])/(action['action_5'] + action['action_6'] + action['action_7'] + action['action_8'] + action['action_9'])
    orderFuture['rate_24_59'] = (action['action_2'] + action['action_3'] + action['action_4'])/(action['action_5'] + action['action_6'] + action['action_7'] + action['action_8'] + action['action_9'])
    # orderFuture['rate_time_1_59'] = (action['time_1'])/(action['time_5'] + action['time_6'] + action['time_7'] + action['time_8'] + action['time_9'])
    return orderFuture

orderFuture_train = rate_24_59(orderFuture_train)
orderFuture_test = rate_24_59(orderFuture_test)


# 全部action 最后一次 的类型
def latest_actionType(orderFuture, action):
    latest = action.groupby(['userid']).last().reset_index()
    latest.rename(columns={'actionType': 'latest_actionType'}, inplace=True)
    orderFuture = pd.merge(orderFuture, latest[['userid', 'latest_actionType']], on='userid', how='left')
    return orderFuture

orderFuture_train = latest_actionType(orderFuture_train, action_train)
orderFuture_test = latest_actionType(orderFuture_test, action_test)


# 全部 action 倒数第2-6次操作的类型
def latest2_actionType(orderFuture, action):
    userid = []
    latest_2_actionType = []
    latest_3_actionType = []
    latest_4_actionType = []
    latest_5_actionType = []
    latest_6_actionType = []
    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest_2 = latest
    for index, row in latest.iterrows():
        userid.append(row.userid)
        if(row.userid == action['userid'][row.actionTime-1]):
            latest_2_actionType.append(action['actionType'][row.actionTime-1])
        else:
            latest_2_actionType.append(None)

        if (row.userid == action['userid'][row.actionTime - 2]):
            latest_3_actionType.append(action['actionType'][row.actionTime - 2])
        else:
            latest_3_actionType.append(None)

        if (row.userid == action['userid'][row.actionTime - 3]):
            latest_4_actionType.append(action['actionType'][row.actionTime - 3])
        else:
            latest_4_actionType.append(None)

        if (row.userid == action['userid'][row.actionTime - 4]):
            latest_5_actionType.append(action['actionType'][row.actionTime - 4])
        else:
            latest_5_actionType.append(None)

        if (row.userid == action['userid'][row.actionTime - 5]):
            latest_6_actionType.append(action['actionType'][row.actionTime - 5])
        else:
            latest_6_actionType.append(None)

    latest_2['latest_2_actionType'] = latest_2_actionType
    latest_2['latest_3_actionType'] = latest_3_actionType
    latest_2['latest_4_actionType'] = latest_4_actionType
    latest_2['latest_5_actionType'] = latest_5_actionType
    latest_2['latest_6_actionType'] = latest_6_actionType
    orderFuture = pd.merge(orderFuture, latest_2[['userid', 'latest_2_actionType', 'latest_3_actionType',
    'latest_4_actionType', 'latest_5_actionType', 'latest_6_actionType']], on='userid', how='left')
    return orderFuture

orderFuture_train = latest2_actionType(orderFuture_train, action_train)
orderFuture_test = latest2_actionType(orderFuture_test, action_test)


# 时间间隔
# 最后1 2 3 4 次操作的时间间隔
# 时间间隔的均值 最小值 最大值 方差
def time_interval(orderFuture, action):
    # 1 2 3 4 5 6
    userid = []
    latest_1_time_interval = []
    latest_2_time_interval = []
    latest_3_time_interval = []
    latest_4_time_interval = []
    latest_5_time_interval = []
    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    latest_2 = latest
    for index, row in latest.iterrows():
        userid.append(row.userid)
        # 1
        latest_1_time_interval.append(action['actionType_time'][row.max_index - 1])

        # 2
        if (row.userid == action['userid'][row.max_index - 2]):
            latest_2_time_interval.append(action['actionType_time'][row.max_index - 2])
        else:
            latest_2_time_interval.append(None)
        # 3
        if (row.userid == action['userid'][row.max_index - 3]):
            latest_3_time_interval.append(action['actionType_time'][row.max_index - 3])
        else:
            latest_3_time_interval.append(None)
        # 4
        if (row.userid == action['userid'][row.max_index - 4]):
            latest_4_time_interval.append(action['actionType_time'][row.max_index - 4])
        else:
            latest_4_time_interval.append(None)
        # 5
        if (row.userid == action['userid'][row.max_index - 5]):
            latest_5_time_interval.append(action['actionType_time'][row.max_index - 5])
        else:
            latest_5_time_interval.append(None)

    latest_2['latest_1_time_interval'] = latest_1_time_interval
    latest_2['latest_2_time_interval'] = latest_2_time_interval
    latest_2['latest_3_time_interval'] = latest_3_time_interval
    latest_2['latest_4_time_interval'] = latest_4_time_interval
    latest_2['latest_5_time_interval'] = latest_5_time_interval
    orderFuture = pd.merge(orderFuture, latest_2[['userid', 'latest_1_time_interval', 'latest_2_time_interval', 'latest_3_time_interval',
                                        'latest_4_time_interval', 'latest_5_time_interval']], on='userid', how='left')

    # 均值
    latest = action.groupby(['userid'])['actionType_time'].mean().reset_index()
    latest.rename(columns={'actionType_time': 'actionType_time_mean'}, inplace=True)
    orderFuture = pd.merge(orderFuture, latest[['userid', 'actionType_time_mean']], on='userid', how='left')
    # 方差
    latest = action.groupby(['userid'])['actionType_time'].agg({'actionType_time_var': 'var'}).reset_index()
    orderFuture = pd.merge(orderFuture, latest[['userid', 'actionType_time_var']], on='userid', how='left')

    # 最小值
    latest = action.groupby(['userid'])['actionType_time'].min().reset_index()
    latest.rename(columns={'actionType_time': 'actionType_time_min'}, inplace=True)
    orderFuture = pd.merge(orderFuture, latest[['userid', 'actionType_time_min']], on='userid', how='left')

    return orderFuture

orderFuture_train = time_interval(orderFuture_train, action_train)
orderFuture_test = time_interval(orderFuture_test, action_test)



# action 最后2 3 4 5 6 次操作时间的方差 和 均值
def var_actionTime(orderFuture, action):
    userid = []
    latest_3_actionTime_var = []
    latest_4_actionTime_var = []
    latest_5_actionTime_var = []
    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest_2 = latest
    for index, row in latest.iterrows():
        userid.append(row.userid)
        # 2
        if ((row.userid == action['userid'][row.actionTime]) and (row.userid == action['userid'][row.actionTime - 1]) and
            (row.userid == action['userid'][row.actionTime - 2])):

            var = pd.Series([action['actionTime'][row.actionTime]-action['actionTime'][row.actionTime-1],
                             action['actionTime'][row.actionTime-1]-action['actionTime'][row.actionTime-2]
                             ]).var()
            latest_3_actionTime_var.append(var)
        else:
            latest_3_actionTime_var.append(None)
        # 3
        if ((row.userid == action['userid'][row.actionTime]) and (row.userid == action['userid'][row.actionTime - 1]) and
            (row.userid == action['userid'][row.actionTime - 2]) and (row.userid == action['userid'][row.actionTime - 3])):

            var = pd.Series([action['actionTime'][row.actionTime]-action['actionTime'][row.actionTime-1],
                   action['actionTime'][row.actionTime-1]-action['actionTime'][row.actionTime-2],
                   action['actionTime'][row.actionTime-2]-action['actionTime'][row.actionTime-3]
                   ]).var()

            latest_4_actionTime_var.append(var)
        else:
            latest_4_actionTime_var.append(None)
        # 4
        if ((row.userid == action['userid'][row.actionTime]) and (row.userid == action['userid'][row.actionTime - 1]) and
            (row.userid == action['userid'][row.actionTime - 2]) and (row.userid == action['userid'][row.actionTime - 3]) and
            (row.userid == action['userid'][row.actionTime - 4])):

            var = pd.Series([action['actionTime'][row.actionTime]-action['actionTime'][row.actionTime-1],
                   action['actionTime'][row.actionTime-1]-action['actionTime'][row.actionTime-2],
                   action['actionTime'][row.actionTime-2]-action['actionTime'][row.actionTime-3],
                   action['actionTime'][row.actionTime-3]-action['actionTime'][row.actionTime-4]
                   ]).var()

            latest_5_actionTime_var.append(var)
        else:
            latest_5_actionTime_var.append(None)

    latest_2['latest_3_actionTime_var'] = latest_3_actionTime_var
    latest_2['latest_4_actionTime_var'] = latest_4_actionTime_var
    latest_2['latest_5_actionTime_var'] = latest_5_actionTime_var

    orderFuture = pd.merge(orderFuture, latest_2[['userid', 'latest_3_actionTime_var', 'latest_4_actionTime_var',
                                'latest_5_actionTime_var']], on='userid', how='left')
    return orderFuture

orderFuture_train = var_actionTime(orderFuture_train, action_train)
orderFuture_test = var_actionTime(orderFuture_test, action_test)


# 对应浏览记录浏览平均时间(可以改成最近几天的)
def sum_actionType_time(orderFuture, action):
    action = action[action.orderid.isnull()]
    action1 = action.groupby(['userid'])['actionType_time'].sum().reset_index()
    action1.rename(columns={'actionType_time': 'actionType_time_sum'}, inplace=True)

    action2 = action.groupby(['userid', 'action_date']).count().reset_index()
    action3 = action2.groupby(['userid'])['action_date'].count().reset_index()
    action3.rename(columns={'action_date': 'days_action'}, inplace=True)

    action3 = pd.merge(action1, action3, on='userid', how='left')
    print(action3)
    action3['actionType_time_day_avg'] = action3['actionType_time_sum']/action3['days_action']
    orderFuture = pd.merge(orderFuture, action3[['userid', 'actionType_time_day_avg']], on='userid', how='left')
    return orderFuture

orderFuture_train = sum_actionType_time(orderFuture_train, action_train)
orderFuture_test = sum_actionType_time(orderFuture_test, action_test)


# 对应浏览记录 1-9 操作所用平均时间
def avg_time_action_c(orderFuture, action, k):
    time_k = []
    select = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        print(index)
        action_k = select[(select['actionType'] == k) & (select['userid'] == row.userid)]
        if (len(action_k) == 0):
            time = None
        else:
            time = 0
            for index1, row1 in action_k.iterrows():
                if(((index1 + 1) < len(action)) and (row1.userid == action['userid'][index1+1])):
                    time = time + (action['actionTime'][index1+1] - row1.actionTime)
        time_k.append(time)
    orderFuture['time_'+ str(k) +'_c'] = time_k
    orderFuture['time_'+ str(k) +'_c'] = orderFuture['time_'+ str(k) +'_c']/orderFuture['action_'+ str(k) +'_c']
    return orderFuture

# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 1)
# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 2)
# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 3)
# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 4)
# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 5)
# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 6)
# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 7)
# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 8)
# orderFuture_test = avg_time_action_c(orderFuture_test, action_test, 9)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 1)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 2)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 3)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 4)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 5)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 6)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 7)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 8)
# orderFuture_train = avg_time_action_c(orderFuture_train, action_train, 9)

# 全部浏览记录 1-9 操作所用平均时间
def avg_time_action(orderFuture, action, k):
    time_k = []
    for index, row in orderFuture.iterrows():
        print(index)
        action_k = action[(action['actionType'] == k) & (action['userid'] == row.userid)]
        if (len(action_k) == 0):
            time = None
        else:
            time = 0
            for index1, row1 in action_k.iterrows():
                if(((index1 + 1) < len(action)) and (row1.userid == action['userid'][index1+1])):
                    time = time + (action['actionTime'][index1+1] - row1.actionTime)
        time_k.append(time)
    orderFuture['time_'+ str(k)] = time_k
    orderFuture['time_'+ str(k)] = orderFuture['time_'+ str(k)]/orderFuture['action_'+ str(k)]
    return orderFuture

# orderFuture_test = avg_time_action(orderFuture_test, action_test, 1)
# orderFuture_test = avg_time_action(orderFuture_test, action_test, 2)
# orderFuture_test = avg_time_action(orderFuture_test, action_test, 3)
# orderFuture_test = avg_time_action(orderFuture_test, action_test, 4)
# orderFuture_test = avg_time_action(orderFuture_test, action_test, 5)
# orderFuture_test = avg_time_action(orderFuture_test, action_test, 6)
# orderFuture_test = avg_time_action(orderFuture_test, action_test, 7)
# orderFuture_test = avg_time_action(orderFuture_test, action_test, 8)
# orderFuture_test = avg_time_action(orderFuture_test, action_test, 9)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 1)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 2)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 3)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 4)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 5)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 6)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 7)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 8)
# orderFuture_train = avg_time_action(orderFuture_train, action_train, 9)



############# 3.action feature_7  #############
"""
# 1. 
# 2. 
# 3. 
# 4. 
# 5. 

# """


# 最近7天的使用时间 eval-auc:0.963724
def latest_7day_count(orderFuture, action):
    userid = []
    latest_7day_actionType_time = []

    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    latest_2 = latest
    for index, row in latest.iterrows():
        userid.append(row.userid)
        # 7天
        action1 = action[
            (action['userid'] == row.userid) & (action['actionTime'] > action['actionTime'][row.max_index] - 604800)]
        latest_7day_actionType_time.append(np.sum(action1.actionType_time))

    latest_2['userid'] = userid
    latest_2['latest_7day_actionType_time'] = latest_7day_actionType_time
    print(latest_2)
    orderFuture = pd.merge(orderFuture, latest_2[['userid', 'latest_7day_actionType_time']], on='userid', how='left')
    return orderFuture

# orderFuture_train = latest_7day_count(orderFuture_train, action_train)
# orderFuture_test = latest_7day_count(orderFuture_test, action_test)


# 最近1天的操作1 2 4 5 6 7的次数
def latest_1day_actionType_count(orderFuture, action):
    userid = []
    latest_1day_actionType1_count = []
    latest_1day_actionType2_count = []
    latest_1day_actionType3_count = []
    latest_1day_actionType4_count = []
    latest_1day_actionType5_count = []
    latest_1day_actionType6_count = []
    latest_1day_actionType7_count = []
    latest_1day_actionType8_count = []
    latest_1day_actionType9_count = []

    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    latest_2 = latest
    for index, row in latest.iterrows():
        print(index)
        userid.append(row.userid)
        # 加速，先大选userid，再小选日期
        action0 = action[action['userid'] == row.userid]
        action_1 = action0[action0['action_date'] == action0['action_date'][row.max_index]]
        action1 = action_1[action_1['actionType'] == 1]
        action2 = action_1[action_1['actionType'] == 2]
        action3 = action_1[action_1['actionType'] == 3]
        action4 = action_1[action_1['actionType'] == 4]
        action5 = action_1[action_1['actionType'] == 5]
        action6 = action_1[action_1['actionType'] == 6]
        action7 = action_1[action_1['actionType'] == 7]
        action8 = action_1[action_1['actionType'] == 8]
        action9 = action_1[action_1['actionType'] == 9]
        latest_1day_actionType1_count.append(len(action1))
        latest_1day_actionType2_count.append(len(action2))
        latest_1day_actionType3_count.append(len(action3))
        latest_1day_actionType4_count.append(len(action4))
        latest_1day_actionType5_count.append(len(action5))
        latest_1day_actionType6_count.append(len(action6))
        latest_1day_actionType7_count.append(len(action7))
        latest_1day_actionType8_count.append(len(action8))
        latest_1day_actionType9_count.append(len(action9))
    latest_2['userid'] = userid
    latest_2['latest_1day_actionType1_count'] = latest_1day_actionType1_count
    latest_2['latest_1day_actionType2_count'] = latest_1day_actionType2_count
    latest_2['latest_1day_actionType3_count'] = latest_1day_actionType3_count
    latest_2['latest_1day_actionType4_count'] = latest_1day_actionType4_count
    latest_2['latest_1day_actionType5_count'] = latest_1day_actionType5_count
    latest_2['latest_1day_actionType6_count'] = latest_1day_actionType6_count
    latest_2['latest_1day_actionType7_count'] = latest_1day_actionType7_count
    latest_2['latest_1day_actionType8_count'] = latest_1day_actionType8_count
    latest_2['latest_1day_actionType9_count'] = latest_1day_actionType9_count
    print(latest_2)
    orderFuture = pd.merge(orderFuture, latest_2[['userid',
                       'latest_1day_actionType1_count', 'latest_1day_actionType2_count', 'latest_1day_actionType3_count',
                       'latest_1day_actionType4_count', 'latest_1day_actionType5_count', 'latest_1day_actionType6_count',
                       'latest_1day_actionType7_count', 'latest_1day_actionType8_count', 'latest_1day_actionType9_count'
                       ]], on='userid', how='left')
    return orderFuture

# orderFuture_train = latest_1day_actionType_count(orderFuture_train, action_train)
# orderFuture_test = latest_1day_actionType_count(orderFuture_test, action_test)


# 最近234567天的操作1 2 3 4 5 6 7 8 9的次数
def latest_2day_actionType_count(orderFuture, action, k):
    print(k)
    userid = []
    latest_2day_actionType1_count = []
    latest_2day_actionType2_count = []
    latest_2day_actionType3_count = []
    latest_2day_actionType4_count = []
    latest_2day_actionType5_count = []
    latest_2day_actionType6_count = []
    latest_2day_actionType7_count = []
    latest_2day_actionType8_count = []
    latest_2day_actionType9_count = []

    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    latest_2 = latest
    for index, row in latest.iterrows():
        userid.append(row.userid)
        # 加速，先大选userid，再小选日期
        action0 = action[action['userid'] == row.userid]
        action0['action_date'] = pd.to_datetime(action0['action_date'])
        action_2 = action0[action0['action_date'] >= (action0['action_date'][row.max_index] - datetime.timedelta(days=(k-1)))]

        action1 = action_2[action_2['actionType'] == 1]
        action2 = action_2[action_2['actionType'] == 2]
        action3 = action_2[action_2['actionType'] == 3]
        action4 = action_2[action_2['actionType'] == 4]
        action5 = action_2[action_2['actionType'] == 5]
        action6 = action_2[action_2['actionType'] == 6]
        action7 = action_2[action_2['actionType'] == 7]
        action8 = action_2[action_2['actionType'] == 8]
        action9 = action_2[action_2['actionType'] == 9]
        latest_2day_actionType1_count.append(len(action1))
        latest_2day_actionType2_count.append(len(action2))
        latest_2day_actionType3_count.append(len(action3))
        latest_2day_actionType4_count.append(len(action4))
        latest_2day_actionType5_count.append(len(action5))
        latest_2day_actionType6_count.append(len(action6))
        latest_2day_actionType7_count.append(len(action7))
        latest_2day_actionType8_count.append(len(action8))
        latest_2day_actionType9_count.append(len(action9))
    latest_2['userid'] = userid
    latest_2['latest_' + str(k) + 'day_actionType1_count'] = latest_2day_actionType1_count
    latest_2['latest_' + str(k) + 'day_actionType2_count'] = latest_2day_actionType2_count
    latest_2['latest_' + str(k) + 'day_actionType3_count'] = latest_2day_actionType3_count
    latest_2['latest_' + str(k) + 'day_actionType4_count'] = latest_2day_actionType4_count
    latest_2['latest_' + str(k) + 'day_actionType5_count'] = latest_2day_actionType5_count
    latest_2['latest_' + str(k) + 'day_actionType6_count'] = latest_2day_actionType6_count
    latest_2['latest_' + str(k) + 'day_actionType7_count'] = latest_2day_actionType7_count
    latest_2['latest_' + str(k) + 'day_actionType8_count'] = latest_2day_actionType8_count
    latest_2['latest_' + str(k) + 'day_actionType9_count'] = latest_2day_actionType9_count
    orderFuture = pd.merge(orderFuture, latest_2[['userid',
                       'latest_' + str(k) + 'day_actionType1_count', 'latest_' + str(k) + 'day_actionType2_count', 'latest_' + str(k) + 'day_actionType3_count',
                       'latest_' + str(k) + 'day_actionType4_count', 'latest_' + str(k) + 'day_actionType5_count', 'latest_' + str(k) + 'day_actionType6_count',
                       'latest_' + str(k) + 'day_actionType7_count', 'latest_' + str(k) + 'day_actionType8_count', 'latest_' + str(k) + 'day_actionType9_count'
                       ]], on='userid', how='left')
    return orderFuture

# orderFuture_train = latest_2day_actionType_count(orderFuture_train, action_train, 2)
# orderFuture_test = latest_2day_actionType_count(orderFuture_test, action_test, 2)
# orderFuture_train = latest_2day_actionType_count(orderFuture_train, action_train, 3)
# orderFuture_test = latest_2day_actionType_count(orderFuture_test, action_test, 3)
# orderFuture_train = latest_2day_actionType_count(orderFuture_train, action_train, 4)
# orderFuture_test = latest_2day_actionType_count(orderFuture_test, action_test, 4)
# orderFuture_train = latest_2day_actionType_count(orderFuture_train, action_train, 5)
# orderFuture_test = latest_2day_actionType_count(orderFuture_test, action_test, 5)
# orderFuture_train = latest_2day_actionType_count(orderFuture_train, action_train, 6)
# orderFuture_test = latest_2day_actionType_count(orderFuture_test, action_test, 6)
# orderFuture_train = latest_2day_actionType_count(orderFuture_train, action_train, 7)
# orderFuture_test = latest_2day_actionType_count(orderFuture_test, action_test, 7)


# 离最近的1-9的距离(间隔操作次数) 只取 56789
def min_distance_k(orderFuture, action):
    userid = []
    min_distance_1 = []
    min_distance_2 = []
    min_distance_3 = []
    min_distance_4 = []
    min_distance_5 = []
    min_distance_6 = []
    min_distance_7 = []
    min_distance_8 = []
    min_distance_9 = []

    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    latest_2 = latest
    for index, row in latest.iterrows():
        print(index)
        # print(row.userid)
        # print(last_max_index)
        # print(row.max_index)
        userid.append(row.userid)
        # 1
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 1):
                    min_distance_1.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 1):
                    min_distance_1.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_1.append(None)
                        break
            else:
                min_distance_1.append(None)
                break

        # 2
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 2):
                    min_distance_2.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 2):
                    min_distance_2.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_2.append(None)
                        break
            else:
                min_distance_2.append(None)
                break

        # 3
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 3):
                    min_distance_3.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 3):
                    min_distance_3.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_3.append(None)
                        break
            else:
                min_distance_3.append(None)
                break
        # 4
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 4):
                    min_distance_4.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 4):
                    min_distance_4.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_4.append(None)
                        break
            else:
                min_distance_4.append(None)
                break

        # 5
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 5):
                    min_distance_5.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 5):
                    min_distance_5.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_5.append(None)
                        break
            else:
                min_distance_5.append(None)
                break
        # 6
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 6):
                    min_distance_6.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 6):
                    min_distance_6.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_6.append(None)
                        break
            else:
                min_distance_6.append(None)
                break

        # 7
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 7):
                    min_distance_7.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 7):
                    min_distance_7.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_7.append(None)
                        break
            else:
                min_distance_7.append(None)
                break

        # 8
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 8):
                    min_distance_8.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 8):
                    min_distance_8.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_8.append(None)
                        break
            else:
                min_distance_8.append(None)
                break

        # 9
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 9):
                    min_distance_9.append(1)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 9):
                    min_distance_9.append(i + 2)
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_distance_9.append(None)
                        break
            else:
                min_distance_9.append(None)
                break

    latest_2['userid'] = userid
    latest_2['min_distance_1'] = min_distance_1
    latest_2['min_distance_2'] = min_distance_2
    latest_2['min_distance_3'] = min_distance_3
    latest_2['min_distance_4'] = min_distance_4
    latest_2['min_distance_5'] = min_distance_5
    latest_2['min_distance_6'] = min_distance_6
    latest_2['min_distance_7'] = min_distance_7
    latest_2['min_distance_8'] = min_distance_8
    latest_2['min_distance_9'] = min_distance_9
    print(latest_2)
    orderFuture = pd.merge(orderFuture,
                           latest_2[['userid', 'min_distance_1', 'min_distance_2', 'min_distance_3', 'min_distance_4',
                                     'min_distance_5', 'min_distance_6', 'min_distance_7', 'min_distance_8',
                                     'min_distance_9']], on='userid', how='left')
    return orderFuture

# orderFuture_test = min_distance_k(orderFuture_test, action_test)
# orderFuture_train = min_distance_k(orderFuture_train, action_train)


# 离最近的1-9的时间
def min_time_k(orderFuture, action):
    userid = []
    min_time_1 = []
    min_time_2 = []
    min_time_3 = []
    min_time_4 = []
    min_time_5 = []
    min_time_6 = []
    min_time_7 = []
    min_time_8 = []
    min_time_9 = []

    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    latest_2 = latest
    # print(latest)
    for index, row in latest.iterrows():
        print(index)
        # print(row.userid)
        # print(last_max_index)
        # print(row.max_index)
        userid.append(row.userid)
        # 1
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 1):
                    min_time_1.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 1):
                    min_time_1.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_1.append(None)
                        break
            else:
                min_time_1.append(None)
                break
        # 2
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 2):
                    min_time_2.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 2):
                    min_time_2.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_2.append(None)
                        break
            else:
                min_time_2.append(None)
                break
        # 3
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 3):
                    min_time_3.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 3):
                    min_time_3.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_3.append(None)
                        break
            else:
                min_time_3.append(None)
                break
        # 4
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 4):
                    min_time_4.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 4):
                    min_time_4.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_4.append(None)
                        break
            else:
                min_time_4.append(None)
                break
        # 5
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 5):
                    min_time_5.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 5):
                    min_time_5.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_5.append(None)
                        break
            else:
                min_time_5.append(None)
                break
        # 6
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 6):
                    min_time_6.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 6):
                    min_time_6.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_6.append(None)
                        break
            else:
                min_time_6.append(None)
                break
        # 7
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 7):
                    min_time_7.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 7):
                    min_time_7.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_7.append(None)
                        break
            else:
                min_time_7.append(None)
                break
        # 8
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 8):
                    min_time_8.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 8):
                    min_time_8.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_8.append(None)
                        break
            else:
                min_time_8.append(None)
                break

        # 9
        for i in range(row.max_index):
            if (row.userid == action['userid'][row.max_index - (i + 1)]):
                if (action['actionType'][row.max_index] == 9):
                    min_time_9.append(0)
                    break
                if (action['actionType'][row.max_index - (i + 1)] == 9):
                    min_time_9.append(action['actionTime'][row.max_index]-action['actionTime'][row.max_index - (i + 1)])
                    break
                else:
                    if ((i + 1) == row.max_index):
                        min_time_9.append(None)
                        break
            else:
                min_time_9.append(None)
                break
    latest_2['userid'] = userid
    latest_2['min_time_1'] = min_time_1
    latest_2['min_time_2'] = min_time_2
    latest_2['min_time_3'] = min_time_3
    latest_2['min_time_4'] = min_time_4
    latest_2['min_time_5'] = min_time_5
    latest_2['min_time_6'] = min_time_6
    latest_2['min_time_7'] = min_time_7
    latest_2['min_time_8'] = min_time_8
    latest_2['min_time_9'] = min_time_9
    print(latest_2)
    orderFuture = pd.merge(orderFuture, latest_2[['userid'
                         , 'min_time_1', 'min_time_2', 'min_time_3'
                         , 'min_time_4', 'min_time_5', 'min_time_6'
                         , 'min_time_7', 'min_time_8', 'min_time_9'
                                     ]], on='userid', how='left')
    return orderFuture

orderFuture_train = min_time_k(orderFuture_train, action_train)
orderFuture_test = min_time_k(orderFuture_test, action_test)


# 56时间间隔总时间
def time_56_all(orderFuture, action):
    count = pd.DataFrame(
        columns=['userid', 'time_56_all', 'time_67_all', 'time_78_all', 'time_89_all'])
    userid = []
    time_56_all = []
    time_67_all = []
    time_78_all = []
    time_89_all = []
    for index, row in orderFuture.iterrows():
        print(index)
        action1 = action[action['userid'] == row.userid].reset_index()
        count56 = 0
        count67 = 0
        count78 = 0
        count89 = 0
        for i in range(len(action1)):
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6)):
                count56 = count56 + (action1['actionTime'][i+1] - action1['actionTime'][i])
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7)):
                count67 = count67 + (action1['actionTime'][i+1] - action1['actionTime'][i])
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8)):
                count78 = count78 + (action1['actionTime'][i+1] - action1['actionTime'][i])
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 8) and (action1['actionType'][i + 1] == 9)):
                count89 = count89 + (action1['actionTime'][i+1] - action1['actionTime'][i])
        userid.append(row.userid)
        time_56_all.append(count56)
        time_67_all.append(count67)
        time_78_all.append(count78)
        time_89_all.append(count89)

    count['userid'] = userid
    count['time_56_all'] = time_56_all
    count['time_67_all'] = time_67_all
    count['time_78_all'] = time_78_all
    count['time_89_all'] = time_89_all
    count['time_56_all'][count['time_56_all'] == 0] = None
    count['time_67_all'][count['time_67_all'] == 0] = None
    count['time_78_all'][count['time_78_all'] == 0] = None
    count['time_89_all'][count['time_89_all'] == 0] = None
    orderFuture = pd.merge(orderFuture, count[['userid', 'time_56_all', 'time_67_all', 'time_78_all', 'time_89_all']],
                           on='userid', how='left')
    return orderFuture

orderFuture_train = time_56_all(orderFuture_train, action_train)
orderFuture_test = time_56_all(orderFuture_test, action_test)


# 56时间间隔和的对应时间
def time_56_c(orderFuture, action):
    count = pd.DataFrame(
        columns=['userid', 'time_56_c', 'time_67_', 'time_78_', 'time_89_c'])
    userid = []
    time_56_all = []
    time_67_all = []
    time_78_all = []
    time_89_all = []
    action = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        print(index)
        action1 = action[action['userid'] == row.userid].reset_index()
        count56 = 0
        count67 = 0
        count78 = 0
        count89 = 0
        for i in range(len(action1)):
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6)):
                count56 = count56 + (action1['actionTime'][i+1] - action1['actionTime'][i])
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7)):
                count67 = count67 + (action1['actionTime'][i+1] - action1['actionTime'][i])
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8)):
                count78 = count78 + (action1['actionTime'][i+1] - action1['actionTime'][i])
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 8) and (action1['actionType'][i + 1] == 9)):
                count89 = count89 + (action1['actionTime'][i+1] - action1['actionTime'][i])
        userid.append(row.userid)
        time_56_all.append(count56)
        time_67_all.append(count67)
        time_78_all.append(count78)
        time_89_all.append(count89)

    count['userid'] = userid
    count['time_56_c'] = time_56_all
    count['time_67_c'] = time_67_all
    count['time_78_c'] = time_78_all
    count['time_89_c'] = time_89_all
    count['time_56_c'][count['time_56_c'] == 0] = None
    count['time_67_c'][count['time_67_c'] == 0] = None
    count['time_78_c'][count['time_78_c'] == 0] = None
    count['time_89_c'][count['time_89_c'] == 0] = None
    orderFuture = pd.merge(orderFuture, count[['userid', 'time_56_c', 'time_67_c', 'time_78_c', 'time_89_c']],
                           on='userid', how='left')
    return orderFuture

orderFuture_train = time_56_c(orderFuture_train, action_train)
orderFuture_test = time_56_c(orderFuture_test, action_test)


# 第一个actionTime
def first_actionTime(orderFuture, action):
    first = action.groupby(['userid']).first().reset_index()
    first.rename(columns={'actionTime': 'first_actionTime'}, inplace=True)
    orderFuture = pd.merge(orderFuture, first[['userid', 'first_actionTime']], on='userid', how='left')
    return orderFuture
orderFuture_train = first_actionTime(orderFuture_train, action_train)
orderFuture_test = first_actionTime(orderFuture_test, action_test)




############# 3.action feature_8   #############
"""
# 1. 总体操作 1 2 3 4 5 6 7 8 9 次数的排名 rank
# 2. 对应操作 1 2 3 4 5 6 7 8 9 次数的排名 rank
# 3. 用户使用APP的天数，分别是否老用户
# 4. 56不连续的次数
# 5. 连续55操作次数
# 6. train中action操作次数对应的1的占比
# 7. 只有一次操作的用户最后一次操作的类型
# """


# 总体操作 1 2 3 4 5 6 7 8 9 次数的排名 rank
def rank_actionType_count(orderFuture, action):
    userid = []
    actionType_1_rank = []
    actionType_2_rank = []
    actionType_3_rank = []
    actionType_4_rank = []
    actionType_5_rank = []
    actionType_6_rank = []
    actionType_7_rank = []
    actionType_8_rank = []
    actionType_9_rank = []
    actionType_count = action.groupby(['userid', 'actionType'])['actionTime'].count().reset_index()
    actionType_count.rename(columns={'actionTime': 'actionType_count'}, inplace=True)
    actionType_count['actionType_count_rank'] = actionType_count.groupby(['userid'])['actionType_count'].rank(
        ascending=0, method='dense')
    # print(actionType_count)
    for index, row in orderFuture.iterrows():
        # print(index)
        userid.append(row.userid)
        actionType_count_user = actionType_count[actionType_count['userid'] == row.userid].reset_index()
        actionType_count_user_1 = actionType_count_user[actionType_count_user['actionType'] == 1].reset_index()
        actionType_count_user_2 = actionType_count_user[actionType_count_user['actionType'] == 2].reset_index()
        actionType_count_user_3 = actionType_count_user[actionType_count_user['actionType'] == 3].reset_index()
        actionType_count_user_4 = actionType_count_user[actionType_count_user['actionType'] == 4].reset_index()
        actionType_count_user_5 = actionType_count_user[actionType_count_user['actionType'] == 5].reset_index()
        actionType_count_user_6 = actionType_count_user[actionType_count_user['actionType'] == 6].reset_index()
        actionType_count_user_7 = actionType_count_user[actionType_count_user['actionType'] == 7].reset_index()
        actionType_count_user_8 = actionType_count_user[actionType_count_user['actionType'] == 8].reset_index()
        actionType_count_user_9 = actionType_count_user[actionType_count_user['actionType'] == 9].reset_index()
        if (len(actionType_count_user_1) > 0):
            actionType_1_rank.append(actionType_count_user_1['actionType_count_rank'][0])
        else:
            actionType_1_rank.append(None)

        if (len(actionType_count_user_2) > 0):
            actionType_2_rank.append(actionType_count_user_2['actionType_count_rank'][0])
        else:
            actionType_2_rank.append(None)

        if (len(actionType_count_user_3) > 0):
            actionType_3_rank.append(actionType_count_user_3['actionType_count_rank'][0])
        else:
            actionType_3_rank.append(None)

        if (len(actionType_count_user_4) > 0):
            actionType_4_rank.append(actionType_count_user_4['actionType_count_rank'][0])
        else:
            actionType_4_rank.append(None)

        if (len(actionType_count_user_5) > 0):
            actionType_5_rank.append(actionType_count_user_5['actionType_count_rank'][0])
        else:
            actionType_5_rank.append(None)

        if (len(actionType_count_user_6) > 0):
            actionType_6_rank.append(actionType_count_user_6['actionType_count_rank'][0])
        else:
            actionType_6_rank.append(None)

        if (len(actionType_count_user_7) > 0):
            actionType_7_rank.append(actionType_count_user_7['actionType_count_rank'][0])
        else:
            actionType_7_rank.append(None)

        if (len(actionType_count_user_8) > 0):
            actionType_8_rank.append(actionType_count_user_8['actionType_count_rank'][0])
        else:
            actionType_8_rank.append(None)

        if (len(actionType_count_user_9) > 0):
            actionType_9_rank.append(actionType_count_user_9['actionType_count_rank'][0])
        else:
            actionType_9_rank.append(None)
    orderFuture['actionType_1_rank'] = actionType_1_rank
    orderFuture['actionType_2_rank'] = actionType_2_rank
    orderFuture['actionType_3_rank'] = actionType_3_rank
    orderFuture['actionType_4_rank'] = actionType_4_rank
    orderFuture['actionType_5_rank'] = actionType_5_rank
    orderFuture['actionType_6_rank'] = actionType_6_rank
    orderFuture['actionType_7_rank'] = actionType_7_rank
    orderFuture['actionType_8_rank'] = actionType_8_rank
    orderFuture['actionType_9_rank'] = actionType_9_rank
    return orderFuture


# orderFuture_train = rank_actionType_count(orderFuture_train, action_train)
# orderFuture_test = rank_actionType_count(orderFuture_test, action_test)


# 对应操作 1 2 3 4 5 6 7 8 9 次数的排名 rank
def rank_actionType_count_c(orderFuture, action):
    userid = []
    actionType_1_rank = []
    actionType_2_rank = []
    actionType_3_rank = []
    actionType_4_rank = []
    actionType_5_rank = []
    actionType_6_rank = []
    actionType_7_rank = []
    actionType_8_rank = []
    actionType_9_rank = []
    action = action[action.orderid.isnull()]
    actionType_count = action.groupby(['userid', 'actionType'])['actionTime'].count().reset_index()
    actionType_count.rename(columns={'actionTime': 'actionType_count'}, inplace=True)
    actionType_count['actionType_count_rank'] = actionType_count.groupby(['userid'])['actionType_count'].rank(
        ascending=0, method='dense')
    # print(actionType_count)
    for index, row in orderFuture.iterrows():
        # print(index)
        userid.append(row.userid)
        actionType_count_user = actionType_count[actionType_count['userid'] == row.userid].reset_index()
        actionType_count_user_1 = actionType_count_user[actionType_count_user['actionType'] == 1].reset_index()
        actionType_count_user_2 = actionType_count_user[actionType_count_user['actionType'] == 2].reset_index()
        actionType_count_user_3 = actionType_count_user[actionType_count_user['actionType'] == 3].reset_index()
        actionType_count_user_4 = actionType_count_user[actionType_count_user['actionType'] == 4].reset_index()
        actionType_count_user_5 = actionType_count_user[actionType_count_user['actionType'] == 5].reset_index()
        actionType_count_user_6 = actionType_count_user[actionType_count_user['actionType'] == 6].reset_index()
        actionType_count_user_7 = actionType_count_user[actionType_count_user['actionType'] == 7].reset_index()
        actionType_count_user_8 = actionType_count_user[actionType_count_user['actionType'] == 8].reset_index()
        actionType_count_user_9 = actionType_count_user[actionType_count_user['actionType'] == 9].reset_index()
        if (len(actionType_count_user_1) > 0):
            actionType_1_rank.append(actionType_count_user_1['actionType_count_rank'][0])
        else:
            actionType_1_rank.append(None)

        if (len(actionType_count_user_2) > 0):
            actionType_2_rank.append(actionType_count_user_2['actionType_count_rank'][0])
        else:
            actionType_2_rank.append(None)

        if (len(actionType_count_user_3) > 0):
            actionType_3_rank.append(actionType_count_user_3['actionType_count_rank'][0])
        else:
            actionType_3_rank.append(None)

        if (len(actionType_count_user_4) > 0):
            actionType_4_rank.append(actionType_count_user_4['actionType_count_rank'][0])
        else:
            actionType_4_rank.append(None)

        if (len(actionType_count_user_5) > 0):
            actionType_5_rank.append(actionType_count_user_5['actionType_count_rank'][0])
        else:
            actionType_5_rank.append(None)

        if (len(actionType_count_user_6) > 0):
            actionType_6_rank.append(actionType_count_user_6['actionType_count_rank'][0])
        else:
            actionType_6_rank.append(None)

        if (len(actionType_count_user_7) > 0):
            actionType_7_rank.append(actionType_count_user_7['actionType_count_rank'][0])
        else:
            actionType_7_rank.append(None)

        if (len(actionType_count_user_8) > 0):
            actionType_8_rank.append(actionType_count_user_8['actionType_count_rank'][0])
        else:
            actionType_8_rank.append(None)

        if (len(actionType_count_user_9) > 0):
            actionType_9_rank.append(actionType_count_user_9['actionType_count_rank'][0])
        else:
            actionType_9_rank.append(None)
    orderFuture['actionType_1_rank_c'] = actionType_1_rank
    orderFuture['actionType_2_rank_c'] = actionType_2_rank
    orderFuture['actionType_3_rank_c'] = actionType_3_rank
    orderFuture['actionType_4_rank_c'] = actionType_4_rank
    orderFuture['actionType_5_rank_c'] = actionType_5_rank
    orderFuture['actionType_6_rank_c'] = actionType_6_rank
    orderFuture['actionType_7_rank_c'] = actionType_7_rank
    orderFuture['actionType_8_rank_c'] = actionType_8_rank
    orderFuture['actionType_9_rank_c'] = actionType_9_rank
    return orderFuture


# orderFuture_train = rank_actionType_count_c(orderFuture_train, action_train)
# orderFuture_test = rank_actionType_count_c(orderFuture_test, action_test)


# 用户使用APP的天数，分别是否老用户
def use_app_days_count(orderFuture, action):
    use_app_days = action.groupby(['userid', 'action_date']).count().reset_index()
    use_app_days_count = use_app_days.groupby(['userid'])['action_date'].count().reset_index()
    use_app_days_count.rename(columns={'action_date': 'use_app_days_count'}, inplace=True)
    orderFuture = pd.merge(orderFuture, use_app_days_count[['userid', 'use_app_days_count']], on='userid', how='left')
    return orderFuture

# orderFuture_train = use_app_days_count(orderFuture_train, action_train)
# orderFuture_test = use_app_days_count(orderFuture_test, action_test)

# 只有一次操作的用户最后一次操作的类型
def action1_last_type(orderFuture, action):
    orderFuture['action1_last_type_is7'] = None
    orderFuture['action1_last_type_is7'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] != 7)] = 0
    orderFuture['action1_last_type_is7'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] == 7)] = 1

    orderFuture['action1_last_type_is6'] = None
    orderFuture['action1_last_type_is6'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] != 6)] = 0
    orderFuture['action1_last_type_is6'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] == 6)] = 1

    orderFuture['action1_last_type_is5'] = None
    orderFuture['action1_last_type_is5'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] != 5)] = 0
    orderFuture['action1_last_type_is5'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] == 5)] = 1

    orderFuture['action1_last_type_is9'] = None
    orderFuture['action1_last_type_is9'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] != 9)] = 0
    orderFuture['action1_last_type_is9'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] == 9)] = 1

    orderFuture['action1_last_type_is1'] = None
    orderFuture['action1_last_type_is1'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] != 1)] = 0
    orderFuture['action1_last_type_is1'][(orderFuture['action_all'] == 1) & (orderFuture['latest_actionType'] == 1)] = 1
    return orderFuture

# orderFuture_train = action1_last_type(orderFuture_train, action_train)
# orderFuture_test = action1_last_type(orderFuture_test, action_test)


# 大间隔的距离
def latest_bigspan(orderFuture, action):
    result = pd.DataFrame(columns=['userid', 'latest1_bigspan', 'latest2_bigspan', 'latest3_bigspan', 'latest4_bigspan', 'latest5_bigspan'])
    userid = []
    latest1_bigspan = []
    latest2_bigspan = []
    latest3_bigspan = []
    latest4_bigspan = []
    latest5_bigspan = []

    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)

    action_span = action[action['actionType_time'] > 129600]
    action_span['span_index'] = action_span.index
    action_span = pd.merge(action_span, latest, on='userid', how='left')

    action_span['latest_bigspan'] = action_span['max_index'] - action_span['span_index'] + 1
    # print(action_span[['userid', 'action_time', 'actionType_time', 'max_index', 'span_index', 'latest_bigspan']])

    latest_span = action_span.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest_span.rename(columns={'actionTime': 'span_max_index'}, inplace=True)

    for index, row in latest_span.iterrows():
        userid.append(row.userid)
        action_span_user = action_span[action_span['userid'] == row.userid]
        if (len(action_span_user) >= 1):
            latest1_bigspan.append(action_span_user['latest_bigspan'][row.span_max_index])
        else:
            latest1_bigspan.append(None)

        if (len(action_span_user) >= 2):
            latest2_bigspan.append(action_span_user['latest_bigspan'][row.span_max_index-1])
        else:
            latest2_bigspan.append(None)

        if (len(action_span_user) >= 3):
            latest3_bigspan.append(action_span_user['latest_bigspan'][row.span_max_index-2])
        else:
            latest3_bigspan.append(None)

        if (len(action_span_user) >= 4):
            latest4_bigspan.append(action_span_user['latest_bigspan'][row.span_max_index-3])
        else:
           latest4_bigspan.append(None)

        if (len(action_span_user) >= 5):
            latest5_bigspan.append(action_span_user['latest_bigspan'][row.span_max_index-4])
        else:
           latest5_bigspan.append(None)

    result['userid'] = userid
    result['latest1_bigspan'] = latest1_bigspan
    result['latest2_bigspan'] = latest2_bigspan
    result['latest3_bigspan'] = latest3_bigspan
    result['latest4_bigspan'] = latest4_bigspan
    result['latest5_bigspan'] = latest5_bigspan
    orderFuture = pd.merge(orderFuture, result[['userid', 'latest1_bigspan', 'latest2_bigspan', 'latest3_bigspan', 'latest4_bigspan', 'latest5_bigspan']], on='userid', how='left')
    return orderFuture

# orderFuture_train = latest_bigspan(orderFuture_train, action_train)
# orderFuture_test = latest_bigspan(orderFuture_test, action_test)

# 全部56 不连续的时间
def discontinuous_56_count(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'discontinuou_56_count'])
    userid = []
    discontinuou_56_count = []
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count56 = 0
        for i in range(len(action1)):
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] != 6)):
                if ((action1['actionType'][i + 1] == 5) and (action1['actionType_time'][i + 1] < 600)):
                    count56 = count56
                else:
                    count56 = count56 + 1
        userid.append(row.userid)
        discontinuou_56_count.append(count56)
    count['userid'] = userid
    count['discontinuou_56_count'] = discontinuou_56_count
    orderFuture = pd.merge(orderFuture, count[['userid', 'discontinuou_56_count']], on='userid', how='left')
    return orderFuture


# orderFuture_train = discontinuous_56_count(orderFuture_train, action_train)
# orderFuture_test = discontinuous_56_count(orderFuture_test, action_test)


# 对应56 不连续的时间
def discontinuous_56_count_c(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'discontinuou_56_count'])
    userid = []
    discontinuou_56_count = []
    action = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count56 = 0
        for i in range(len(action1)):
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] != 6)):
                if ((action1['actionType'][i + 1] == 5) and (action1['actionType_time'][i + 1] < 600)):
                    count56 = count56
                else:
                    count56 = count56 + 1
        userid.append(row.userid)
        discontinuou_56_count.append(count56)
    count['userid'] = userid
    count['discontinuou_56_count_c'] = discontinuou_56_count
    orderFuture = pd.merge(orderFuture, count[['userid', 'discontinuou_56_count_c']], on='userid', how='left')
    return orderFuture


# orderFuture_train = discontinuous_56_count_c(orderFuture_train, action_train)
# orderFuture_test = discontinuous_56_count_c(orderFuture_test, action_test)


# 连续55、15、156、1556操作的时间
def continuous_55_count(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'continuou_15_time', 'continuou_55_time', 'continuou_156_time', 'continuou_1556_time'])
    userid = []
    continuou_15_count = []
    continuou_55_count = []
    action_156_count = []
    action_1556_count = []
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count15 = 0
        count55 = 0
        count156 = 0
        count1556 = 0
        for i in range(len(action1)):
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 1) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType_time'][i + 1] < 1800)):
                    count15 = count15 + action1['actionType_time'][i + 1]

            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType_time'][i + 1] < 1800)):
                    count55 = count55 + action1['actionType_time'][i + 1]

            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 1) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType'][i + 2] == 6) and (action1['actionType_time'][i + 1] < 1800)
                and (action1['actionType_time'][i + 2] < 1800)):
                count156 = count156 + (action1['actionType_time'][i + 1] + action1['actionType_time'][i + 2])

            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 1) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType'][i + 2] == 5) and (action1['actionType'][i + 3] == 6)
                and (action1['actionType_time'][i + 1] < 1800) and (action1['actionType_time'][i + 2] < 1800)
                and (action1['actionType_time'][i + 3] < 1800)):
                count1556 = count1556 + (action1['actionType_time'][i + 1] + (action1['actionType_time'][i + 2] < 1800) + (action1['actionType_time'][i + 3] < 1800))

        userid.append(row.userid)
        continuou_15_count.append(count15)
        continuou_55_count.append(count55)
        action_156_count.append(count156)
        action_1556_count.append(count1556)
    count['userid'] = userid
    count['continuou_15_time'] = continuou_15_count
    count['continuou_55_time'] = continuou_55_count
    count['continuou_156_time'] = action_156_count
    count['continuou_1556_time'] = action_1556_count
    count['continuou_15_time'][count['continuou_15_time'] == 0] = None
    count['continuou_55_time'][count['continuou_55_time'] == 0] = None
    count['continuou_156_time'][count['continuou_156_time'] == 0] = None
    count['continuou_1556_time'][count['continuou_1556_time'] == 0] = None
    orderFuture = pd.merge(orderFuture, count[['userid', 'continuou_15_time', 'continuou_55_time', 'continuou_156_time', 'continuou_1556_time']], on='userid', how='left')
    return orderFuture

# orderFuture_train = continuous_55_count(orderFuture_train, action_train)
# orderFuture_test = continuous_55_count(orderFuture_test, action_test)


# 对应连续55、15、156、1556操作的时间
def continuous_55_count_c(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'continuou_15_time_c', 'continuou_55_time_c', 'continuou_156_time_c', 'continuou_1556_time_c'])
    userid = []
    continuou_15_count = []
    continuou_55_count = []
    action_156_count = []
    action_1556_count = []
    action = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        action1 = action[action['userid'] == row.userid].reset_index()
        count15 = 0
        count55 = 0
        count156 = 0
        count1556 = 0
        for i in range(len(action1)):
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 1) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType_time'][i + 1] < 1800)):
                    count15 = count15 + action1['actionType_time'][i + 1]

            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType_time'][i + 1] < 1800)):
                    count55 = count55 + action1['actionType_time'][i + 1]

            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 1) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType'][i + 2] == 6) and (action1['actionType_time'][i + 1] < 1800)
                and (action1['actionType_time'][i + 2] < 1800)):
                count156 = count156 + (action1['actionType_time'][i + 1] + action1['actionType_time'][i + 2])

            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 1) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType'][i + 2] == 5) and (action1['actionType'][i + 3] == 6)
                and (action1['actionType_time'][i + 1] < 1800) and (action1['actionType_time'][i + 2] < 1800)
                and (action1['actionType_time'][i + 3] < 1800)):
                count1556 = count1556 + (action1['actionType_time'][i + 1] + (action1['actionType_time'][i + 2] < 1800) + (action1['actionType_time'][i + 3] < 1800))

        userid.append(row.userid)
        continuou_15_count.append(count15)
        continuou_55_count.append(count55)
        action_156_count.append(count156)
        action_1556_count.append(count1556)
    count['userid'] = userid
    count['continuou_15_time_c'] = continuou_15_count
    count['continuou_55_time_c'] = continuou_55_count
    count['continuou_156_time_c'] = action_156_count
    count['continuou_1556_time_c'] = action_1556_count
    count['continuou_15_time_c'][count['continuou_15_time_c'] == 0] = None
    count['continuou_55_time_c'][count['continuou_55_time_c'] == 0] = None
    count['continuou_156_time_c'][count['continuou_156_time_c'] == 0] = None
    count['continuou_1556_time_c'][count['continuou_1556_time_c'] == 0] = None
    orderFuture = pd.merge(orderFuture, count[['userid', 'continuou_15_time_c', 'continuou_55_time_c', 'continuou_156_time_c', 'continuou_1556_time_c']], on='userid', how='left')
    return orderFuture

# orderFuture_train = continuous_55_count_c(orderFuture_train, action_train)
# orderFuture_test = continuous_55_count_c(orderFuture_test, action_test)


# actionType5-6的时间差小于timespanthred的数量
def getActionTimeSpan(orderFuture, action, actiontypeA, actiontypeB, timethred=200):
    userid = []
    actiontimespancount_5_6_c = []
    action = action[action.orderid.isnull()]
    for index, row in orderFuture.iterrows():
        print(index)
        userid.append(row.userid)
        df_action_of_userid = action[action['userid'] == row.userid]
        timespan_list = []
        i = 0
        while i < (len(df_action_of_userid)-1):
            if df_action_of_userid['actionType'].iat[i] == actiontypeA:
                timeA = df_action_of_userid['actionTime'].iat[i]
                for j in range(i+1, len(df_action_of_userid)):
                    if df_action_of_userid['actionType'].iat[j] == actiontypeA:
                        timeA = df_action_of_userid['actionTime'].iat[j]
                    if df_action_of_userid['actionType'].iat[j] == actiontypeB:
                        timeB = df_action_of_userid['actionTime'].iat[j]
                        timespan_list.append(timeB-timeA)
                        i = j
                        break
            i += 1
        actiontimespancount_5_6_c.append(np.sum(np.array(timespan_list) <= timethred) / (np.sum(np.array(timespan_list)) + 1.0))
    orderFuture['actiontimespancount_5_6_c'] = actiontimespancount_5_6_c
    return orderFuture

timespanthred=200
# orderFuture_train = getActionTimeSpan(orderFuture_train, action_train, 5, 6, timespanthred)
# orderFuture_test = getActionTimeSpan(orderFuture_test, action_test, 5, 6, timespanthred)


# 倒数第二/三次的actionType5与6之间的时间间隔
def get2ActionTimeSpanLast(orderFuture, action, actiontypeA, actiontypeB, k):
    userid = []
    actiontimespanlast2_5_6 = []
    for index, row in orderFuture.iterrows():
        print(index)
        userid.append(row.userid)
        df_action_of_userid = action[action['userid'] == row.userid]
        timespan_list = []
        i = 0
        while i < (len(df_action_of_userid)-1):
            if df_action_of_userid['actionType'].iat[i] == actiontypeA:
                timeA = df_action_of_userid['actionTime'].iat[i]
                for j in range(i+1, len(df_action_of_userid)):
                    if df_action_of_userid['actionType'].iat[j] == actiontypeA:
                        timeA = df_action_of_userid['actionTime'].iat[j]
                        continue
                    if df_action_of_userid['actionType'].iat[j] == actiontypeB:
                        timeB = df_action_of_userid['actionTime'].iat[j]
                        timespan_list.append(timeB-timeA)
                        i = j
                        break
            i += 1
        if len(timespan_list) >= k:
            actiontimespanlast2_5_6.append(timespan_list[-k])
        else:
            actiontimespanlast2_5_6.append(-1)

    orderFuture['actiontimespanlast'+ str(k) +'_5_6'] = actiontimespanlast2_5_6
    return orderFuture

# orderFuture_train = get2ActionTimeSpanLast(orderFuture_train, action_train, 5, 6, 2)
# orderFuture_test = get2ActionTimeSpanLast(orderFuture_test, action_test, 5, 6, 2)

# orderFuture_train = get2ActionTimeSpanLast(orderFuture_train, action_train, 5, 6, 3)
# orderFuture_test = get2ActionTimeSpanLast(orderFuture_test, action_test, 5, 6, 3)


# 最后20次的时间间隔
def latest20_actionType_time(orderFuture, action):
    result = pd.DataFrame(columns=['userid', 'latest1_actionType_time', 'latest2_actionType_time', 'latest3_actionType_time',
                                   'latest4_actionType_time', 'latest5_actionType_time', 'latest6_actionType_time',
                                   'latest7_actionType_time', 'latest8_actionType_time', 'latest9_actionType_time',
                                   'latest10_actionType_time', 'latest11_actionType_time', 'latest12_actionType_time',
                                   'latest13_actionType_time', 'latest14_actionType_time', 'latest15_actionType_time',
                                   'latest16_actionType_time', 'latest17_actionType_time', 'latest18_actionType_time',
                                   'latest19_actionType_time', 'latest20_actionType_time'
                                   ])
    userid = []
    latest1_actionType_time = []
    latest2_actionType_time = []
    latest3_actionType_time = []
    latest4_actionType_time = []
    latest5_actionType_time = []
    latest6_actionType_time = []
    latest7_actionType_time = []
    latest8_actionType_time = []
    latest9_actionType_time = []
    latest10_actionType_time = []
    latest11_actionType_time = []
    latest12_actionType_time = []
    latest13_actionType_time = []
    latest14_actionType_time = []
    latest15_actionType_time = []
    latest16_actionType_time = []
    latest17_actionType_time = []
    latest18_actionType_time = []
    latest19_actionType_time = []
    latest20_actionType_time = []

    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    for index, row in latest.iterrows():
        print(index)
        userid.append(row.userid)
        action_span_user = action[action['userid'] == row.userid]

        if (len(action_span_user) >= 1):
            latest1_actionType_time.append(action_span_user['actionType_time'][row.max_index])
        else:
            latest1_actionType_time.append(None)

        if (len(action_span_user) >= 2):
            latest2_actionType_time.append(action_span_user['actionType_time'][row.max_index-1])
        else:
            latest2_actionType_time.append(None)

        if (len(action_span_user) >= 3):
            latest3_actionType_time.append(action_span_user['actionType_time'][row.max_index-2])
        else:
            latest3_actionType_time.append(None)

        if (len(action_span_user) >= 4):
            latest4_actionType_time.append(action_span_user['actionType_time'][row.max_index-3])
        else:
            latest4_actionType_time.append(None)

        if (len(action_span_user) >= 5):
            latest5_actionType_time.append(action_span_user['actionType_time'][row.max_index-4])
        else:
            latest5_actionType_time.append(None)

        if (len(action_span_user) >= 6):
            latest6_actionType_time.append(action_span_user['actionType_time'][row.max_index-5])
        else:
            latest6_actionType_time.append(None)

        if (len(action_span_user) >= 7):
            latest7_actionType_time.append(action_span_user['actionType_time'][row.max_index-6])
        else:
            latest7_actionType_time.append(None)

        if (len(action_span_user) >= 8):
            latest8_actionType_time.append(action_span_user['actionType_time'][row.max_index-7])
        else:
            latest8_actionType_time.append(None)

        if (len(action_span_user) >= 9):
            latest9_actionType_time.append(action_span_user['actionType_time'][row.max_index-8])
        else:
            latest9_actionType_time.append(None)

        if (len(action_span_user) >= 10):
            latest10_actionType_time.append(action_span_user['actionType_time'][row.max_index-9])
        else:
            latest10_actionType_time.append(None)

        if (len(action_span_user) >= 11):
            latest11_actionType_time.append(action_span_user['actionType_time'][row.max_index-10])
        else:
            latest11_actionType_time.append(None)

        if (len(action_span_user) >= 12):
            latest12_actionType_time.append(action_span_user['actionType_time'][row.max_index-11])
        else:
            latest12_actionType_time.append(None)

        if (len(action_span_user) >= 13):
            latest13_actionType_time.append(action_span_user['actionType_time'][row.max_index-12])
        else:
            latest13_actionType_time.append(None)

        if (len(action_span_user) >= 14):
            latest14_actionType_time.append(action_span_user['actionType_time'][row.max_index-13])
        else:
            latest14_actionType_time.append(None)

        if (len(action_span_user) >= 15):
            latest15_actionType_time.append(action_span_user['actionType_time'][row.max_index-14])
        else:
            latest15_actionType_time.append(None)

        if (len(action_span_user) >= 16):
            latest16_actionType_time.append(action_span_user['actionType_time'][row.max_index-15])
        else:
            latest16_actionType_time.append(None)

        if (len(action_span_user) >= 17):
            latest17_actionType_time.append(action_span_user['actionType_time'][row.max_index-16])
        else:
            latest17_actionType_time.append(None)

        if (len(action_span_user) >= 18):
            latest18_actionType_time.append(action_span_user['actionType_time'][row.max_index-17])
        else:
            latest18_actionType_time.append(None)

        if (len(action_span_user) >= 19):
            latest19_actionType_time.append(action_span_user['actionType_time'][row.max_index-18])
        else:
            latest19_actionType_time.append(None)

        if (len(action_span_user) >= 20):
            latest20_actionType_time.append(action_span_user['actionType_time'][row.max_index-19])
        else:
            latest20_actionType_time.append(None)

    result['userid'] = userid
    result['latest1_actionType_time'] = latest1_actionType_time
    result['latest2_actionType_time'] = latest2_actionType_time
    result['latest3_actionType_time'] = latest3_actionType_time
    result['latest4_actionType_time'] = latest4_actionType_time
    result['latest5_actionType_time'] = latest5_actionType_time
    result['latest6_actionType_time'] = latest6_actionType_time
    result['latest7_actionType_time'] = latest7_actionType_time
    result['latest8_actionType_time'] = latest8_actionType_time
    result['latest9_actionType_time'] = latest9_actionType_time
    result['latest10_actionType_time'] = latest10_actionType_time
    result['latest11_actionType_time'] = latest11_actionType_time
    result['latest12_actionType_time'] = latest12_actionType_time
    result['latest13_actionType_time'] = latest13_actionType_time
    result['latest14_actionType_time'] = latest14_actionType_time
    result['latest15_actionType_time'] = latest15_actionType_time
    result['latest16_actionType_time'] = latest16_actionType_time
    result['latest17_actionType_time'] = latest17_actionType_time
    result['latest18_actionType_time'] = latest18_actionType_time
    result['latest19_actionType_time'] = latest19_actionType_time
    result['latest20_actionType_time'] = latest20_actionType_time
    orderFuture = pd.merge(orderFuture, result[['userid',
                        'latest1_actionType_time', 'latest2_actionType_time', 'latest3_actionType_time',
                        'latest4_actionType_time', 'latest5_actionType_time', 'latest6_actionType_time',
                        'latest7_actionType_time', 'latest8_actionType_time', 'latest9_actionType_time',
                        'latest10_actionType_time', 'latest11_actionType_time', 'latest12_actionType_time',
                        'latest13_actionType_time', 'latest14_actionType_time', 'latest15_actionType_time',
                        'latest16_actionType_time', 'latest17_actionType_time', 'latest18_actionType_time',
                        'latest19_actionType_time', 'latest20_actionType_time']], on='userid', how='left')
    return orderFuture

# orderFuture_train = latest20_actionType_time(orderFuture_train, action_train)
# orderFuture_test = latest20_actionType_time(orderFuture_test, action_test)




# print(orderFuture_train)
# print(orderFuture_test)



print("开始提取：", start)
print("提取完成：", datetime.datetime.now())
orderFuture_train.to_csv(dire + 'train8_1_select1.csv', index=False, encoding='utf-8')
orderFuture_test.to_csv(dire + 'test8_1_select1.csv', index=False, encoding='utf-8')