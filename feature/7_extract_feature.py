# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime
# from datetime import datetime

dire = '../../data/'
start = datetime.datetime.now()
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderFuture_train = pd.read_csv(dire + 'train/orderFuture_train6.csv', encoding='utf-8')
userProfile_train = pd.read_csv(dire + 'train/userProfile_train.csv', encoding='utf-8')
userComment_train = pd.read_csv(dire + 'train/userComment_train.csv', encoding='utf-8')
action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')

orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
orderFuture_test = pd.read_csv(dire + 'test/orderFuture_test6.csv', encoding='utf-8')
userProfile_test = pd.read_csv(dire + 'test/userProfile_test.csv', encoding='utf-8')
userComment_test = pd.read_csv(dire + 'test/userComment_test.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')
# """

############# 1.user feature   #############
"""
# 1. 

"""

############# 2.history order feature   #############
"""
# 1.

"""

############# 3.action feature   #############
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


############# 4.time feature   #############
"""
# 1. 季节特征


"""

############# 4.comment feature   #############
"""
# 1. 

"""
# 用户评论标签，枚举特征
def comment_tags_type1(orderFuture, userComment):
    comment_tags = pd.DataFrame(columns=['userid', 'tags'])
    userid = []
    tags = []
    for index, row in userComment.iterrows():
        # print(row.tags)
        for tag in str(row.tags).split('|'):
            if (tag != 'nan'):
                userid.append(row.userid)
                tags.append(tag)
    comment_tags['userid'] = userid
    comment_tags['tags'] = tags

    tag_list = list(set(list(comment_tags.tags)))
    column = ['userid']
    for tag_l in tag_list:
        column.append('tag_' + str(tag_l))
    comment_tags_list = pd.DataFrame(columns=column)
    comment_tags_list['userid'] = orderFuture['userid']

    for index, row in comment_tags.iterrows():
        comment_tags_list['tag_' + str(row.tags)][comment_tags_list['userid'] == row.userid] = 1

    print(comment_tags_list)
    orderFuture = pd.merge(orderFuture, comment_tags_list, on='userid', how='left')
    return orderFuture

# orderFuture_train = comment_tags_type1(orderFuture_train, userComment_train)
# orderFuture_test = comment_tags_type1(orderFuture_test, userComment_test)


# 用户评论表的标签数
def comment_tags_count(orderFuture, userComment):
    comment_tags = pd.DataFrame(columns=['userid', 'tags'])
    userid = []
    tags = []
    for index, row in userComment.iterrows():
        for tag in str(row.tags).split('|'):
            if (tag != 'nan'):
                userid.append(row.userid)
                tags.append(tag)
    comment_tags['userid'] = userid
    comment_tags['tags'] = tags

    comment_KeyWords = pd.DataFrame(columns=['userid', 'KeyWords'])
    userid = []
    KeyWords = []
    for index, row in userComment.iterrows():
        for KeyWord in str(row.commentsKeyWords).split(','):
            if (KeyWord != 'nan'):
                userid.append(row.userid)
                KeyWords.append(KeyWord)
    comment_KeyWords['userid'] = userid
    comment_KeyWords['KeyWords'] = KeyWords
    comment_tags = comment_tags.groupby(['userid']).count().reset_index()
    comment_tags.rename(columns={'tags': 'tag_count'}, inplace=True)
    comment_KeyWords = comment_KeyWords.groupby(['userid']).count().reset_index()
    comment_KeyWords.rename(columns={'KeyWords': 'KeyWord_count'}, inplace=True)
    orderFuture = pd.merge(orderFuture, comment_tags, on='userid', how='left')
    # orderFuture = pd.merge(orderFuture, comment_KeyWords, on='userid', how='left')
    return orderFuture

# orderFuture_train = comment_tags_count(orderFuture_train, userComment_train)
# orderFuture_test = comment_tags_count(orderFuture_test, userComment_test)


# print(orderFuture_train)
# print(orderFuture_test)

print("开始提取：", start)
print("提取完成：", datetime.datetime.now())
orderFuture_train.to_csv(dire + 'train3.csv', index=False, encoding='utf-8')
orderFuture_test.to_csv(dire + 'test3.csv', index=False, encoding='utf-8')