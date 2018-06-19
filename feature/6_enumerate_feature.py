# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime

dire = '../../data/'
start = datetime.now()
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderFuture_train = pd.read_csv(dire + 'train/orderFuture_train3.csv', encoding='utf-8')
userProfile_train = pd.read_csv(dire + 'train/userProfile_train.csv', encoding='utf-8')
userComment_train = pd.read_csv(dire + 'train/userComment_train.csv', encoding='utf-8')
action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')

orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
orderFuture_test = pd.read_csv(dire + 'test/orderFuture_test3.csv', encoding='utf-8')
userProfile_test = pd.read_csv(dire + 'test/userProfile_test.csv', encoding='utf-8')
userComment_test = pd.read_csv(dire + 'test/userComment_test.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')




############# 3.action feature   #############
"""
# 1. 每日用户action的次数
# 2. 每日用户action的时间

# """




# 每日用户action的次数
def user_day_count(orderFuture, action):
    temp = pd.DatetimeIndex(action['action_time'])
    action['action_date'] = temp.date
    action_date_count = action.groupby(['userid', 'action_date'])['actionType'].count().reset_index()
    action_date_count.rename(columns={'actionType': 'action_date_count'}, inplace=True)

    action_date_list = list(set(list(action.action_date)))
    column = ['userid']
    for action_date in action_date_list:
        column.append('user_day_count_' + str(action_date))
    user_date_count = pd.DataFrame(columns=column)
    user_date_count['userid'] = orderFuture['userid']

    for index, row in user_date_count.iterrows():
        print(index)
        print(row.userid)
        action_date_cnt1 = action_date_count[action_date_count['userid'] == row.userid]
        for action_date in action_date_list:
            action_date_cnt = action_date_cnt1[action_date_cnt1['action_date'] == action_date].reset_index()
            if (len(action_date_cnt) > 0):
                user_date_count['user_day_count_' + str(action_date)][user_date_count['userid'] == row.userid] = \
                action_date_cnt['action_date_count'][0]
    orderFuture = pd.merge(orderFuture, user_date_count, on='userid', how='left')
    return orderFuture

# orderFuture_train = user_day_count(orderFuture_train, action_train)
# orderFuture_test = user_day_count(orderFuture_test, action_test)



# 每日用户action的时间
def user_day_time(orderFuture, action):

    action_date_count = action.groupby(['userid', 'action_date'])['actionType_time'].sum().reset_index()
    action_date_count.rename(columns={'actionType_time': 'actionType_time_sum'}, inplace=True)
    print(action_date_count)

    action_date_list = list(set(list(action.action_date)))
    column = ['userid']
    for action_date in action_date_list:
        column.append('user_day_time_' + str(action_date))
    user_date_count = pd.DataFrame(columns=column)
    user_date_count['userid'] = orderFuture['userid']

    for index, row in user_date_count.iterrows():
        print(index)
        print(row.userid)
        action_date_cnt1 = action_date_count[action_date_count['userid'] == row.userid]
        for action_date in action_date_list:
            action_date_cnt = action_date_cnt1[action_date_cnt1['action_date'] == action_date].reset_index()
            if (len(action_date_cnt) > 0):
                user_date_count['user_day_time_' + str(action_date)][user_date_count['userid'] == row.userid] = \
                action_date_cnt['actionType_time_sum'][0]
        print(user_date_count)
    orderFuture = pd.merge(orderFuture, user_date_count, on='userid', how='left')
    return orderFuture

orderFuture_train = user_day_time(orderFuture_train, action_train)
orderFuture_test = user_day_time(orderFuture_test, action_test)


print("开始提取：", start)
print("提取完成：", datetime.now())
# orderFuture_train.to_csv(dire + 'train_enumerate.csv', index=False, encoding='utf-8')
# orderFuture_test.to_csv(dire + 'test_enumerate.csv', index=False, encoding='utf-8')