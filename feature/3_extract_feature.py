# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime


dire = '../../data/'
start = datetime.now()
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderFuture_train = pd.read_csv(dire + 'train/orderFuture_train.csv', encoding='utf-8')
userProfile_train = pd.read_csv(dire + 'train/userProfile_train.csv', encoding='utf-8')
userComment_train = pd.read_csv(dire + 'train/userComment_train.csv', encoding='utf-8')
action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')

orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
orderFuture_test = pd.read_csv(dire + 'test/orderFuture_test.csv', encoding='utf-8')
userProfile_test = pd.read_csv(dire + 'test/userProfile_test.csv', encoding='utf-8')
userComment_test = pd.read_csv(dire + 'test/userComment_test.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')
# """

############# 1.user feature   #############
"""
# 1. 用户表关联

"""
# 用户表关联
def merge_user_table(orderFuture, userProfile):
    order = pd.merge(orderFuture, userProfile, on='userid', how='left')
    return order

orderFuture_train = merge_user_table(orderFuture_train, userProfile_train)
orderFuture_test = merge_user_table(orderFuture_test, userProfile_test)


############# 2.history order feature   #############
"""
# 1. 评论表关联
# 2. 历史订单数量，1 的数量和占比
# 3. 最近依次出行是否为 1

"""
# 评论表关联
def merge_comment_table(orderHistory, userComment):
    order = pd.merge(orderHistory, userComment[['orderid', 'rating']], on='orderid', how='left')
    return order

orderHistory_train = merge_comment_table(orderHistory_train, userComment_train)
orderHistory_test = merge_comment_table(orderHistory_test, userComment_test)


# 历史订单数、为1的数量和占比
def count_1(orderFuture, orderHistory):
    count_all = orderHistory.groupby(orderHistory.userid)['orderid'].count().reset_index()  # 每个用户订单总数
    count_all.rename(columns={'orderid': 'count_all'}, inplace=True)
    orderHistory_1 = orderHistory[orderHistory['orderType'] == 1]
    count_1 = orderHistory_1.groupby(orderHistory_1.userid)['orderid'].count().reset_index()  # 为 1 订单总数
    count_1.rename(columns={'orderid': 'count_1'}, inplace=True)
    count = pd.merge(count_all, count_1, on='userid', how='left').fillna(0)
    count['type_1_rate'] = count['count_1']/count['count_all']
    # print(count)
    orderFuture = pd.merge(orderFuture, count, on='userid', how='left')
    return orderFuture


orderFuture_train = count_1(orderFuture_train, orderHistory_train)
orderFuture_test = count_1(orderFuture_test, orderHistory_test)


# 历史出现0的次数, 和0所占比例
def count_0(orderFuture):
    count_0 = []
    type_0_rate = []
    for index, row in orderFuture.iterrows():
        if(row.count_all >= 0):
            count_0.append(row.count_all - row.count_1)
            type_0_rate.append((row.count_all - row.count_1) / row.count_all)
        else:
            count_0.append(1)
            type_0_rate.append(1)

    orderFuture['count_0'] = count_0
    orderFuture['type_0_rate'] = type_0_rate
    return orderFuture

orderFuture_train = count_0(orderFuture_train)
orderFuture_test = count_0(orderFuture_test)


# 历史订单是否出现过1
def ever_1(orderFuture):
    ever_1 = []
    for index, row in orderFuture.iterrows():
        if(row.count_1 > 0):
            ever_1.append(1)
        else:
            ever_1.append(0)
    return ever_1

orderFuture_train['ever_1'] = ever_1(orderFuture_train)
orderFuture_test['ever_1'] = ever_1(orderFuture_test)


# 历史订单最近一次是什么类型0/1
def latest_type(orderFuture, orderHistory):
    latest = orderHistory[['userid', 'orderid', 'orderType', 'order_time']].groupby(['userid']).last().reset_index()
    latest.rename(columns={'orderType': 'latest_type'}, inplace=True)
    orderFuture = pd.merge(orderFuture, latest[['userid', 'latest_type']], on='userid', how='left')  # 历史下单 类型为最近历史订单
    orderFuture['latest_type'][~(orderFuture['latest_type'] >= 0)] = 0   # 历史未下单 类型为0
    return orderFuture

orderFuture_train = latest_type(orderFuture_train, orderHistory_train)
orderFuture_test = latest_type(orderFuture_test, orderHistory_test)


# 历史订单最近一次去的州、国家、城市
def latest_place(orderFuture, orderHistory):
    # city
    orderHistory_1 = orderHistory[orderHistory['orderType'] == 1]
    count_1 = orderHistory_1.groupby(orderHistory_1.city_e)['orderType'].count().reset_index()    # 城市为 1 的次数
    count = orderHistory.groupby(orderHistory.city_e)['orderType'].count().reset_index()          # 城市 的总次数
    count_1.rename(columns={'orderType': 'city_1'}, inplace=True)
    count.rename(columns={'orderType': 'city'}, inplace=True)
    city = pd.merge(count, count_1, on='city_e', how='left')
    city['city_1_rate'] = city['city_1']/city['city']
    # country
    orderHistory_1 = orderHistory[orderHistory['orderType'] == 1]
    count_1 = orderHistory_1.groupby(orderHistory_1.country_e)['orderType'].count().reset_index()    # 国家为 1 的次数
    count = orderHistory.groupby(orderHistory.country_e)['orderType'].count().reset_index()          # 国家 的总次数
    count_1.rename(columns={'orderType': 'country_1'}, inplace=True)
    count.rename(columns={'orderType': 'country'}, inplace=True)
    country = pd.merge(count, count_1, on='country_e', how='left')
    country['country_1_rate'] = country['country_1']/country['country']

    latest = orderHistory[['userid', 'orderid', 'orderType', 'order_time', 'city_e', 'country_e', 'continent_e']].groupby(['userid']).last().reset_index()
    orderFuture = pd.merge(orderFuture, latest[['userid', 'city_e', 'country_e', 'continent_e']], on='userid', how='left')  # 历史下单 类型为最近历史订单
    # orderFuture = pd.merge(orderFuture, city[['city_e', 'city_1_rate']], on='city_e', how='left')
    # orderFuture = pd.merge(orderFuture, country[['country_e', 'country_1_rate']], on='country_e', how='left')
    return orderFuture

orderFuture_train = latest_place(orderFuture_train, orderHistory_train)
orderFuture_test = latest_place(orderFuture_test, orderHistory_test)


# 历史订单评分平均
def history_ave_rating(orderFuture, orderHistory):
    avg_rating = orderHistory.groupby(orderHistory.userid)['rating'].mean().reset_index()
    avg_rating.rename(columns={'rating': 'history_avg_rating'}, inplace=True)
    orderFuture = pd.merge(orderFuture, avg_rating, on='userid', how='left')
    return orderFuture

orderFuture_train = history_ave_rating(orderFuture_train, orderHistory_train)
orderFuture_test = history_ave_rating(orderFuture_test, orderHistory_test)


# 用户历史评分平均
def user_history_ave_rating(orderFuture, userComment):
    user_avg_rating = userComment.groupby(userComment.userid)['rating'].mean().reset_index()
    user_avg_rating.rename(columns={'rating': 'user_history_ave_rating'}, inplace=True)
    orderFuture = pd.merge(orderFuture, user_avg_rating, on='userid', how='left')
    return orderFuture

orderFuture_train = user_history_ave_rating(orderFuture_train, orderHistory_train)
orderFuture_test = user_history_ave_rating(orderFuture_test, orderHistory_test)

############# 3.action feature   #############
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
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6)):
                count56 = count56 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7)):
                count67 = count67 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8)):
                count78 = count78 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 8) and (action1['actionType'][i + 1] == 9)):
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
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6)):
                count56 = count56 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7)):
                count67 = count67 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8)):
                count78 = count78 + 1
            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 8) and (action1['actionType'][i + 1] == 9)):
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
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7)):
                count567 = count567 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7) and (action1['actionType'][i + 2] == 8)):
                count678 = count678 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8) and (action1['actionType'][i + 2] == 9)):
                count789 = count789 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 6)):
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
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7)):
                count567 = count567 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7) and (action1['actionType'][i + 2] == 8)):
                count678 = count678 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 7) and (action1['actionType'][i + 1] == 8) and (action1['actionType'][i + 2] == 9)):
                count789 = count789 + 1
            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 6)):
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
            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7) and (action1['actionType'][i + 3] == 8)):
                count5678 = count5678 + 1
            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7) and (action1['actionType'][i + 2] == 8) and (action1['actionType'][i + 3] == 9)):
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
            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7) and (action1['actionType'][i + 3] == 8)):
                count5678 = count5678 + 1
            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 6) and (action1['actionType'][i + 1] == 7) and (action1['actionType'][i + 2] == 8) and (action1['actionType'][i + 3] == 9)):
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
            if (((i + 4) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7) and (action1['actionType'][i + 3] == 8) and (action1['actionType'][i + 4] == 9)):
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
            if (((i + 4) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 6) and (action1['actionType'][i + 2] == 7) and (action1['actionType'][i + 3] == 8) and (action1['actionType'][i + 4] == 9)):
                count56789 = count56789 + 1
        userid.append(row.userid)
        action_56789_count_c.append(count56789)
    count['userid'] = userid
    count['action_56789_count_c'] = action_56789_count_c
    orderFuture = pd.merge(orderFuture, count[['userid', 'action_56789_count_c']], on='userid', how='left')
    return orderFuture

orderFuture_train = appear_56789_c(orderFuture_train, action_train)
orderFuture_test = appear_56789_c(orderFuture_test, action_test)


############# 4.time feature   #############
"""
# 1. 历史订单最近月份
# 2. 以最近的浏览记录作为要预测的用户订单时间

"""
# 历史订单最近月份
def history_mean_month(orderFuture, orderHistory):
    temp = pd.DatetimeIndex(orderHistory['order_time'])
    orderHistory['order_month'] = temp.month
    latest = orderHistory.groupby(['userid']).last().reset_index()
    orderFuture = pd.merge(orderFuture, latest[['userid', 'order_month']], on='userid', how='left') 
    return orderFuture

orderFuture_train = history_mean_month(orderFuture_train, orderHistory_train)
orderFuture_test = history_mean_month(orderFuture_test, orderHistory_test)
# """

# 以最近的浏览记录作为要预测的用户订单时间
def orderFuture_time(orderFuture, action):
    latest = action.groupby(['userid']).last().reset_index()
    latest.rename(columns={'actionTime': 'actionTime_future', 'action_time': 'action_time_future'}, inplace=True)
    print(latest)
    # 待确定是否需要加入actionType(已经加入)
    orderFuture = pd.merge(orderFuture, latest[['userid', 'actionTime_future', 'action_time_future']], on='userid', how='left')
    temp = pd.DatetimeIndex(orderFuture['action_time_future'])

    orderFuture['future_month'] = temp.month
    orderFuture['future_day'] = temp.day
    orderFuture['future_dayofweek'] = temp.dayofweek
    orderFuture['future_hour'] = temp.hour
    return orderFuture
# orderFuture_time(orderFuture_train, action_train)

orderFuture_train = orderFuture_time(orderFuture_train, action_train)
orderFuture_test = orderFuture_time(orderFuture_test, action_test)


print(orderFuture_train)
print(orderFuture_test)

print("开始提取：", start)
print("提取完成：", datetime.now())
orderFuture_train.to_csv(dire + 'train1.csv', index=False, encoding='utf-8')
orderFuture_test.to_csv(dire + 'test1.csv', index=False, encoding='utf-8')
