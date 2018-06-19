# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime

# from datetime import datetime

dire = '../../data/'
start = datetime.datetime.now()
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderFuture_train = pd.read_csv(dire + 'train/orderFuture_train7.csv', encoding='utf-8')
userProfile_train = pd.read_csv(dire + 'train/userProfile_train.csv', encoding='utf-8')
userComment_train = pd.read_csv(dire + 'train/userComment_train1.csv', encoding='utf-8')
action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')
city = pd.read_csv(dire + 'train/city.csv', encoding='utf-8')

orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
orderFuture_test = pd.read_csv(dire + 'test/orderFuture_test7.csv', encoding='utf-8')
userProfile_test = pd.read_csv(dire + 'test/userProfile_test.csv', encoding='utf-8')
userComment_test = pd.read_csv(dire + 'test/userComment_test1.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')
# """


############# 1.time feature   #############
"""
# 1. 判断是否节假日

"""
# def is_holiday():
#     pass


# 是否周末
orderFuture_train['future_is_weekend'] = orderFuture_train['future_dayofweek']
orderFuture_train['future_is_weekend'][orderFuture_train['future_is_weekend'] < 5] = 0
orderFuture_train['future_is_weekend'][orderFuture_train['future_is_weekend'] > 4] = 1

orderFuture_test['future_is_weekend'] = orderFuture_test['future_dayofweek']
orderFuture_test['future_is_weekend'][orderFuture_test['future_is_weekend'] < 5] = 0
orderFuture_test['future_is_weekend'][orderFuture_test['future_is_weekend'] > 4] = 1

############# 2.user feature   #############
"""
# 1. 个人信息完善度

"""


def person_info_count(orderFuture):
    orderFuture['user_info_count'] = orderFuture['gender_e']
    orderFuture['user_info_count'][
        orderFuture.gender_e.isnull() & orderFuture.age_e.isnull() & orderFuture.province_e.isnull()] = 0
    orderFuture['user_info_count'][
        orderFuture.gender_e.notnull() & orderFuture.age_e.isnull() & orderFuture.province_e.isnull()] = 1
    orderFuture['user_info_count'][
        orderFuture.gender_e.isnull() & orderFuture.age_e.notnull() & orderFuture.province_e.isnull()] = 1
    orderFuture['user_info_count'][
        orderFuture.gender_e.isnull() & orderFuture.age_e.isnull() & orderFuture.province_e.notnull()] = 1
    orderFuture['user_info_count'][
        orderFuture.gender_e.isnull() & orderFuture.age_e.notnull() & orderFuture.province_e.notnull()] = 2
    orderFuture['user_info_count'][
        orderFuture.gender_e.notnull() & orderFuture.age_e.isnull() & orderFuture.province_e.notnull()] = 2
    orderFuture['user_info_count'][
        orderFuture.gender_e.notnull() & orderFuture.age_e.notnull() & orderFuture.province_e.isnull()] = 2
    orderFuture['user_info_count'][
        orderFuture.gender_e.notnull() & orderFuture.age_e.notnull() & orderFuture.province_e.notnull()] = 3
    return orderFuture


orderFuture_train = person_info_count(orderFuture_train)
orderFuture_test = person_info_count(orderFuture_test)

############# 3.history order feature   #############
"""
# 1.历史出现0的次数, 和0所占比例(修正)
# 2.地点按月份下单的热度
"""


# 历史出现0的次数, 和0所占比例(修正)
def count_0(orderFuture):
    count_0 = []
    type_0_rate = []
    for index, row in orderFuture.iterrows():
        if (row.count_all >= 0):
            count_0.append(row.count_all - row.count_1)
            type_0_rate.append((row.count_all - row.count_1) / row.count_all)
        else:
            count_0.append(None)
            type_0_rate.append(None)
    orderFuture['count_0'] = count_0
    orderFuture['type_0_rate'] = type_0_rate
    return orderFuture


orderFuture_train = count_0(orderFuture_train)
orderFuture_test = count_0(orderFuture_test)


# 城市司机数 和 精品线路数平均
def avg_city_info(orderHistory, orderFuture, city):
    # city
    city['type1_rate'] = city['type1_count']/np.sum(city['type1_count'])
    city['driver_rate'] = city['driver_count']/np.sum(city['type1_count'])
    city['count_rate'] = city['count']/np.sum(city['type1_count'])
    orderHistory_city = pd.merge(orderHistory, city, on='city', how='left')
    avg_city = orderHistory_city.groupby(['userid']).mean().reset_index()
    avg_city.rename(
        columns={'count': 'city_count_avg', 'driver_count': 'driver_count_avg', 'type1_count': 'type1_count_avg',
                 'type1_rate': 'type1_rate_avg', 'driver_rate': 'driver_rate_avg', 'count_rate': 'count_rate_avg'
                 }, inplace=True)
    orderFuture = pd.merge(orderFuture, avg_city[['userid', 'city_count_avg', 'driver_count_avg', 'type1_count_avg']],
                           on='userid', how='left')
    # country
    country = city.groupby(['country']).mean().reset_index()
    orderHistory_country = pd.merge(orderHistory, country, on='country', how='left')
    avg_country = orderHistory_country.groupby(['userid']).mean().reset_index()
    avg_country.rename(columns={'count': 'country_count_avg', 'driver_count': 'country_driver_count_avg',
                                'type1_count': 'country_type1_count_avg'}, inplace=True)
    orderFuture = pd.merge(orderFuture, avg_country[
        ['userid', 'country_count_avg', 'country_driver_count_avg', 'country_type1_count_avg']], on='userid',
                           how='left')
    # continent
    continent = city.groupby(['continent']).mean().reset_index()
    orderHistory_continent = pd.merge(orderHistory, continent, on='continent', how='left')
    avg_continent = orderHistory_continent.groupby(['userid']).mean().reset_index()
    avg_continent.rename(columns={'count': 'continent_count_avg', 'driver_count': 'continent_driver_count_avg',
                                  'type1_count': 'continent_type1_count_avg'}, inplace=True)
    orderFuture = pd.merge(orderFuture, avg_continent[
        ['userid', 'continent_count_avg', 'continent_driver_count_avg', 'continent_type1_count_avg']], on='userid',
                           how='left')
    return orderFuture

orderFuture_train = avg_city_info(orderHistory_train, orderFuture_train, city)
orderFuture_test = avg_city_info(orderHistory_test, orderFuture_test, city)


# 城市选1 的平均和，比率
def city_type1_rate(orderHistory, orderFuture, city):
    city['city_type1_rate'] = city['city_type1_count'] / city['count']
    city['driver_count_rate'] = city['driver_count'] / city['count']
    city['type1_count_rate'] = city['type1_count'] / city['count']
    city['type1_driver_rate'] = city['type1_count'] / city['driver_count']
    city['type1_city_type1_count_rate'] = city['type1_count'] / city['city_type1_count']
    orderHistory_city = pd.merge(orderHistory, city[
        ['city', 'city_type1_count', 'city_type1_rate', 'driver_count_rate', 'type1_count_rate', 'type1_driver_rate',
         'type1_city_type1_count_rate']], on='city', how='left')
    avg_city_rate = orderHistory_city.groupby(['userid']).mean().reset_index()
    orderFuture = pd.merge(orderFuture, avg_city_rate[
        ['userid', 'city_type1_count', 'city_type1_rate', 'driver_count_rate', 'type1_count_rate', 'type1_driver_rate',
         'type1_city_type1_count_rate']], on='userid', how='left')
    return orderFuture


orderFuture_train = city_type1_rate(orderHistory_train, orderFuture_train, city)
orderFuture_test = city_type1_rate(orderHistory_test, orderFuture_test, city)


# 历史订单是否出现过0
def ever_0(orderFuture):
    ever_0 = []
    for index, row in orderFuture.iterrows():
        if (row.count_0 > 0):
            ever_0.append(1)
        else:
            ever_0.append(0)
    return ever_0

orderFuture_train['ever_0'] = ever_0(orderFuture_train)
orderFuture_test['ever_0'] = ever_0(orderFuture_test)


# 历史订单是否出现过0 1 修正般 1 0 None
def ever_0_f(orderFuture):
    ever_0_f = []
    ever_1_f = []
    for index, row in orderFuture.iterrows():
        if (row.count_0 > 0):
            ever_0_f.append(1)
        else:
            if (row.count_0 == 0):
                ever_0_f.append(0)
            else:
                ever_0_f.append(None)

        if (row.count_1 > 0):
            ever_1_f.append(1)
        else:
            if (row.count_1 == 0):
                ever_1_f.append(0)
            else:
                ever_1_f.append(None)
    orderFuture['ever_0_f'] = ever_0_f
    orderFuture['ever_1_f'] = ever_1_f
    return orderFuture

orderFuture_train = ever_0_f(orderFuture_train)
orderFuture_test = ever_0_f(orderFuture_test)


# 城市和月份组合
def city_month(orderHistory, orderFuture, city_s):
    print(city_s)
    temp = pd.DatetimeIndex(orderHistory['order_time'])
    orderHistory['order_month'] = temp.month
    userid = []
    city_1 = []
    city_2 = []
    city_3 = []
    city_4 = []
    city_5 = []
    city_6 = []
    city_7 = []
    city_8 = []
    city_9 = []
    city_10 = []
    city_11 = []
    city_12 = []
    for index, row in orderFuture.iterrows():
        userid.append(row.userid)
        order_user = orderHistory[(orderHistory['userid']==row.userid) & (orderHistory['city']==city_s)]
        len_1 = len(order_user[(order_user['order_month'] == 1)])
        len_2 = len(order_user[(order_user['order_month'] == 2)])
        len_3 = len(order_user[(order_user['order_month'] == 3)])
        len_4 = len(order_user[(order_user['order_month'] == 4)])
        len_5 = len(order_user[(order_user['order_month'] == 5)])
        len_6 = len(order_user[(order_user['order_month'] == 6)])
        len_7 = len(order_user[(order_user['order_month'] == 7)])
        len_8 = len(order_user[(order_user['order_month'] == 8)])
        len_9 = len(order_user[(order_user['order_month'] == 9)])
        len_10 = len(order_user[(order_user['order_month'] == 10)])
        len_11 = len(order_user[(order_user['order_month'] == 11)])
        len_12 = len(order_user[(order_user['order_month'] == 12)])
        if(len(order_user)>0):
            if(len_1 > 0):
                city_1.append(len_1)
            else:
                city_1.append(None)

            if(len_2 > 0):
                city_2.append(len_2)
            else:
                city_2.append(None)

            if(len_3 > 0):
                city_3.append(len_3)
            else:
                city_3.append(None)

            if(len_4 > 0):
                city_4.append(len_4)
            else:
                city_4.append(None)

            if(len_5>0):
                city_5.append(len_5)
            else:
                city_5.append(None)

            if(len_6>0):
                city_6.append(len_6)
            else:
                city_6.append(None)

            if(len_7>0):
                city_7.append(len_7)
            else:
                city_7.append(None)

            if(len_8>0):
                city_8.append(len_8)
            else:
                city_8.append(None)

            if(len_9>0):
                city_9.append(len_9)
            else:
                city_9.append(None)

            if(len_10>0):
                city_10.append(len_10)
            else:
                city_10.append(None)

            if(len_11>0):
                city_11.append(len_11)
            else:
                city_11.append(None)

            if(len_12>0):
                city_12.append(len_12)
            else:
                city_12.append(None)
        else:
            city_1.append(None)
            city_2.append(None)
            city_3.append(None)
            city_4.append(None)
            city_5.append(None)
            city_6.append(None)
            city_7.append(None)
            city_8.append(None)
            city_9.append(None)
            city_10.append(None)
            city_11.append(None)
            city_12.append(None)
    orderFuture['city'+city_s+'_1_count'] = city_1
    orderFuture['city'+city_s+'_2_count'] = city_2
    orderFuture['city'+city_s+'_3_count'] = city_3
    orderFuture['city'+city_s+'_4_count'] = city_4
    orderFuture['city'+city_s+'_5_count'] = city_5
    orderFuture['city'+city_s+'_6_count'] = city_6
    orderFuture['city'+city_s+'_7_count'] = city_7
    orderFuture['city'+city_s+'_8_count'] = city_8
    orderFuture['city'+city_s+'_9_count'] = city_9
    orderFuture['city'+city_s+'_10_count'] = city_10
    orderFuture['city'+city_s+'_11_count'] = city_11
    orderFuture['city'+city_s+'_12_count'] = city_12
    return orderFuture

# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '东京')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '东京')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '巴黎')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '巴黎')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '北海道--札幌')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '北海道--札幌')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '大阪')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '大阪')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '首尔')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '首尔')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '台北')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '台北')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '墨尔本')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '墨尔本')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '米兰')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '米兰')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '纽约')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '纽约')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '吉隆坡')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '吉隆坡')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '曼谷')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '曼谷')
# orderFuture_train = city_month(orderHistory_train, orderFuture_train, '新加坡')
# orderFuture_test = city_month(orderHistory_test, orderFuture_test, '新加坡')


# 修正count_all

# 历史订单数、为1的数量和占比
def count_all(orderFuture, orderHistory, userComment):
    train = pd.concat([orderHistory[['userid', 'orderid']], userComment[['userid', 'orderid']]])
    dataset = train.drop_duplicates()

    count_all = dataset.groupby(dataset.userid)['orderid'].count().reset_index()  # 每个用户订单总数
    count_all.rename(columns={'orderid': 'count_all_fix'}, inplace=True)
    orderFuture = pd.merge(orderFuture, count_all, on='userid', how='left')
    return orderFuture

orderFuture_train = count_all(orderFuture_train, orderHistory_train, userComment_train)
orderFuture_test = count_all(orderFuture_test, orderHistory_test, userComment_test)




############# 3.action feature   #############
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


# 最后三个是否为连续567 或者 56
def continuous_55_count(orderFuture, action):
    count = pd.DataFrame(columns=['userid', 'last_is_1567', 'last_is_56', 'last_is_156'])
    userid = []
    last_is_1567 = []
    last_is_56 = []
    last_is_156 = []
    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    for index, row in latest.iterrows():
        userid.append(row.userid)
        action1 = action[action['userid'] == row.userid]
        # 1567
        if ((4 <= len(action1)) and (action1['actionType'][row.max_index-3] == 1) and (action1['actionType'][row.max_index-2] == 5)
            and (action1['actionType'][row.max_index-1] == 6) and (action1['actionType'][row.max_index] == 7)
            and (action1['actionType_time'][row.max_index - 2] < 1800) and (action1['actionType_time'][row.max_index-1] < 1800)
            and (action1['actionType_time'][row.max_index] < 1800)):
            last_is_1567.append(1)
        else:
            last_is_1567.append(0)
        # 56
        if ((2 <= len(action1)) and (action1['actionType'][row.max_index-1] == 5) and (action1['actionType'][row.max_index] == 6)
            and (action1['actionType_time'][row.max_index] < 1800)):
            last_is_56.append(1)
        else:
            last_is_56.append(0)
        # 156
        if ((3 <= len(action1)) and (action1['actionType'][row.max_index-2] == 1) and (action1['actionType'][row.max_index-1] == 5)
            and (action1['actionType'][row.max_index] == 6) and (action1['actionType_time'][row.max_index-1] < 1800)
            and (action1['actionType_time'][row.max_index] < 1800)):
            last_is_156.append(1)
        else:
            last_is_156.append(0)
    count['userid'] = userid
    count['last_is_1567'] = last_is_1567
    count['last_is_56'] = last_is_56
    count['last_is_156'] = last_is_156
    orderFuture = pd.merge(orderFuture, count[['userid', 'last_is_1567', 'last_is_56', 'last_is_156']], on='userid', how='left')
    return orderFuture

# orderFuture_train = continuous_55_count(orderFuture_train, action_train)
# orderFuture_test = continuous_55_count(orderFuture_test, action_test)


# 对应连续55、15、156、1556操作的次数
def continuous_55_count_c(orderFuture, action):
    count = pd.DataFrame(
        columns=['userid', 'continuou_15_count_c', 'continuou_55_count_c', 'continuou_156_count_c', 'continuou_1556_count_c'])
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
                count15 = count15 + 1

            if (((i + 1) < len(action1)) and (action1['actionType'][i] == 5) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType_time'][i + 1] < 1800)):
                count55 = count55 + 1

            if (((i + 2) < len(action1)) and (action1['actionType'][i] == 1) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType'][i + 2] == 6) and (action1['actionType_time'][i + 1] < 1800)
                and (action1['actionType_time'][i + 2] < 1800)):
                count156 = count156 + 1

            if (((i + 3) < len(action1)) and (action1['actionType'][i] == 1) and (action1['actionType'][i + 1] == 5)
                and (action1['actionType'][i + 2] == 5) and (action1['actionType'][i + 3] == 6)
                and (action1['actionType_time'][i + 1] < 1800) and (action1['actionType_time'][i + 2] < 1800)
                and (action1['actionType_time'][i + 3] < 1800)):
                count1556 = count1556 + 1

        userid.append(row.userid)
        continuou_15_count.append(count15)
        continuou_55_count.append(count55)
        action_156_count.append(count156)
        action_1556_count.append(count1556)
    count['userid'] = userid
    count['continuou_15_count_c'] = continuou_15_count
    count['continuou_55_count_c'] = continuou_55_count
    count['continuou_156_count_c'] = action_156_count
    count['continuou_1556_count_c'] = action_1556_count

    orderFuture = pd.merge(orderFuture, count[
        ['userid', 'continuou_15_count_c', 'continuou_55_count_c', 'continuou_156_count_c', 'continuou_1556_count_c']],
                           on='userid', how='left')
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

city['type1_rate'] = city['type1_count'] / np.sum(city['type1_count'])
city['driver_rate'] = city['driver_count'] / np.sum(city['driver_count'])
city['count_rate'] = city['count'] / np.sum(city['count'])
# 关联用户要去city相关概率
def print_othner(orderFuture, orderHistory, action, city):
    # 相隔小于1天
    action_his = action[action.orderid.notnull()]
    action_last = action.groupby(['userid'])['actionTime'].last().reset_index()
    action_his_last = action_his.groupby(['userid'])['actionTime'].last().reset_index()
    action_his_last.rename(columns={'actionTime': 'his_actionTime'}, inplace=True)
    action_last = pd.merge(action_last, action_his_last, on='userid', how='left')
    action_last['diff'] = action_last['actionTime'] - action_last['his_actionTime']
    action_last = action_last[action_last['diff'] <= 86400]
    # 历史均为0
    action_last = pd.merge(action_last[['userid']], action, on='userid', how='left')
    action_type_sum = action_last.groupby(['userid'])['orderType'].sum().reset_index()
    action_type_sum = action_type_sum[action_type_sum['orderType'] == 0]

    # 的历史最后一单
    history_last = orderHistory.groupby(['userid'])['orderTime', 'city'].last().reset_index()
    history_last = pd.merge(action_type_sum[['userid']], history_last, on='userid', how='left')

    # city表

    city.rename(columns={'count': 'city_count'}, inplace=True)
    history_last = pd.merge(history_last, city[['city', 'city_count', 'driver_count', 'type1_count',
                                  'type1_rate', 'driver_rate', 'count_rate']], on='city', how='left')
    orderFuture = pd.merge(orderFuture, history_last[['userid', 'city_count', 'driver_count', 'type1_count',
                                  'type1_rate', 'driver_rate', 'count_rate']], on='userid', how='left')

    # orderFuture['city_count'][orderFuture['city_count'].isnull()] = -1
    # orderFuture['driver_count'][orderFuture['driver_count'].isnull()] = -1
    # orderFuture['type1_count'][orderFuture['type1_count'].isnull()] = -1
    #
    # orderFuture['type1_rate'][orderFuture['type1_rate'].isnull()] = -1
    # orderFuture['type1_rate'][(orderFuture['type1_rate'].isnull()) & (orderFuture['count_all'] > 0)] = 0
    # orderFuture['driver_rate'][orderFuture['driver_rate'].isnull()] = -1
    # orderFuture['driver_rate'][(orderFuture['driver_rate'].isnull()) & (orderFuture['count_all'] > 0)] = 0
    # orderFuture['count_rate'][orderFuture['count_rate'].isnull()] = -1
    # orderFuture['count_rate'][(orderFuture['count_rate'].isnull()) & (orderFuture['count_all'] > 0)] = 0
    return orderFuture


# orderFuture_train = print_othner(orderFuture_train, orderHistory_train, action_train, city)
# orderFuture_test = print_othner(orderFuture_test, orderHistory_test, action_test, city)

############# 4.comment feature   #############
"""
# 1. 


"""
# 用户各评论类型1的比率
def comment_tags_1_rate(orderFuture, userComment):
    tags_count = userComment.groupby(['tags'])['userid'].count().reset_index()
    tags_count.rename(columns={'userid': 'tags_appear_count'}, inplace=True)
    tags_count['tag_appear_rate'] = tags_count['tags_appear_count']/(len(userComment[userComment['orderType'] == 1]))
    userComment = pd.merge(userComment, tags_count, on='tags', how='left')
    orderFuture = pd.merge(orderFuture, userComment[['userid', 'tags_appear_count', 'tag_appear_rate']], on='userid', how='left')
    return orderFuture

orderFuture_train = comment_tags_1_rate(orderFuture_train, userComment_train)
orderFuture_test = comment_tags_1_rate(orderFuture_test, userComment_test)




# print(orderFuture_train)
# print(orderFuture_test)



print("开始提取：", start)
print("提取完成：", datetime.datetime.now())
orderFuture_train.to_csv(dire + 'train6.csv', index=False, encoding='utf-8')
orderFuture_test.to_csv(dire + 'test6.csv', index=False, encoding='utf-8')