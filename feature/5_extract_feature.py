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
# """

############# 1.user feature   #############
"""
# 1. 用户地点划分1 2 3 线城市

"""
def province_123(userProfile, orderFuture):
    province_1 = ['上海', '北京', '广东']
    province_2 = ['福建', '重庆', '山东', '湖南', '陕西', '广西', '辽宁', '安徽', '河北', '重庆', '四川', '湖北', '江苏', '浙江', '天津']
    province_3 = ['云南', '黑龙江', '河南', '江西', '贵州', '山西', '内蒙古', '甘肃', '新疆', '海南', '宁夏', '青海', '西藏']
    userProfile['province_123'] = None
    userProfile['province_123'][userProfile['province'].isin(province_1)] = 1
    userProfile['province_123'][userProfile['province'].isin(province_2)] = 2
    userProfile['province_123'][userProfile['province'].isin(province_3)] = 3
    print(userProfile[['userid', 'province', 'province_123']])
    order = pd.merge(orderFuture, userProfile[['userid', 'province_123']], on='userid', how='left')
    return order

# # orderFuture_train = province_123(userProfile_train, orderFuture_train)
# # orderFuture_test = province_123(userProfile_test, orderFuture_test)

############# 2.history order feature   #############
"""
# 1.

"""
# 历史纪录中城市的精品占比
def history_type1_rate(orderFuture, orderHistory):
    all = len(orderHistory)
    print("all:", all)
    city_type1_rate = pd.DataFrame(columns=['city', 'city_rate'])
    country_type1_rate = pd.DataFrame(columns=['country', 'country_rate'])
    continent_type1_rate = pd.DataFrame(columns=['continent', 'continent_rate'])
    city1 = []
    country1 = []
    continent1 = []
    city_rate = []
    country_rate = []
    continent_rate = []

    city_list = list(set(list(orderHistory.city)))
    print(len(city_list))
    country_list = list(set(list(orderHistory.country)))
    continent_list = list(set(list(orderHistory.continent)))
    for city in city_list:
        city1.append(city)
        city_rate.append((len(orderHistory[orderHistory['city'] == city])/all)*(len(orderHistory[(orderHistory['city'] == city) & (orderHistory['orderType'] == 1)])/len(orderHistory[orderHistory['city'] == city])))
    for country in country_list:
        country1.append(country)
        country_rate.append((len(orderHistory[orderHistory['country'] == country])/all)*(len(orderHistory[(orderHistory['country'] == country) & (orderHistory['orderType'] == 1)])/len(orderHistory[orderHistory['country'] == country])))
    for continent in continent_list:
        continent1.append(continent)
        continent_rate.append((len(orderHistory[orderHistory['continent'] == continent])/all)*(len(orderHistory[(orderHistory['continent'] == continent) & (orderHistory['orderType'] == 1)])/len(orderHistory[orderHistory['continent'] == continent])))
    city_type1_rate['city'] = city1
    city_type1_rate['city_rate'] = city_rate
    country_type1_rate['country'] = country1
    country_type1_rate['country_rate'] = country_rate
    continent_type1_rate['continent'] = continent1
    continent_type1_rate['continent_rate'] = continent_rate
    orderHistory = pd.merge(orderHistory, city_type1_rate, on='city', how='left')
    orderHistory = pd.merge(orderHistory, country_type1_rate, on='country', how='left')
    orderHistory = pd.merge(orderHistory, continent_type1_rate, on='continent', how='left')
    orderHistory = orderHistory.groupby(orderHistory.userid)['city_rate', 'country_rate', 'continent_rate'].mean().reset_index()
    orderFuture = pd.merge(orderFuture, orderHistory[['userid', 'city_rate', 'country_rate', 'continent_rate']], on='userid', how='left')
    return orderFuture

# orderFuture = pd.concat([orderFuture_train,orderFuture_test])
# orderHistory = pd.concat([orderHistory_train,orderHistory_test])
# dataset = history_type1_rate(orderFuture, orderHistory)
# orderFuture_train = dataset[dataset.orderType.notnull()]
# orderFuture_test = dataset[dataset.orderType.isnull()]

############# 3.action feature   #############
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
    # 1
    latest = action.groupby(['userid']).last().reset_index()
    latest.rename(columns={'actionType_time': 'latest_1_time_interval'}, inplace=True)
    orderFuture = pd.merge(orderFuture, latest[['userid', 'latest_1_time_interval']], on='userid', how='left')
    # 2 3 4 5 6
    userid = []
    latest_2_time_interval = []
    latest_3_time_interval = []
    latest_4_time_interval = []
    latest_5_time_interval = []
    latest = action.groupby(['userid'])['actionTime'].idxmax().reset_index()
    latest.rename(columns={'actionTime': 'max_index'}, inplace=True)
    latest_2 = latest
    for index, row in latest.iterrows():
        userid.append(row.userid)
        # 2
        if (row.userid == action['userid'][row.max_index - 1]):
            latest_2_time_interval.append(action['actionType_time'][row.max_index - 1])
        else:
            latest_2_time_interval.append(None)
        # 3
        if (row.userid == action['userid'][row.max_index - 2]):
            latest_3_time_interval.append(action['actionType_time'][row.max_index - 2])
        else:
            latest_3_time_interval.append(None)
        # 4
        if (row.userid == action['userid'][row.max_index - 3]):
            latest_4_time_interval.append(action['actionType_time'][row.max_index - 3])
        else:
            latest_4_time_interval.append(None)
        # 5
        if (row.userid == action['userid'][row.max_index - 4]):
            latest_5_time_interval.append(action['actionType_time'][row.max_index - 4])
        else:
            latest_5_time_interval.append(None)

    latest_2['latest_2_time_interval'] = latest_2_time_interval
    latest_2['latest_3_time_interval'] = latest_3_time_interval
    latest_2['latest_4_time_interval'] = latest_4_time_interval
    latest_2['latest_5_time_interval'] = latest_5_time_interval
    orderFuture = pd.merge(orderFuture, latest_2[['userid', 'latest_2_time_interval', 'latest_3_time_interval',
                                        'latest_4_time_interval', 'latest_5_time_interval']], on='userid', how='left')

    # 均值
    latest = action.groupby(['userid'])['actionType_time'].mean().reset_index()
    latest.rename(columns={'actionType_time': 'actionType_time_mean'}, inplace=True)
    orderFuture = pd.merge(orderFuture, latest[['userid', 'actionType_time_mean']], on='userid', how='left')
    # 方差
    latest = action.groupby(['userid'])['actionType_time'].agg({'actionType_time_var':'var'}).reset_index()
    orderFuture = pd.merge(orderFuture, latest[['userid', 'actionType_time_var']], on='userid', how='left')

    # 最小值
    latest = action.groupby(['userid'])['actionType_time'].min().reset_index()
    latest.rename(columns={'actionType_time': 'actionType_time_min'}, inplace=True)
    orderFuture = pd.merge(orderFuture, latest[['userid', 'actionType_time_min']], on='userid', how='left')

    return orderFuture

orderFuture_train = time_interval(orderFuture_train, action_train)
orderFuture_test = time_interval(orderFuture_test, action_test)



# action 最后4 5 6 次操作时间的方差 和 均值
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



############# 4.time feature   #############
"""
# 1. 季节特征
 

"""
def season(orderFuture):
    orderFuture['season'] = 3
    orderFuture['season'][orderFuture['future_month'] <= 3] = 1
    orderFuture['season'][(orderFuture['future_month'] >= 4) & (orderFuture['future_month'] <= 6)] = 2
    orderFuture['season'][orderFuture['future_month'] >= 10] = 4
    return orderFuture
orderFuture_train = season(orderFuture_train)
orderFuture_test = season(orderFuture_test)


# print(orderFuture_train)
# print(orderFuture_test)

print("开始提取：", start)
print("提取完成：", datetime.now())
orderFuture_train.to_csv(dire + 'train3.csv', index=False, encoding='utf-8')
orderFuture_test.to_csv(dire + 'test3.csv', index=False, encoding='utf-8')