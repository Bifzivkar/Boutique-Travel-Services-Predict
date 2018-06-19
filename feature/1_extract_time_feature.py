# -*- encoding:utf-8 -*-
# ===================================================================== #
# 时间处理
# 从时间戳提取出，日，小时，dayofweek
# ===================================================================== #
import time
import pandas as pd
dire = '../../data/'
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')

# 时间戳转时间
def timestamp_to_time(timestamps):
    times = []
    for timestamp in timestamps:
        times.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)))
    return times

orderHistory_train['order_time'] = timestamp_to_time(orderHistory_train['orderTime'])
orderHistory_test['order_time'] = timestamp_to_time(orderHistory_test['orderTime'])
action_train['action_time'] = timestamp_to_time(action_train['actionTime'])
action_test['action_time'] = timestamp_to_time(action_test['actionTime'])


orderHistory_train.to_csv(dire + 'train/orderHistory_train.csv', index=False, encoding='utf-8')
orderHistory_test.to_csv(dire + 'test/orderHistory_test.csv', index=False, encoding='utf-8')
action_train.to_csv(dire + 'train/action_train.csv', index=False, encoding='utf-8')
action_test.to_csv(dire + 'test/action_test.csv', index=False, encoding='utf-8')
