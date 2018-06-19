# -*- encoding:utf-8 -*-
import pandas as pd
from sklearn import preprocessing

dire = '../../data/'
# orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
# orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
# userProfile_train = pd.read_csv(dire + 'train/userProfile_train.csv', encoding='utf-8')
# userProfile_test = pd.read_csv(dire + 'test/userProfile_test.csv', encoding='utf-8')
#
# action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')
# action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')

# orderHistory_train.sort_values(['userid', 'order_time'], inplace=True)
# orderHistory_test.sort_values(['userid', 'order_time'], inplace=True)
#
# action_train.sort_values(['userid', 'action_time'], inplace=True)
# action_test.sort_values(['userid', 'action_time'], inplace=True)
#
#
# orderHistory_train.to_csv(dire + 'train/orderHistory_train.csv', index=False, encoding='utf-8')
# orderHistory_test.to_csv(dire + 'test/orderHistory_test.csv', index=False, encoding='utf-8')
# action_train.to_csv(dire + 'train/action_train.csv', index=False, encoding='utf-8')
# action_test.to_csv(dire + 'test/action_test.csv', index=False, encoding='utf-8')


train = pd.read_csv(dire + 'train2.csv', encoding='utf-8')
test = pd.read_csv(dire + 'test2.csv', encoding='utf-8')

def label_56(orderFuture):
    orderFuture['time_6_c'] = orderFuture['time_6_c']/orderFuture['action_6_c']
    return orderFuture

train = label_56(train)
test = label_56(test)

train.to_csv(dire + 'train2.csv', index=False, encoding='utf-8')
test.to_csv(dire + 'test2.csv', index=False, encoding='utf-8')