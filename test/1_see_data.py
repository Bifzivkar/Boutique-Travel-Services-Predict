# -*- encoding:utf-8 -*-
import pandas as pd
from sklearn import preprocessing

dire = '../../data/'
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
userProfile_train = pd.read_csv(dire + 'train/userProfile_train.csv', encoding='utf-8')
userProfile_test = pd.read_csv(dire + 'test/userProfile_test.csv', encoding='utf-8')

action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')


# count = orderHistory_train['continent'].value_counts()
# print(count)
# print(len(count))
# count = orderHistory_train['country'].value_counts()
# print(count)
# print(len(count))
# count = orderHistory_train[orderHistory_train['orderType'] == 1]['city'].value_counts()
# # print(count[:20])
# count = orderHistory_train[orderHistory_train['country'] == '中国']['city'].value_counts()
# print(count)
# print(len(count))
# count = userProfile_train['province'].value_counts()
# print(count)
# print(len(count))
count = action_train['userid'].value_counts()
print(count)
print(len(count))

print("==========")
# count = orderHistory_test[orderHistory_test['orderType'] == 1]['continent'].value_counts()
# # print(count)
# count = orderHistory_test[orderHistory_test['orderType'] == 1]['country'].value_counts()
# # print(count)
# count = orderHistory_test[orderHistory_test['country'] == '中国']['city'].value_counts()
# print(count[:20])
# print(len(count))
#
# count = userProfile_test['province'].value_counts()
# # print(count)
# # print(len(count))
count = action_test['userid'].value_counts()
print(count)
print(len(count))


