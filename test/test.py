# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime


dire = '../../data/'
start = datetime.now()
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderFuture_train = pd.read_csv(dire + 'train/orderFuture_train.csv', encoding='utf-8')
userComment_train = pd.read_csv(dire + 'train/userComment_train1.csv', encoding='utf-8')
action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')
train = pd.read_csv(dire + 'train3.csv', encoding='utf-8')

orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
orderFuture_test = pd.read_csv(dire + 'test/orderFuture_test.csv', encoding='utf-8')
userComment_test = pd.read_csv(dire + 'test/userComment_test1.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')
test = pd.read_csv(dire + 'test3.csv', encoding='utf-8')


# 用户评论表的标签数
def comment_tags_count(orderFuture, userComment):
    orderFuture.rename(columns={'orderType': 'orderType_future'}, inplace=True)
    userComment = pd.merge(userComment, orderFuture, on='userid', how='left')
    print(userComment[userComment['orderType'] == 1]['tags'].value_counts()[:10])
    print(len(userComment[userComment['orderType'] == 1]))
    print(len(userComment[userComment['orderType'] == 1]['tags'].value_counts()))

    print(userComment[userComment['orderType'] == 0]['tags'].value_counts()[:10])
    print(len(userComment[userComment['orderType'] == 0]))
    print(len(userComment[userComment['orderType'] == 0]['tags'].value_counts()))
    count0 = 0
    count1 = 0
    for index, row in userComment.iterrows():
        if '举牌迎接' in str(row.tags).split('|'):
            if ((len(str(row.tags).split('|')) < 5) and (len(str(row.commentsKeyWords).split(',')) < 0)):
                if (row.orderType_future == 0):
                    print(row.userid)
                    count0 = count0 + 1
                else:
                    count1 = count1 + 1

    print("0:", count0)
    print("1:", count1)
    return userComment

orderFuture_train = comment_tags_count(orderFuture_train, userComment_train)
# orderFuture_test = comment_tags_count(orderFuture_test, userComment_test)
columns = ['userid', 'orderid', 'rating', 'orderType', 'tags', 'commentsKeyWords']

# userComment_train.to_csv(dire + 'train/userComment_train1.csv', index=False, encoding='utf-8', columns=columns)
# userComment_test.to_csv(dire + 'test/userComment_test1.csv', index=False, encoding='utf-8', columns=columns)



# 删除掉没用的特征
# columns = ['latest_3_actionTime_mean', 'latest_4_actionTime_mean', 'latest_5_actionTime_mean',
#            'min_distance_1', 'min_distance_2', 'min_distance_3', 'min_distance_4',]
# train.drop(columns, axis=1, inplace=True)
# test.drop(columns, axis=1, inplace=True)
# train.to_csv(dire + 'train33.csv', index=False, encoding='utf-8')
# test.to_csv(dire + 'test33.csv', index=False, encoding='utf-8')

# columns = train.columns
# columns_t = test.columns
# train.drop(columns[2:111], axis=1,inplace=True)
# test.drop(columns_t[1:110], axis=1,inplace=True)
# print(train)
# print(test)
# train.to_csv(dire + 'train/train33.csv', index=False, encoding='utf-8')
# test.to_csv(dire + 'test/test33.csv', index=False, encoding='utf-8')


# # 单次action操作的时间
# def action_time(action):
#
#     userid = []
#     actionType_time = []
#     temp = pd.DatetimeIndex(action['action_time'])
#     action['action_date'] = temp.date
#     last_userid = 0
#     last_actionTime = 0
#     for index, row in action.iterrows():
#         print(index)
#         userid.append(row.userid)
#         if(row.userid != last_userid):
#             actionType_time.append(None)
#         else:
#             actionType_time.append(row.actionTime-last_actionTime)
#         last_userid = row.userid
#         last_actionTime = row.actionTime
#     action['actionType_time'] = actionType_time
#     print(action)
#     return action
#
# action_train = action_time(action_train)
# action_test = action_time(action_test)
#
# print(action_train)
# print(action_test)
#
#
# action_train.to_csv(dire + 'train/action_train1.csv', index=False, encoding='utf-8')
# action_test.to_csv(dire + 'test/action_test1.csv', index=False, encoding='utf-8')


