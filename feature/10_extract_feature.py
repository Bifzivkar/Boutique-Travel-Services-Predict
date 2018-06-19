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
# 1. 

"""
#

############# 2.user feature   #############
"""
# 1. 

"""

############# 3.history order feature   #############
"""
# 1.
# 2.
"""

############# 3.action feature   #############
"""
# 1. 
# 2. 
# 3. 
# """


############# 4.comment feature   #############
"""
# 1. 
"""





# print(orderFuture_train)
# print(orderFuture_test)



print("开始提取：", start)
print("提取完成：", datetime.datetime.now())
orderFuture_train.to_csv(dire + 'train6.csv', index=False, encoding='utf-8')
orderFuture_test.to_csv(dire + 'test6.csv', index=False, encoding='utf-8')