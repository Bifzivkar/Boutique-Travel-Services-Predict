# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
# 提交处理
dire = '../../data/'
prob1 = pd.read_csv(dire + 'submit/cat/sub_cat_prob78_26.csv', encoding='utf-8')
# prob2 = pd.read_csv(dire + 'submit/cat/sub_cat_prob58.csv', encoding='utf-8')
# prob3 = pd.read_csv(dire + 'submit/cat/sub_cat_prob65_3.csv', encoding='utf-8')
# prob1.rename(columns={'orderType': 'orderType_1'}, inplace=True)
# prob2.rename(columns={'orderType': 'orderType_2'}, inplace=True)
# prob3.rename(columns={'orderType': 'orderType_3'}, inplace=True)
# prob1 = pd.merge(prob1, prob2, on='userid', how='le

m = np.array(prob1.sort_values(by='orderType').index)
prob1['rank'] = m
prob1['new'] = prob1['rank'] / 10075
print(prob1)
prob1['new'][prob1['new'] > 0.5] = 1
prob1['new'][~(prob1['new'] > 0.5)] = 0
print("1的数目：", len(prob1[prob1['new'] == 1]))
# for index, row in prob1.iterrows():
#     if((row.orderType_1 > 0.5) and (row.orderType_2 > 0.5) and (row.orderType_3 > 0.5)):
#         # print(np.max([row.orderType_1, ft')
# prob1 = pd.merge(prob1, prob3, on='userid', how='left')
# # print(prob1)
# orderType = []row.orderType_2, row.orderType_3]))
#         orderType.append(np.max([row.orderType_1, row.orderType_2, row.orderType_3]))
#
#     if ((row.orderType_1 < 0.5) and (row.orderType_2 < 0.5) and (row.orderType_3 < 0.5)):
#         # print(np.min([row.orderType_1, row.orderType_2, row.orderType_3]))
#         orderType.append(np.min([row.orderType_1, row.orderType_2, row.orderType_3]))
#
#     if ((row.orderType_1 > 0.5) and (row.orderType_2 < 0.5) and (row.orderType_3 < 0.5)):
#         print(index)
#         if (abs(row.orderType_2 - 0.5) > abs(row.orderType_3 - 0.5)):
#             orderType.append(row.orderType_2)
#         else:
#             orderType.append(row.orderType_3)
#
#     if ((row.orderType_1 < 0.5) and (row.orderType_2 > 0.5) and (row.orderType_3 < 0.5)):
#         print(index)
#         if (abs(row.orderType_1 - 0.5) > abs(row.orderType_3 - 0.5)):
#             orderType.append(row.orderType_1)
#         else:
#             orderType.append(row.orderType_3)
#
#     if ((row.orderType_1 < 0.5) and (row.orderType_2 < 0.5) and (row.orderType_3 > 0.5)):
#         print(index)
#         if (abs(row.orderType_1 - 0.5) > abs(row.orderType_2 - 0.5)):
#             orderType.append(row.orderType_1)
#         else:
#             orderType.append(row.orderType_2)
#
#     if ((row.orderType_1 < 0.5) and (row.orderType_2 > 0.5) and (row.orderType_3 > 0.5)):
#         print(index)
#         if (abs(row.orderType_2 - 0.5) > abs(row.orderType_3 - 0.5)):
#             orderType.append(row.orderType_2)
#         else:
#             orderType.append(row.orderType_3)
#
#     if ((row.orderType_1 > 0.5) and (row.orderType_2 < 0.5) and (row.orderType_3 > 0.5)):
#         print(index)
#         if (abs(row.orderType_1 - 0.5) > abs(row.orderType_3 - 0.5)):
#             orderType.append(row.orderType_1)
#         else:
#             orderType.append(row.orderType_3)
#
#     if ((row.orderType_1 > 0.5) and (row.orderType_2 > 0.5) and (row.orderType_3 < 0.5)):
#         print(index)
#         if (abs(row.orderType_1 - 0.5) > abs(row.orderType_2 - 0.5)):
#             orderType.append(row.orderType_1)
#         else:
#             orderType.append(row.orderType_2)
# prob1['orderType'] = orderType
# prob1.drop(['orderType_1', 'orderType_2', 'orderType_3'], axis=1, inplace=True)
# prob1.to_csv(dire + 'submit/fusion/fusion_cat57_cat58_cat65_3_8.csv', index=False)
# #
# prob1['orderType'][prob1['orderType'] > 0.5] = 1
# prob1['orderType'][~(prob1['orderType'] > 0.5)] = 0
# print("1的数目：", len(prob1[prob1['orderType'] == 1]))

