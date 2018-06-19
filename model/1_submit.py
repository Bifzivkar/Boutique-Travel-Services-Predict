# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
# 提交处理
# rule_data1.fft+ever_1
# rule_data2.规则2+fft+ever_1
# rule_data3.规则4+规则2+fft+ever_1
# rule_data4.规则4
# rule_data5.规则5
dire = '../../data/'
train = pd.read_csv(dire + 'train5.csv', encoding='utf-8')
test = pd.read_csv(dire + 'test5.csv', encoding='utf-8')
rule_data1 = pd.read_csv(dire + 'backup/LOVECT/rule_data1.csv', encoding='utf-8')
rule_data2 = pd.read_csv(dire + 'backup/LOVECT/rule_data2.csv', encoding='utf-8')
rule_data3 = pd.read_csv(dire + 'backup/LOVECT/rule_data3.csv', encoding='utf-8')
rule_data4 = pd.read_csv(dire + 'backup/LOVECT/rule_data4.csv', encoding='utf-8')
rule_data5 = pd.read_csv(dire + 'backup/LOVECT/rule_data5.csv', encoding='utf-8')
rule_data6 = pd.read_csv(dire + 'backup/LOVECT/rule_data6.csv', encoding='utf-8')
rule_data7 = pd.read_csv(dire + 'backup/LOVECT/rule_data7.csv', encoding='utf-8')
rule_data8 = pd.read_csv(dire + 'backup/LOVECT/rule_data8.csv', encoding='utf-8')
rule_data_all = pd.read_csv(dire + 'backup/LOVECT/rule_data_all.csv', encoding='utf-8')
# prob1 分数更高
prob1 = pd.read_csv(dire + 'submit/fusion/sub_stacking_prob20180127_221155_cat_prob78_36_no_train56789_34.csv', encoding='utf-8')
prob2 = pd.read_csv(dire + 'submit/stacking/sub_stacking_prob20180124_015740.csv', encoding='utf-8')
prob1.rename(columns={'orderType': 'orderType_1'}, inplace=True)
prob2.rename(columns={'orderType': 'orderType_2'}, inplace=True)
rule_data1.rename(columns={'orderType': 'rule_data1'}, inplace=True)
rule_data2.rename(columns={'orderType': 'rule_data2'}, inplace=True)
rule_data3.rename(columns={'orderType': 'rule_data'}, inplace=True)
rule_data4.rename(columns={'orderType': 'rule_data4'}, inplace=True)
rule_data5.rename(columns={'orderType': 'rule_data'}, inplace=True)
rule_data6.rename(columns={'orderType': 'rule_data6'}, inplace=True)
rule_data7.rename(columns={'orderType': 'rule_data'}, inplace=True)
rule_data8.rename(columns={'orderType': 'rule_data'}, inplace=True)
prob = pd.merge(prob1, prob2, on='userid', how='left')
# prob = pd.merge(prob1, rule_data_all, on='userid', how='left')
# prob = pd.merge(prob, test[['userid', 'ever_1_f']], on='userid', how='left')

# rule_data2 = pd.merge(rule_data2, rule_data1, on='userid', how='left')
# rule_data2['rule_data'] = None
# rule_data2['rule_data'][(rule_data2['rule_data1'].isnull()) & (rule_data2['rule_data2'] == 1)] = 1
# rule_data2['rule_data'][(rule_data2['rule_data1'].isnull()) & (rule_data2['rule_data2'] == 0)] = 0


# rule_data3 = pd.concat([rule_data3, rule_data5])
# # rule_data = rule_data3.drop_duplicates(['userid'])
# rule_data = rule_data3.drop_duplicates(['userid', 'rule_data'])
#
# prob = pd.merge(prob, rule_data, on='userid', how='left')
# print(prob)

orderType = []
i = 0
for index, row in prob.iterrows():
#     # if ((0.40 < row.orderType_1 < 0.5) and (row.orderType_2 > 0.85)):
#     #     orderType.append(row.orderType_2)
#     #     print(row.userid)
#     #     i = i + 1
#     # else:
#     #     orderType.append(row.orderType_1)
#
    if(abs(row.orderType_1-0.5) < abs(row.orderType_2-0.5) and (row.orderType_1 > 0.5) and (row.orderType_2 > 0.5)):
        orderType.append(row.orderType_2)
        print(row.userid)
        i = i + 1
    else:
        if ((abs(row.orderType_1 - 0.5) < abs(row.orderType_2 - 0.5)) and (row.orderType_1 < 0.5) and (row.orderType_2 < 0.5)):
            orderType.append(row.orderType_2)
            print(row.userid)
            i = i + 1
        else:
            orderType.append(row.orderType_1)
#
#     # if ((row.rule_data == 1) and (row.orderType_1 > 0.5)):
#     #     orderType.append(1)
#     #     print(row.userid)
#     #     print("1111111")
#     #     i = i + 1
#     # else:
#     if ((row.rule_data == 0) and (row.orderType_1 < 0.5)):
#         orderType.append(0)
#         print(row.userid)
#         print("0000000")
#         i = i + 1
#     else:
#         orderType.append(row.orderType_1)
#
#
print(i)
prob['orderType'] = orderType
print(prob)
prob.drop(['orderType_1', 'orderType_2'], axis=1, inplace=True)
prob.to_csv(dire + 'submit/fusion/sub_susion_34_stacking_prob20180124_015740_36.csv', index=False)

prob['orderType'][prob['orderType'] > 0.5] = 1
prob['orderType'][~(prob['orderType'] > 0.5)] = 0
print("1的数目：", len(prob[prob['orderType'] == 1]))

