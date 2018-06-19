# -*- encoding:utf-8 -*-
import pandas as pd

dire = '../../data/'
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')

action_train = pd.read_csv(dire + 'train/action_train.csv', encoding='utf-8')
action_test = pd.read_csv(dire + 'test/action_test.csv', encoding='utf-8')

# 浏览记录表按时间顺序关联历史订单表
def action_link_orderHistory(action, orderHistory):
    orderid = []
    for index, row in action.iterrows():
        order = orderHistory[(orderHistory['userid'] == row.userid)].reset_index()
        i = len(order)-1
        # print(order['order_time'][i-1])
        while(1):
            print(i)
            if (len(order) == 0):
                orderid.append(None)
                break
            if(row.action_time > order['order_time'][len(order)-1]):
                orderid.append(None)
                break
            # print("订单时间0", order['order_time'][i-1])
            # print("订单时间1", order['order_time'][i])
            # print("浏览时间", row.action_time)
            if ((i != 0) and (row.action_time > order['order_time'][i-1]) and (row.action_time <= order['order_time'][i])):
                orderid.append(order['orderid'][i])
                break
            else:
                if((i == 0) and (row.action_time <= order['order_time'][i])):
                    orderid.append(order['orderid'][i])
                    break
                else:
                    i = i-1
        print(index)
        # print(orderid)
    return orderid

# action_train['orderid'] = action_link_orderHistory(action_train, orderHistory_train)
# action_test['orderid'] = action_link_orderHistory(action_test, orderHistory_test)

# 修正边界处订单号问题
def fix(action):
    orderid = []
    last_userid = -1
    last_orderid = -1
    last_actionType = -1
    for index, row in action.iterrows():
        print(index)
        if(row.userid == last_userid):
            if((row.orderid != last_orderid) and ((row.actionType >= last_actionType) or (row.actionType >= 6))):
                action['orderid'][index] = last_orderid
                row.orderid = last_orderid
        last_userid = row.userid
        last_orderid = row.orderid
        last_actionType = row.actionType
    return action
action_train = fix(action_train)
action_test = fix(action_test)

print(action_train)

action_train.to_csv(dire + 'train/action_train1.csv', index=False, encoding='utf-8')
action_test.to_csv(dire + 'test/action_test1.csv', index=False, encoding='utf-8')




