from tqdm import tqdm
import pandas as pd
import numpy as np
import gc
import copy

def AutoComple(action,caction):
    action['actionTime'] = action['actionTime'].apply(lambda x: pd.to_datetime(x))
    aT4 = action[action['actionType'] > 3]
    temp1 = aT4.groupby(['userid'],as_index=True)['actionType'].diff()
    #print(temp1)
    #print(caction)
    temp2 = action.groupby(['userid'],as_index=True)['actionTime'].diff()
    temp2 = pd.DataFrame(temp2)
    temp2['actionTime'] = temp2['actionTime'].apply(lambda x: x.seconds)
    temp2 = temp2[temp2['actionTime'] <= 21600]
    #print(temp2)
    temp2 = pd.DataFrame(temp2)
    temp1 =pd.DataFrame(temp1)
    diffd2 = temp1[temp1['actionType'] > 1]
    #print(diffd2)
    #indexs = list(diffd2.index)
    indexs = list(set(diffd2.index).intersection(set(temp2.index)))
    list.sort(indexs)
    #print(indexs)
    count = 0
    c = 0
    for index in tqdm(indexs):
        c += 1
        d = index
        index += count
        if int(caction.loc[index-1, ['actionType']]) < 4:
            continue
        above = caction.loc[:index-1]
        below = caction.loc[index:]
        c = int(diffd2.loc[d, ['actionType']])
        count += c-1
        for i in range(1, c):
            action_type = i + int(caction.loc[index-1, ['actionType']])
            userid = int(caction.loc[index-1, ['userid']])
            time = (int(caction.loc[index,['actionTime']])-int(caction.loc[index-1,['actionTime']]))/c * i + int(caction.loc[index-1,['actionTime']])
            insertrow = pd.DataFrame([[userid, action_type,time]],columns=['userid','actionType','actionTime'])
            #print(insertrow)
            above = above.append(insertrow, ignore_index=True)
        caction = above.append(below, ignore_index=True)
    caction.to_csv('insert_action2.csv',index=False)


def AutoComple1(action):
    caction = copy.deepcopy(action)
    caction.dropna(subset=['actionType_time'], inplace=True)
    caction.reset_index(inplace=True)
    action['action_time'] = action['action_time'].apply(lambda x: pd.to_datetime(x))
    temp = action.groupby(['userid'], as_index=True)['action_time'].diff()
    temp = pd.DataFrame(temp)
    temp['action_time'] = temp['action_time'].apply(lambda x: x.seconds)
    temp = temp[temp['action_time'] >= 21600]  #选出时间间隔大于2小时的操作
    indexs = list(temp.index)
    del temp
    gc.collect()
    list.sort(indexs)
    count = 0
    for index in tqdm(indexs):

        d = index
        index += count
        buquan = int(action.loc[index, ['actionType']])
        if buquan == 1:  #间隔大 且当前actionType不为1的需要补全
            continue
        userid1 = int(caction.loc[d, ['userid']])
        temp1 = caction[caction['userid'] == userid1]
        temp1.reset_index(inplace=True)
        total_time = 0
        cou = 0
        for i in range(len(temp1)-1):
            if int(temp1.loc[i,['actionType']]) == 1 and int(temp1.loc[i+1,['actionType']]) == buquan and int(temp1.loc[i,['actionType_time']]) < 1200:
                total_time += int(temp1.loc[i,['actionType_time']])
                cou += 1
        if total_time == 0:
            for i in range(len(caction)-1):
                if int(caction.loc[i,['actionType']]) == 1 and int(caction.loc[i+1,['actionType']]) == buquan and int(caction.loc[i,['actionType_time']]) < 1200:
                    total_time += int(caction.loc[i,['actionType_time']])
                    cou += 1
        above = action.loc[:index - 1]
        below = action.loc[index:]
        count += 1                                      #补一行1之后索引加1
        action_type = 1
        userid = int(caction.loc[index, ['userid']])
        time = int(action.loc[index, ['actionTime']]) - total_time / cou
        del action,temp1
        gc.collect()
        insertrow = pd.DataFrame([[userid, action_type, time, 1]], columns=['userid','actionType','actionTime', 'is_insert'])
        above = above.append(insertrow, ignore_index=True)
        action = above.append(below, ignore_index=True)
        del above, below, insertrow
        gc.collect()
    action.to_csv('./new/test_insert_1.csv')


def AutoComple2(action):
    action['action_time'] = action['action_time'].apply(lambda x: pd.to_datetime(x))
    temp = action.groupby(['userid'], as_index=True)['action_time'].diff()
    temp = pd.DataFrame(temp)
    temp['action_time'] = temp['action_time'].apply(lambda x: x.seconds)
    temp = temp[temp['action_time'] >= 21600]  #选出时间间隔大于2小时的操作
    indexs = list(temp.index)
    del temp
    gc.collect()
    list.sort(indexs)
    count = 0
    for index in tqdm(indexs):
        index += count
        buquan = int(action.loc[index, ['actionType']])
        if buquan == 1:  #间隔大 且当前actionType不为1的需要补全
            continue
        above = action.loc[:index - 1]
        below = action.loc[index:]
        count += 1                                      #补一行1之后索引加1
        action_type = 1
        userid = int(action.loc[index, ['userid']])
        time = 0
        del action
        gc.collect()
        insertrow = pd.DataFrame([[userid, action_type, time, 1]], columns=['userid','actionType','actionTime', 'is_insert'])
        above = above.append(insertrow, ignore_index=True)
        action = above.append(below, ignore_index=True)
        del above, below, insertrow
        gc.collect()
    action.to_csv('./new/test_insert_1.csv')

def addtimestamp(action):
    temp = action[action['actionTime'] == 0]
    caction = copy.deepcopy(action)
    caction.dropna(subset=['actionType_time'],inplace=True)
    caction.reset_index(inplace=True)
    indexs = temp.index
    indexs = list(indexs)
    print(indexs)
    all_total_time = 0
    all_cou = 0
    time = []
    list_ = [2,3,4,5,6,7,8,9]
    for k in list_:
        print(k)
        for i in tqdm(range(len(caction) - 1)):
            if int(caction.loc[i, ['actionType']]) == 1 and int(caction.loc[i + 1, ['actionType']]) == k and int(
                    caction.loc[i, ['actionType_time']]) < 1200:
                all_total_time += int(caction.loc[i, ['actionType_time']])
                all_cou += 1
        time.append([k,all_total_time,all_cou])
    for index in tqdm(indexs):
        userid = int(action.loc[index,['userid']])
        temp1 = caction[caction['userid'] == userid]
        temp1.reset_index(inplace=True)
        buquan = int(action.loc[index+1,['actionType']])
        total_time = 0
        cou = 0
        for i in range(len(temp1)-1):
            if int(temp1.loc[i,['actionType']]) == 1 and int(temp1.loc[i+1,['actionType']]) == buquan and int(temp1.loc[i,['actionType_time']]) < 1200:
                total_time += int(temp1.loc[i, ['actionType_time']])
                cou += 1
        if total_time == 0:
            total_time_list = time[buquan-2]
            total_time = total_time_list[1]
            cou = total_time_list[2]
        action.loc[index,['actionTime']] = int(action.loc[index+1,['actionTime']]) - total_time / cou
    columns = ['userid', 'actionType', 'actionTime', 'action_time', 'orderid', 'action_date', 'actionType_time', 'is_insert']
    action.to_csv('insert_action_train_2.csv', index=False, columns=columns)



file2 = pd.read_csv('insert_action_train_1.csv')
#file3 = pd.read_csv('./data/test/action_test.csv')
#file4 = pd.read_csv('./data/test/action_test1.csv')
addtimestamp(file2)