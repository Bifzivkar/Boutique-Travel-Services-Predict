# -*- encoding:utf-8 -*-
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.externals import joblib
from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split

start_time = datetime.now()
# 计算AUC
def caculateAUC(test_pre, test_label):
    pre = np.array(test_pre)
    result = metrics.roc_auc_score(test_label, pre)
    return result

dire = '../../data/'
k = '6'
x = '1'
train = pd.read_csv(dire + 'backup/train6.csv', encoding='utf-8')
test = pd.read_csv(dire + 'backup/test6.csv', encoding='utf-8')

data_train = pd.read_csv(dire + 'data_open/data_train.csv', encoding='gb2312')
data_test = pd.read_csv(dire + 'data_open/data_test.csv', encoding='gb2312')
data_train.drop(['futureOrderType'], axis=1, inplace=True)
train = pd.merge(train, data_train, on='userid', how='left')
test = pd.merge(test, data_test, on='userid', how='left')

# data_train = pd.read_csv(dire + 'data_open/data_train1.csv', encoding='utf-8')
# data_test = pd.read_csv(dire + 'data_open/data_test1.csv', encoding='utf-8')
# data_train.drop(['futureOrderType'], axis=1, inplace=True)
# train = pd.merge(train, data_train, on='userid', how='left')
# test = pd.merge(test, data_test, on='userid', how='left')

train_feature = pd.read_csv(dire + 'backup/LOVECT/train_feature1.csv', encoding='utf-8', header=None)
test_feature = pd.read_csv(dire + 'backup/LOVECT/test_feature1.csv', encoding='utf-8', header=None)
train = pd.concat([train, train_feature], axis=1)
test = pd.concat([test, test_feature], axis=1)

# matf_train = dire + 'backup/LOVECT/train_feature.mat'
# matf_test = dire + 'backup/LOVECT/test_feature.mat'
# data_train = sio.loadmat(matf_train)
# data_test = sio.loadmat(matf_test)
# train_feature = data_train['feature']
# test_feature = data_test['feature']
# train_feature = np.array(train_feature)
# test_feature = np.array(test_feature)
# train_feature = pd.DataFrame(train_feature)
# test_feature = pd.DataFrame(test_feature)
# train = pd.concat([train, train_feature], axis=1)
# test = pd.concat([test, test_feature], axis=1)

# fet_zyf_train = pd.read_csv(dire + 'backup/zyf/fet_zyf_train.csv', encoding='utf-8')
# fet_zyf_test = pd.read_csv(dire + 'backup/zyf/fet_zyf_test.csv', encoding='utf-8')
# train = pd.merge(train, fet_zyf_train, on='userid', how='left')
# test = pd.merge(test, fet_zyf_test, on='userid', how='left')

print("训练集：", train.shape)
print("测试集：", test.shape)

# 重构的时间特征
# train8 = pd.read_csv(dire + 'train8_1_select1.csv', encoding='utf-8')
# test8 = pd.read_csv(dire + 'test8_1_select1.csv', encoding='utf-8')
# # train9 = pd.read_csv(dire + 'train9.csv', encoding='utf-8')
# # test9 = pd.read_csv(dire + 'test9.csv', encoding='utf-8')
# reextract_action_feature = [x for x in train8.columns if x not in ['userid', 'orderType', 'use_app_days_count']]
# train.drop(reextract_action_feature, axis=1, inplace=True)
# test.drop(reextract_action_feature, axis=1, inplace=True)
# train = pd.merge(train, train8, on=['userid', 'orderType'], how='left')
# test = pd.merge(test, test8, on='userid', how='left')


# checkpoint_train = pd.read_csv(dire + 'backup/open/checkpoint_train.csv', encoding='utf-8')
# checkpoint_test = pd.read_csv(dire + 'backup/open/checkpoint_test.csv', encoding='utf-8')
# feature_checkpoint = ['userid', 'count']
# train = pd.merge(train, checkpoint_train[feature_checkpoint], on='userid', how='left')
# test = pd.merge(test, checkpoint_test[feature_checkpoint], on='userid', how='left')

print("融合重构action特征后训练集：", train.shape)
print("融合重构action特征后测试集：", test.shape)

# 计算AUC
# def caculateAUC(test_pre, test_label):
#     pre = np.array(test_pre[:, 1])
#     result = metrics.roc_auc_score(test_label, pre)
#     return result

# initialize data
def one_hot(table, one_hot_feature):
    for f in one_hot_feature:
        dummies = pd.get_dummies(table[f], prefix= f)
        table = pd.concat([table, dummies], axis=1)
        table.drop([f], axis=1, inplace=True)
    return table

one_hot_feature = ['gender_e', 'age_e', 'province_e', 'continent_e', 'season']
dataset = pd.concat([train, test])
dataset = one_hot(dataset, one_hot_feature)
train = dataset[dataset.orderType.notnull()]
test = dataset[dataset.orderType.isnull()]


feature = [x for x in dataset.columns if x not in ['userid', 'gender', 'province', 'age',
         'orderType', 'actionTime_future', 'action_time_future', 'orderTime', 'order_time',
         'time_1', 'time_2', 'time_2_c'
         , 'gender_e', 'age_e', 'province_e', 'continent_e', 'season', 'use_app_days_count'

         ,'莫斯科', '胡志明市', 'tag_举牌迎接', 'tag_额外收费', '芭堤雅', '花莲', '芽庄', '苏梅岛', 'tag_驾驶平稳', 'province_e_23.0', 'tag_建筑师', 'tag_驾驶技术一般', 'tag_留学生', 'province_e_21.0', 'province_e_26.0', '菲律宾', '萨尔茨堡', 'province_e_20.0', '蒂卡波湖', '蒙特利尔', '西藏', 'tag_驾驶鲁莽', 'tag_接待地点不合理', '谢菲尔德', '贝尔法斯特', '考文垂', 'tag_临时换司导', '老挝', '科茨沃尔德', 'tag_车况有点旧', '瑞士', 'rating_min', 'tag_车旧/脏', 'continent_e_5.0', '盐湖城', '神户', 'tag_不协助搬运行李', 'continent_e_2.0', 'tag_非订单车辆', '福森', '科隆', 'tag_提前就终止了服务', '箱根', 'tag_频繁催促', '约克', '约翰内斯堡', 'province_e_30.0', 'province_e_29.0', 'province_e_28.0', '维多利亚', 'province_e_27.0', '缅甸', 'tag_不爱讲话', 'tag_路线不熟', '费城', 'age_e_3.0', 'province_e_5.0', 'province_e_10.0', 'tag_行前联系不上', 'province_e_8.0', '雅典', '雅加达', '雷克雅未克', '霍巴特', 'province_e_7.0', '静冈县', 'province_e_6.0', '首尔', '香港', '赫尔辛基', '马拉加', 'tag_未提前联系', '马赛', 'tag_行前未主动联系', '黄金海岸', '黑龙江', 'tag_沉默寡言', 'province_e_2.0', 'province_e_1.0', 'province_e_0.0', 'tag_没水/空调不足', 'tag_景点不熟悉', '阿里山', '珀斯', '阿维尼翁', 'tag_言语粗鲁', '越南', 'province_e_15.0', '轻井泽', 'action_56789_count_c', '达拉斯', '迈阿密', 'tag_提前结束行程', '那不勒斯', '都柏林', '里斯本', '里昂', 'province_e_14.0', 'actiontimespancount_8_9', 'tag_行程有点紧', '金泽', '金边', '釜山', '长崎', '阿姆斯特丹', '阿尔勒', 'province_e_12.0', 'tag_行程安排不合理', '北海道--小樽', 'action_6789_count', '爱尔兰', '俄罗斯', '宿务', '宜兰', '伯明翰', '奥斯陆', '奥兰多', '奈良', '冰岛', '爱丁堡', '大西洋城', '大叻', '多哈', '夏威夷茂宜岛', '夏威夷大岛', '墨西哥城', '富国岛', '富士山', '富士河口湖', '富山市', '富良野', '尼斯', '尼泊尔', '休斯敦', '岐阜县', '岘港', '巴伦西亚', '巴厘岛', '伊尔库茨克', '巴尔的摩', '巴斯', '巴西', '布达佩斯', '墨西哥', '冲绳市', '塞维利亚', '台南', '北海道--登别', '千叶', '千叶市', '华欣', '华沙', '北海道--函馆', '南投', '南非', '卡塔尔', '卡萨布兰卡', '卢塞恩', '印度尼西亚', '台中', '台北', '匈牙利', '塞班岛', '加德满都', '名古屋', '哈尔施塔特', '哥德堡', '嘉义', '利瓦绿洲', '因特拉肯', '圣保罗', '圣地亚哥', '圣彼得堡', '垦丁', '埃及', '基督城', '堪培拉', '布里斯班', '布鲁塞尔', '京都', '波士顿', '楠迪', '横滨', '比利时', '毛里求斯', '毛里求斯.1', '汉堡', '汉密尔顿', 'continent_driver_count_avg', 'histord_sum_cont4', '河内', 'histord_sum_cont5', 'histord_time_last_2_year', '法兰克福', '波兰', '波尔图', '柬埔寨', 'action_5678_count', 'histord_time_last_3_year', '洞爷湖', 'latest_1day_actionType9_count', '北海道--旭川', 'action_678_count', 'action_678_count_c', '渥太华', '温莎', 'age_00', 'action_789_count', 'action_78_count', '澳门', '熊本市', '格拉纳达', '柏林', '亚特兰大', '捷克', '广岛', '云顶高原', 'tag_车内有异味', '开普敦', '开罗', '御殿场市', '德国', '惠灵顿', '慕尼黑', '成田市', '戛纳', '丹佛', '拉科鲁尼亚', '挪威', '摩洛哥', 'histord_sum_all', '摩纳哥', '斯克兰顿（宾夕法尼亚州）', '中国香港', '新加坡.1', '新北', '新疆', '新竹', '新西兰', '中国澳门', '中国', '星野度假村', '上海', '暹粒', '曼彻斯特', '万象'
         , 'latest_2day_actionType9_count', 'continent_e_1.0', 'continent_type1_count_avg', 'continent_e_4.0', '高雄', 'province_e_17.0', 'gender_e_1.0', 'latest_7day_actionType9_count', 'age_e_1.0', 'province_e_25.0', 'latest_5day_actionType8_count', 'province_e_9.0', 'province_e_22.0', 'province_e_13.0', '韩国', 'latest_6day_actionType9_count', '宁夏', '青海', 'action_5678_count_c', '奥地利', '布拉格', '多伦多', '哥本哈根', '斐济', '斯德哥尔摩', '北海道--札幌', '槟城', '佛罗伦萨', '丹麦', '波尔多', 'action_56789_count', '海南', '牛津', 'rating_mean', '皇后镇', 'tag_车辆和订单显示不符', 'tag_车况脏乱', '米兰', 'tag_私聊不回复', 'tag_没做景点介绍', '芬兰', '英国', '荷兰', '葡萄牙', 'tag_未举牌服务', 'tag_景点不介绍', 'tag_主动热情', 'tag_路线熟悉'
          ]]


print("训练集：", train.shape)
print("测试集：", test.shape)

train1 = train[train.count_all.isnull()]   # count_all为空    历史订单中无对应的
train2 = train[train.count_all.notnull()]  # count_all不为空  历史订单中有对应的

train1.fillna(train1.min(), inplace=True)
train2.fillna(train2.min(), inplace=True)
test.fillna(test.min(), inplace=True)

train1_x, val1_x, train1_y, val1_y = train_test_split(train1[feature], train1["orderType"], test_size=0.2, random_state=0, stratify=train1["orderType"])
train2_x, val2_x, train2_y, val2_y = train_test_split(train2[feature], train2["orderType"], test_size=0.2, random_state=0, stratify=train2["orderType"])

val_x = val1_x.append(val2_x, ignore_index=True)
val_y = val1_y.append(val2_y, ignore_index=True)
train_x = train1_x.append(train2_x, ignore_index=True)
train_y = train1_y.append(train2_y, ignore_index=True)

# train.fillna(train.min(), inplace=True)
# test.fillna(test.min(), inplace=True)
# train_x, val_x, train_y, val_y = train_test_split(train[feature], train["orderType"], test_size=0.2, random_state=0, stratify=train["orderType"])


print(val_x.shape)
print(val_y.shape)
print(val_y)
print(val_y[val_y['orderType'] == 1].shape)
print(train_x.shape)
print(train_y.shape)
print(train_y[train_y['orderType'] == 1].shape)

# specify the training parameters

# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'auc'},
#     'num_leaves': 230,  # 130 0.968759
#     'learning_rate': 0.01,
#     'feature_fraction': 0.87,  # 0.96744 0.87
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 130,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 4,
    'verbose': -1
}
lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)

gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=80)

# 验证集的准确率
print("AUC:" + str(caculateAUC(gbm.predict(val_x), val_y)))
# print("准确度:" + str(gbm.score(val_x, val_y)))
# joblib.dump(gbm, dire + 'submit/lgb/model/lgb' + k + '_' + x + '.model')
# gbm = joblib.load(dire + 'submit/lgb/model/lgb' + k + '_' + x + '.model')
# lgb.plot_importance(gbm, importance_type='gain', ignore_zero=False, figsize=(10, 6))

preds = gbm.predict(test[feature], num_iteration=gbm.best_iteration)

prob = test[['userid']]
prob['orderType'] = preds
prob = pd.DataFrame(prob)
prob.to_csv(dire + 'submit/lgb/prob_lgb' + k + '_' + x + '.csv', index=False)
prob['orderType'][prob['orderType'] > 0.5] = 1
prob['orderType'][~(prob['orderType'] > 0.5)] = 0
print("1的数目：", len(prob[prob['orderType'] == 1]))

print("开始时间:", start_time)
print("结束时间:", datetime.now())

importances = gbm.feature_importance()
df_featImp = pd.DataFrame({'tags': feature, 'importance': importances})
df_featImp_sorted = df_featImp.sort_values(by=['importance'], ascending=False)
df_featImp_sorted.plot(x='tags', y='importance', kind='bar')
df_featImp_sorted.to_csv(dire + 'submit/lgb/feature/feat_lgb' + k + '_' + x + '.csv')