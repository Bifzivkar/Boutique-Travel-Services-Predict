# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from datetime import datetime
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

dire = '../../data/'
train = pd.read_csv(dire + 'train3.csv', encoding='utf-8')
test = pd.read_csv(dire + 'test3.csv', encoding='utf-8')

data_train = pd.read_csv(dire + 'data_train.csv', encoding='gb2312')
data_test = pd.read_csv(dire + 'data_test.csv', encoding='gb2312')
data_train.drop(['futureOrderType'], axis=1, inplace=True)

train = pd.merge(train, data_train, on='userid', how='left')
test = pd.merge(test, data_test, on='userid', how='left')

# train_enumerate = pd.read_csv(dire + 'backup/train_enumerate1.csv', encoding='utf-8')
# test_enumerate = pd.read_csv(dire + 'backup/test_enumerate1.csv', encoding='utf-8')
# columns = train_enumerate.columns
# columns_t = test_enumerate.columns
# train_enumerate.drop(columns[1:111], axis=1, inplace=True)
# test_enumerate.drop(columns_t[1:110], axis=1, inplace=True)
# # print(train_enumerate)
# # print(test_enumerate)
# train = pd.merge(train, train_enumerate, on='userid', how='left')
# test = pd.merge(test, test_enumerate, on='userid', how='left')

print("训练集0：", train.shape)
print("测试集0：", test.shape)

def one_hot(table, one_hot_feature):
    for f in one_hot_feature:
        dummies = pd.get_dummies(table[f], prefix= f)
        table = pd.concat([table, dummies], axis=1)
        table.drop([f], axis=1, inplace=True)
    return table

one_hot_feature = ['gender_e', 'age_e', 'province_e', 'continent_e', 'season']
dataset = pd.concat([train,test])
# dataset['tag_1'].fillna(0, inplace=True)
dataset = one_hot(dataset, one_hot_feature)
train = dataset[dataset.orderType.notnull()]
test = dataset[dataset.orderType.isnull()]


# feature = [x for x in dataset.columns if x not in ['userid', 'gender', 'province', 'age',
#          'orderType', 'actionTime_future', 'action_time_future', 'orderTime', 'order_time',
#          'time_1', 'time_2', 'time_2_c',
         # 'latest_1day_actionType1_count', 'latest_1day_actionType2_count',
         # 'latest_1day_actionType4_count', 'latest_1day_actionType5_count',
         # 'latest_1day_actionType6_count', 'latest_1day_actionType7_count',
         # open自带要去掉的
         # 'timespanmean_last_4', 'timespanmean_last_7', 'timespanmean_last_8',
         # 'timespanmean_last_9',
         # 'actiontype_last_8', 'actiontype_last_9', 'actiontype_last_10',
         # 'actiontype_last_11', 'actiontype_last_12', 'actiontype_last_13',
         # 'actiontype_last_14', 'actiontype_last_15', 'actiontype_last_16',
         # 'actiontype_last_17', 'actiontype_last_18', 'actiontype_last_19',
         # 'actiontype_last_20',
         # 'actiontime_last_8', 'actiontime_last_9', 'actiontime_last_10',
         # 'actiontime_last_11', 'actiontime_last_12', 'actiontime_last_13',
         # 'actiontime_last_14', 'actiontime_last_15', 'actiontime_last_16',
         # 'actiontime_last_17', 'actiontime_last_18', 'actiontime_last_19',
         # 'actiontime_last_20',
         # 'histord_time_last_2', 'histord_time_last_2_year',
         # 'histord_time_last_2_month', 'histord_time_last_3',
         # 'histord_time_last_3_year', 'histord_time_last_3_month', '新竹', '布里斯班',
         # '丹佛', '成田市', '莫斯科', '维多利亚', '上海', '雷克雅未克', '开普敦', '京都', '墨西哥城', '牛津',
         # '阿姆斯特丹', '都柏林', '东京', '珀斯', '华盛顿', '卡萨布兰卡', '亚特兰大', '巴尔的摩', '法兰克福',
         # '富山市', '惠灵顿', '萨尔茨堡', '富国岛', '戛纳', '仰光', '名古屋', '台南', '堪培拉', '冲绳--那霸',
         # '新加坡', '马赛', '达拉斯', '垦丁', '北海道--札幌', '马尼拉', '温哥华', '澳门', '苏梅岛', '波尔多',
         # '斯德哥尔摩', '温莎', '巴伦西亚', '渥太华', '里昂', '迈阿密', '卢塞恩', '西雅图', '奥兰多', '岐阜县',
         # '汉密尔顿', '波士顿', '尼斯', '那不勒斯', '拉科鲁尼亚', '多哈', '华沙', '云顶高原', '横滨', '圣地亚哥',
         # '考文垂', '福冈', '霍巴特', '台中', '墨尔本', '冲绳市', '毛里求斯', '洛杉矶', '蒙特利尔', '曼彻斯特',
         # '嘉义', '大叻', '伊尔库茨克', '楠迪', '阿里山', '布拉格', '沙巴--亚庇', '日内瓦', '蒂卡波湖',
         # '里斯本', '巴黎', '釜山', '洞爷湖', '静冈县', '普吉岛', '台北', '夏威夷欧胡岛（檀香山）', '科茨沃尔德',
         # '科隆', '夏威夷大岛', '开罗', '凯恩斯', '圣保罗', '威尼斯', '旧金山', '奈良', '赫尔辛基', '哥本哈根',
         # '阿布扎比', '御殿场市', '万象', '哈尔施塔特', '富良野', '阿德莱德', '香港', '岘港', '金泽', '高雄',
         # '巴厘岛', '谢菲尔德', '罗马', '哥德堡', '济州岛', '北海道--登别', '马拉加', '华欣', '塞班岛',
         # '塞维利亚', '北海道--旭川', '吉隆坡', '新山', '奥克兰', '马德里', '斯克兰顿（宾夕法尼亚州）', '槟城',
         # '格拉纳达', '伦敦', '黄金海岸', '皇后镇', '轻井泽', '暹粒', '广岛', '大西洋城', '汉堡', '神户',
         # '河内', '布法罗', '福森', '巴塞罗那', '富士河口湖', '宜兰', '多伦多', '夏威夷茂宜岛', '胡志明市',
         # '纽约', '星野度假村', '熊本市', '约克', '布达佩斯', '奥斯陆', '盐湖城', '杜塞尔多夫', '阿维尼翁',
         # '爱丁堡', '千叶市', '维也纳', '雅典', '大阪', '圣彼得堡', '清迈', '巴斯', '芭堤雅', '芝加哥',
         # '富士山', '贝尔法斯特', '费城', '迪拜', '基督城', '新北', '加德满都', '底特律', '柏林', '箱根',
         # '金边', '花莲', '米兰', '曼谷', '利瓦绿洲', '甲米', '佛罗伦萨', '因特拉肯', '慕尼黑', '波尔图',
         # '约翰内斯堡', '苏黎世', '北海道--小樽', '休斯敦', '拉斯维加斯', '布鲁塞尔', '千叶', '阿尔勒', '摩纳哥',
         # '长崎', '悉尼', '芽庄', '卡尔加里', '兰卡威', '首尔', '雅加达', '宿务', '北海道--函馆', '伊斯坦布尔',
         # '伯明翰', '南投', 'action_sum', 'rating_min', 'rating_last', 'gender_exist',
         # 'gender_male', 'gender_female'

        # ]]

feature = [x for x in dataset.columns if x not in ['userid', 'gender', 'province', 'age',
         'orderType', 'actionTime_future', 'action_time_future', 'orderTime', 'order_time',
         'time_1', 'time_2', 'time_2_c'
          ]]


print("训练集1：", train.shape)
print("测试集1：", test.shape)

train1 = train[train.count_all.isnull()]   # count_all为空   历史订单中无对应的
train2 = train[train.count_all.notnull()]  # count_all不为空 历史订单中有对应的

# train1_x, val1_x, train1_y, val1_y = train_test_split(train1[feature], train1["orderType"], test_size=0.2, random_state=0, stratify = train1["orderType"])
# train2_x, val2_x, train2_y, val2_y = train_test_split(train2[feature], train2["orderType"], test_size=0.2, random_state=0, stratify = train2["orderType"])
#
# val_x = val1_x.append(val2_x, ignore_index=True)
# val_y = val1_y.append(val2_y, ignore_index=True)
# train_x = train1_x.append(train2_x, ignore_index=True)
# train_y = train1_y.append(train2_y, ignore_index=True)
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)
train_x, val_x, train_y, val_y = train_test_split(train[feature], train["orderType"], test_size=0.2, random_state=0, stratify = train["orderType"])

print(val_x.shape)
print(val_y.shape)
print(train_x.shape)
print(train_y.shape)
########## XGBOOST ##########
params = {
    'booster': "gbtree",
    'objective': "binary:logistic",
    'eval_metric': "auc",
    'max_depth': 9,
    'subsample': 0.8,
    'colsample_bytree ': 0.9,
    'eta': 0.01,
    'lambda': 3,
    'silent': 0,
    'scale_pos_weight': 4,
    'min_child_weight': 1,
    'missing': -999,
}
# xgb_train = xgb.DMatrix(train_x, train_y, missing=np.nan)
# xgb_eval = xgb.DMatrix(val_x, val_y, missing=np.nan)
# watchlist = [(xgb_train, 'train'), (xgb_eval, 'eval')]

xgb_train = xgb.DMatrix(train[feature], train["orderType"], missing=np.nan)
watchlist = [(xgb_train, 'train')]

k = '68'
# gbm = xgb.train(params, xgb_train, num_boost_round=2262, evals=watchlist, early_stopping_rounds=50)
print("训练完成：", datetime.now())
# joblib.dump(gbm, './model/xgb' + k + '.model')

# 预测
print("start predict")
gbm = joblib.load('./model/xgb' + k + '.model')
pred = gbm.predict(xgb.DMatrix(test[feature], missing=np.nan), ntree_limit=gbm.best_ntree_limit)

prob = test[['userid']]
prob['orderType'] = pred
sub = pd.DataFrame(prob)
sub.to_csv(dire + 'submit/xgb/sub_xgb_porb' + k + '.csv', index=False)

prob['orderType'][prob['orderType'] > 0.5] = 1
prob['orderType'][~(prob['orderType'] > 0.5)] = 0
print("1的数目：", len(prob[prob['orderType'] == 1]))

# importances = gbm.feature_importances_
# df_featImp = pd.DataFrame({'tags': feature, 'importance': importances})
# df_featImp_sorted = df_featImp.sort_values(by=['importance'], ascending=False)
# df_featImp_sorted.plot(x='tags', y='importance', kind='bar')
# df_featImp.to_csv(dire + 'submit/xgb/feature/feat_xgb'+ k +'.csv')
# df_featImp_sorted.to_csv(dire + 'submit/xgb/feature/feat_xgb'+ k +'.csv')