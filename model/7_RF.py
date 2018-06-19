# -*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import model_selection


dire = '../../data/'
train = pd.read_csv(dire + 'train5.csv', encoding='utf-8')
test = pd.read_csv(dire + 'test5.csv', encoding='utf-8')

data_train = pd.read_csv(dire + 'data_train.csv', encoding='gb2312')
data_test = pd.read_csv(dire + 'data_test.csv', encoding='gb2312')
data_train.drop(['futureOrderType'], axis=1, inplace=True)

train = pd.merge(train, data_train, on='userid', how='left')
test = pd.merge(test, data_test, on='userid', how='left')

print("训练集：", train.shape)
print("测试集：", test.shape)


# 计算AUC
def caculateAUC(test_pre, test_label):
    pre = np.array(test_pre[:, 1])
    result = metrics.roc_auc_score(test_label, pre)
    return result

# initialize data
def one_hot(table, one_hot_feature):
    for f in one_hot_feature:
        dummies = pd.get_dummies(table[f], prefix= f)
        table = pd.concat([table, dummies], axis=1)
        table.drop([f], axis=1, inplace=True)
    return table

one_hot_feature = ['gender_e', 'age_e', 'province_e', 'continent_e', 'season']
dataset = pd.concat([train,test])
dataset = one_hot(dataset, one_hot_feature)
train = dataset[dataset.orderType.notnull()]
test = dataset[dataset.orderType.isnull()]

# drop_tags = [idcol, target,
#          #'actiontypeprop_1', 'actiontypeprop_2', 'actiontypeprop_3', 'actiontypeprop_4', 'actiontypeprop_5', 'actiontypeprop_6', 'actiontypeprop_7', 'actiontypeprop_8', 'actiontypeprop_9',
#          'timespanmean_last_4', 'timespanmean_last_7', 'timespanmean_last_8', 'timespanmean_last_9',
#          'actiontype_last_8', 'actiontype_last_9', 'actiontype_last_10', 'actiontype_last_11', 'actiontype_last_12', 'actiontype_last_13', 'actiontype_last_14', 'actiontype_last_15', 'actiontype_last_16', 'actiontype_last_17', 'actiontype_last_18', 'actiontype_last_19', 'actiontype_last_20',
#          'actiontime_last_8', 'actiontime_last_9', 'actiontime_last_10', 'actiontime_last_11', 'actiontime_last_12', 'actiontime_last_13', 'actiontime_last_14', 'actiontime_last_15', 'actiontime_last_16', 'actiontime_last_17', 'actiontime_last_18', 'actiontime_last_19', 'actiontime_last_20',
#          'histord_time_last_2', 'histord_time_last_2_year', 'histord_time_last_2_month', 'histord_time_last_3', 'histord_time_last_3_year', 'histord_time_last_3_month', '新竹', '布里斯班', '丹佛', '成田市', '莫斯科', '维多利亚', '上海', '雷克雅未克', '开普敦', '京都', '墨西哥城', '牛津', '阿姆斯特丹', '都柏林', '东京', '珀斯', '华盛顿', '卡萨布兰卡', '亚特兰大', '巴尔的摩', '法兰克福', '富山市', '惠灵顿', '萨尔茨堡', '富国岛', '戛纳', '仰光', '名古屋', '台南', '堪培拉', '冲绳--那霸', '新加坡', '马赛', '达拉斯', '垦丁', '北海道--札幌', '马尼拉', '温哥华', '澳门', '苏梅岛', '波尔多', '斯德哥尔摩', '温莎', '巴伦西亚', '渥太华', '里昂', '迈阿密', '卢塞恩', '西雅图', '奥兰多', '岐阜县', '汉密尔顿', '波士顿', '尼斯', '那不勒斯', '拉科鲁尼亚', '多哈', '华沙', '云顶高原', '横滨', '圣地亚哥', '考文垂', '福冈', '霍巴特', '台中', '墨尔本', '冲绳市', '毛里求斯', '洛杉矶', '蒙特利尔', '曼彻斯特', '嘉义', '大叻', '伊尔库茨克', '楠迪', '阿里山', '布拉格', '沙巴--亚庇', '日内瓦', '蒂卡波湖', '里斯本', '巴黎', '釜山', '洞爷湖', '静冈县', '普吉岛', '台北', '夏威夷欧胡岛（檀香山）', '科茨沃尔德', '科隆', '夏威夷大岛', '开罗', '凯恩斯', '圣保罗', '威尼斯', '旧金山', '奈良', '赫尔辛基', '哥本哈根', '阿布扎比', '御殿场市', '万象', '哈尔施塔特', '富良野', '阿德莱德', '香港', '岘港', '金泽', '高雄', '巴厘岛', '谢菲尔德', '罗马', '哥德堡', '济州岛', '北海道--登别', '马拉加', '华欣', '塞班岛', '塞维利亚', '北海道--旭川', '吉隆坡', '新山', '奥克兰', '马德里', '斯克兰顿（宾夕法尼亚州）', '槟城', '格拉纳达', '伦敦', '黄金海岸', '皇后镇', '轻井泽', '暹粒', '广岛', '大西洋城', '汉堡', '神户', '河内', '布法罗', '福森', '巴塞罗那', '富士河口湖', '宜兰', '多伦多', '夏威夷茂宜岛', '胡志明市', '纽约', '星野度假村', '熊本市', '约克', '布达佩斯', '奥斯陆', '盐湖城', '杜塞尔多夫', '阿维尼翁', '爱丁堡', '千叶市', '维也纳', '雅典', '大阪', '圣彼得堡', '清迈', '巴斯', '芭堤雅', '芝加哥', '富士山', '贝尔法斯特', '费城', '迪拜', '基督城', '新北', '加德满都', '底特律', '柏林', '箱根', '金边', '花莲', '米兰', '曼谷', '利瓦绿洲', '甲米', '佛罗伦萨', '因特拉肯', '慕尼黑', '波尔图', '约翰内斯堡', '苏黎世', '北海道--小樽', '休斯敦', '拉斯维加斯', '布鲁塞尔', '千叶', '阿尔勒', '摩纳哥', '长崎', '悉尼', '芽庄', '卡尔加里', '兰卡威', '首尔', '雅加达', '宿务', '北海道--函馆', '伊斯坦布尔', '伯明翰', '南投', 'action_sum', 'rating_min', 'rating_last', 'gender_exist', 'gender_male', 'gender_female'
#          ]

# feature = [x for x in dataset.columns if x not in ['userid', 'gender', 'province', 'age',
#          'orderType', 'actionTime_future', 'action_time_future', 'orderTime', 'order_time',
#          'time_1', 'time_2', 'time_2_c'
#          # 'latest_1day_actionType1_count', 'latest_1day_actionType2_count',
#          # 'latest_1day_actionType4_count', 'latest_1day_actionType5_count',
#          # 'latest_1day_actionType6_count', 'latest_1day_actionType7_count',
#          # '轻井泽', '贵州', '谢菲尔德', '中国香港', '费城', '赫尔辛基', '西藏', '丹佛', '蒙特利尔', '蒂卡波湖', '贝尔法斯特', '四川', '萨尔茨堡', '达拉斯', '迈阿密', '迪拜', '那不勒斯', '中国', '里斯本', '里昂', '重庆', '金泽', '金边', '釜山', '阿姆斯特丹', '阿尔勒', '葡萄牙', '莫斯科', '菲律宾', '维也纳', '甲米', '皇后镇', '伯明翰', '神户', '休斯敦', '福建', '福森', '科茨沃尔德', '科隆', '箱根', '米兰', '约克', '约翰内斯堡', '伊尔库茨克', '维多利亚', '万象', '缅甸', '仰光', '京都', '考文垂', '云顶高原', '云南', '丹麦', '芭堤雅', '花莲', '芽庄', '苏梅岛', '苏黎世', '英国', '荷兰', '上海', '阿里山', '阿维尼翁', 'action_67_count', 'province_e_9.0', 'province_e_10.0', 'province_e_11.0', 'province_e_12.0', 'province_e_13.0', 'province_e_15.0', 'province_e_16.0', 'province_e_17.0', 'province_e_18.0', 'action_78_count', 'action_789_count_c', 'province_e_21.0', 'action_789_count', 'province_e_23.0', 'province_e_25.0', 'province_e_7.0', 'province_e_26.0', 'province_e_27.0', 'action_678_count_c', 'province_e_29.0', 'action_678_count', 'action_6789_count_c', 'action_6789_count', 'action_5678_count_c', 'action_56789_count_c', 'continent_e_4.0', 'continent_e_5.0', 'action_56789_count', 'season_2', 'season_3', 'province_e_8.0', 'action_78_count_c', 'latest_6_actionType', '马来西亚', '瑞士', 'history_avg_rating', 'histord_time_last_3_year', '雅加达', '雷克雅未克', 'histord_time_last_3', '青海', '静冈县', '韩国', '首尔', 'histord_time_last_2_year', '马尼拉', 'histord_time_last_2_month', '马拉加', 'histord_sum_cont5', 'action_89_count', '黄金海岸', 'histord_sum_cont4', 'gender_male', 'age_80', 'age_00', 'age_e_1.0', 'age_e_2.0', 'age_e_3.0', 'actiontypeproplast20_9', 'actiontype_last_5', 'actiontype_last_14', 'actiontype_last_10', 'actiontimespancount_8_9', 'action_9', '甘肃', '牛津', '佛罗伦萨', '巴黎', '富士河口湖', '富山市', '富良野', '尼斯', '尼泊尔', '山东', '台中', '岐阜县', '岘港', '巴伦西亚', '巴厘岛', '巴尔的摩', '巴斯', '巴西', '布拉格', '富国岛', '布达佩斯', '布里斯班', '布鲁塞尔', '卡塔尔', '广东', '广岛', '广西', '底特律', '开普敦', '开罗', '御殿场市', '德国', '南非', '惠灵顿', '富士山', '宿务', '慕尼黑', '哈尔施塔特', '嘉义', '圣保罗', '圣地亚哥', '圣彼得堡', '垦丁', '埃及', '基督城', '堪培拉', '塞班岛', '哥本哈根', '哥德堡', '墨西哥', '墨西哥城', '夏威夷大岛', '夏威夷茂宜岛', '宜兰', '名古屋', '多哈', '大叻', '大西洋城', '吉林', '天津', '奈良', '奥克兰', '奥兰多', '奥地利', '奥斯陆', '台南', '宁夏', '台北', '南投', '成田市', '珀斯', '洞爷湖', '汉密尔顿', '匈牙利', '加拿大', '加德满都', '河内', '河北', '河南', '法兰克福', '法国', '波兰', '波士顿', '波尔图', '泰国', '洛杉矶', '利瓦绿洲', '毛里求斯.1', '凯恩斯', '海南', '冰岛', '渥太华', '温哥华', '温莎', '内蒙古', '湖南', '俄罗斯', '澳门', '熊本市', '爱丁堡', '爱尔兰', '因特拉肯', '汉堡', '毛里求斯', '戛纳', '新西兰', '华沙', '拉科鲁尼亚', '挪威', '摩洛哥', '摩纳哥', '斐济', '斯克兰顿（宾夕法尼亚州）', '斯德哥尔摩', '华欣', '新加坡.1', '新北', '千叶市', '新疆', '新竹', '日内瓦', '比利时', '千叶', '北海道--登别', '星野度假村', '北海道--札幌', '暹粒', '曼彻斯特', '北海道--旭川', '北海道--小樽', '柏林', '北海道--函馆', '格拉纳达', '楠迪', '槟城', '横滨', 'season_4'
#
#          # '老挝', '芬兰', '葡萄牙', '爱尔兰', '荷兰', '澳大利亚', '湖南', '湖北', '海南', '瑞士', 'province_e_28.0', '西藏', 'age_e_1.0', 'province_e_5.0', 'province_e_7.0', 'province_e_8.0', 'province_e_9.0', 'province_e_10.0', 'province_e_12.0', 'province_e_13.0', 'province_e_14.0', 'province_e_15.0', 'age_e_2.0', 'province_e_18.0', 'province_e_21.0', 'continent_e_4.0', 'province_e_23.0', 'gender_e_1.0', 'province_e_25.0', 'province_e_26.0', 'province_e_27.0', 'province_e_0.0', 'province_e_29.0', 'province_e_30.0', '青海', '重庆', '越南', '波兰', '中国澳门', '河南', 'age_00', 'tag_私聊不回复', 'tag_行前未主动联系', 'tag_行前联系不上', 'tag_行程有点紧', 'tag_言语粗鲁', 'tag_路线不熟', 'tag_路线熟悉', 'tag_车内有异味', 'tag_车况有点旧', 'tag_车况脏乱', 'tag_车旧/脏', 'tag_车辆和订单显示不符', 'tag_非订单车辆', 'tag_驾驶平稳', 'tag_驾驶技术一般', 'tag_驾驶鲁莽', 'age_80', 'tag_留学生', 'tag_没做景点介绍', 'tag_沉默寡言', 'tag_举牌迎接', 'histord_sum_cont5', 'histord_sum_cont1', 'tag_不协助搬运行李', 'tag_不懂普通话', 'tag_不爱讲话', 'tag_临时换司导', 'tag_主动热情', 'tag_建筑师', 'tag_未提前联系', 'tag_接待地点不合理', 'tag_提前就终止了服务', 'tag_提前结束行程', 'tag_景点不介绍', 'tag_景点不熟悉', 'tag_景点介绍详尽', 'tag_未举牌服务', 'age_60', 'actiontypeproplast20_9', '河北', 'actiontypeprop_9', '奥地利', '宁夏', '安徽', '尼泊尔', '山西', '巴西', '广东', '广西', '挪威', '捷克', '摩洛哥', '新疆', '新西兰', '柬埔寨', '比利时', '毛里求斯.1', '江苏', '天津', '墨西哥', '埃及', 'action_56789_count_c', 'actiontimespancount_8_9', 'actiontime_last_1_year', 'action_78_count', 'action_789_count', 'action_678_count', 'action_6789_count', 'action_5678_count', 'action_56789_count', '卡塔尔', '中国', '丹麦', '俄罗斯', '内蒙古', '加拿大', '匈牙利', '南非', 'tag_行程安排不合理',
#          # 'continent_e_1.0', 'action_6789_count_c', 'action_5678_count_c', 'action_678_count_c', 'continent_e_5.0', 'province_e_20.0', 'season_1', 'tag_没水/空调不足', 'province_e_17.0', '斐济', 'tag_司导态度差', 'count_all', 'age_e_3.0', '新加坡.1', '黑龙江', '陕西', 'actiontype_lasttime_9', '菲律宾', '冰岛', '缅甸', 'tag_额外收费', '浙江', '德国', '福建',
#
#          # 'timespanmean_last_4', 'timespanmean_last_7', 'timespanmean_last_8', 'timespanmean_last_9',
#          # 'actiontype_last_8', 'actiontype_last_9', 'actiontype_last_10', 'actiontype_last_11', 'actiontype_last_12', 'actiontype_last_13', 'actiontype_last_14', 'actiontype_last_15', 'actiontype_last_16', 'actiontype_last_17', 'actiontype_last_18', 'actiontype_last_19', 'actiontype_last_20',
#          # 'actiontime_last_8', 'actiontime_last_9', 'actiontime_last_10', 'actiontime_last_11', 'actiontime_last_12', 'actiontime_last_13', 'actiontime_last_14', 'actiontime_last_15', 'actiontime_last_16', 'actiontime_last_17', 'actiontime_last_18', 'actiontime_last_19', 'actiontime_last_20',
#          # 'histord_time_last_2', 'histord_time_last_2_year', 'histord_time_last_2_month', 'histord_time_last_3', 'histord_time_last_3_year', 'histord_time_last_3_month', '新竹', '布里斯班', '丹佛', '成田市', '莫斯科', '维多利亚', '上海', '雷克雅未克', '开普敦', '京都', '墨西哥城', '牛津', '阿姆斯特丹', '都柏林', '东京', '珀斯', '华盛顿', '卡萨布兰卡', '亚特兰大', '巴尔的摩', '法兰克福', '富山市', '惠灵顿', '萨尔茨堡', '富国岛', '戛纳', '仰光', '名古屋', '台南', '堪培拉', '冲绳--那霸', '新加坡', '马赛', '达拉斯', '垦丁', '北海道--札幌', '马尼拉', '温哥华', '澳门', '苏梅岛', '波尔多', '斯德哥尔摩', '温莎', '巴伦西亚', '渥太华', '里昂', '迈阿密', '卢塞恩', '西雅图', '奥兰多', '岐阜县', '汉密尔顿', '波士顿', '尼斯', '那不勒斯', '拉科鲁尼亚', '多哈', '华沙', '云顶高原', '横滨', '圣地亚哥', '考文垂', '福冈', '霍巴特', '台中', '墨尔本', '冲绳市', '毛里求斯', '洛杉矶', '蒙特利尔', '曼彻斯特', '嘉义', '大叻', '伊尔库茨克', '楠迪', '阿里山', '布拉格', '沙巴--亚庇', '日内瓦', '蒂卡波湖', '里斯本', '巴黎', '釜山', '洞爷湖', '静冈县', '普吉岛', '台北', '夏威夷欧胡岛（檀香山）', '科茨沃尔德', '科隆', '夏威夷大岛', '开罗', '凯恩斯', '圣保罗', '威尼斯', '旧金山', '奈良', '赫尔辛基', '哥本哈根', '阿布扎比', '御殿场市', '万象', '哈尔施塔特', '富良野', '阿德莱德', '香港', '岘港', '金泽', '高雄', '巴厘岛', '谢菲尔德', '罗马', '哥德堡', '济州岛', '北海道--登别', '马拉加', '华欣', '塞班岛', '塞维利亚', '北海道--旭川', '吉隆坡', '新山', '奥克兰', '马德里', '斯克兰顿（宾夕法尼亚州）', '槟城', '格拉纳达', '伦敦', '黄金海岸', '皇后镇', '轻井泽', '暹粒', '广岛', '大西洋城', '汉堡', '神户', '河内', '布法罗', '福森', '巴塞罗那', '富士河口湖', '宜兰', '多伦多', '夏威夷茂宜岛', '胡志明市', '纽约', '星野度假村', '熊本市', '约克', '布达佩斯', '奥斯陆', '盐湖城', '杜塞尔多夫', '阿维尼翁', '爱丁堡', '千叶市', '维也纳', '雅典', '大阪', '圣彼得堡', '清迈', '巴斯', '芭堤雅', '芝加哥', '富士山', '贝尔法斯特', '费城', '迪拜', '基督城', '新北', '加德满都', '底特律', '柏林', '箱根', '金边', '花莲', '米兰', '曼谷', '利瓦绿洲', '甲米', '佛罗伦萨', '因特拉肯', '慕尼黑', '波尔图', '约翰内斯堡', '苏黎世', '北海道--小樽', '休斯敦', '拉斯维加斯', '布鲁塞尔', '千叶', '阿尔勒', '摩纳哥', '长崎', '悉尼', '芽庄', '卡尔加里', '兰卡威', '首尔', '雅加达', '宿务', '北海道--函馆', '伊斯坦布尔', '伯明翰', '南投', 'action_sum', 'rating_min', 'rating_last', 'gender_exist', 'gender_male', 'gender_female'
#          ]]

# feature = [x for x in dataset.columns if x not in ['userid', 'gender', 'province', 'age',
#          'orderType', 'actionTime_future', 'action_time_future', 'orderTime', 'order_time',
#          'time_1', 'time_2', 'time_2_c',
#          # 小于0.1 97149
#          'province_e_17.0', '皇后镇', 'province_e_18.0', 'province_e_30.0', 'gender_e_1.0', '波尔图', '盐湖城', '神户', '波士顿', '黑龙江', '波兰', 'province_e_14.0', '福建', '黄金海岸', '法兰克福', '瑞士', '牛津', '波尔多', 'province_e_29.0', '渥太华', 'province_e_0.0', 'age_e_2.0', 'province_e_1.0', 'age_e_1.0', '温莎', 'province_e_5.0', 'province_e_6.0', '湖南', '福森', '洞爷湖', '澳大利亚', '澳门', '熊本市', 'province_e_7.0', '爱丁堡', 'province_e_8.0', 'age_e_3.0', '珀斯', 'province_e_15.0', '约翰内斯堡', '科茨沃尔德', '科隆', '马尼拉', '西藏', '谢菲尔德', '贝尔法斯特', '静冈县', '赫尔辛基', '越南', '轻井泽', '霍巴特', '达拉斯', '迈阿密', '雷克雅未克', '雅加达', '陕西', '那不勒斯', '阿里山', '都柏林', '里斯本', '阿维尼翁', '重庆', '金泽', '金边', '釜山', '长崎', '阿姆斯特丹', '蒙特利尔', '蒂卡波湖', '葡萄牙', '考文垂', '箱根', 'province_e_20.0', 'province_e_21.0', '阿尔勒', '马赛', '马来西亚', 'province_e_23.0', '维多利亚', '缅甸', 'province_e_26.0', '老挝', '胡志明市', '萨尔茨堡', 'continent_e_4.0', 'continent_e_5.0', '芬兰', '芭堤雅', '花莲', 'season_2', '芽庄', '马拉加', '荷兰', '莫斯科', '菲律宾', 'province_e_16.0', '匈牙利', '河内', 'tag_非订单车辆', '京都', '亚特兰大', '丹麦', '丹佛', '中国澳门', '中国', 'action_56789_count', 'action_56789_count_c', 'action_5678_count', 'action_6789_count', 'action_678_count', 'action_678_count_c', 'action_789_count', 'action_78_count', 'tag_驾驶鲁莽', 'tag_驾驶技术一般', 'tag_额外收费', '仰光', '伊尔库茨克', '休斯敦', '北海道--登别', '卡塔尔', '南非', '南投', '华沙', '华欣', '千叶市', '千叶', '北海道--旭川', '伯明翰', '北海道--小樽', '北海道--函馆', '加德满都', '利瓦绿洲', '冲绳市', '冰岛', '俄罗斯', 'tag_频繁催促', 'tag_车辆和订单显示不符', '沙巴--亚庇', 'tag_车旧/脏', 'tag_建筑师', 'tag_临时换司导', 'tag_不爱讲话', 'tag_不懂普通话', 'rating_min', 'actiontime_last_1_year', 'latest_7day_actionType9_count', 'latest_6day_actionType9_count', 'latest_6day_actionType7_count', 'latest_5day_actionType9_count', 'actiontimespancount_8_9', 'latest_2day_actionType8_count', 'histord_time_last_3_year', 'histord_time_last_2_year', 'histord_sum_cont5', 'histord_sum_cont1', 'age_00', 'tag_接待地点不合理', 'tag_提前就终止了服务', 'tag_提前结束行程', 'tag_行前联系不上', 'tag_车况有点旧', 'tag_车内有异味', 'tag_路线熟悉', 'tag_路线不熟', 'tag_言语粗鲁', 'tag_行程有点紧', 'tag_行程安排不合理', 'tag_行前未主动联系', 'tag_景点不介绍', 'tag_私聊不回复', 'tag_留学生', 'tag_没水/空调不足', 'tag_没做景点介绍', 'tag_沉默寡言', 'tag_未举牌服务', 'tag_景点不熟悉', '卡萨布兰卡', '卢塞恩', '台中', '台南', '摩纳哥', '摩洛哥', '捷克', '挪威', '戛纳', '成田市', '慕尼黑', '惠灵顿', '德国', '御殿场市', '开罗', '开普敦', '广岛', '布鲁塞尔', '布里斯班', '布达佩斯', '布拉格', '斐济', '斯克兰顿（宾夕法尼亚州）', '斯德哥尔摩', '柬埔寨', '汉密尔顿', '汉堡', '毛里求斯.1', '毛里求斯', '比利时', '横滨', '格拉纳达', '柏林', '新加坡.1', '曼彻斯特', '暹粒', '星野度假村', '旧金山', '新竹', '新疆', '新北', '巴西', '巴伦西亚', '岘港', '垦丁', '墨西哥', '墨尔本', '塞维利亚', '塞班岛', '堪培拉', '基督城', '埃及', '圣彼得堡', '夏威夷大岛', '圣地亚哥', '因特拉肯', '嘉义', '哥本哈根', '哥德堡', '哈尔施塔特', '名古屋', '墨西哥城', '夏威夷茂宜岛', '岐阜县', '宿务', '尼泊尔', '尼斯', '富良野', '富山市', '富士河口湖', '富士山', '富国岛', '宜兰', '多伦多', '宁夏', '奥斯陆', '奥地利', '奈良', '大西洋城', '大叻', '多哈', 'season_4'
#          , 'action_5678_count_c', '拉科鲁尼亚', '万象', '上海', '中国台湾', '巴斯', 'tag_未提前联系', '巴尔的摩', '奥兰多', 'province_e_2.0', 'province_e_27.0', 'province_e_28.0', '云顶高原', 'continent_e_1.0', 'tag_不协助搬运行李', 'actiontypeproplast20_7', '新西兰', 'latest_6_actionType', 'tag_驾驶平稳', '苏梅岛', '爱尔兰', '英国', '费城', 'hasprovince', '里昂', '阿联酋', 'action_7', '雅典', '韩国', '首尔', '香港', 'latest_6day_actionType8_count', '楠迪', '米兰', '圣保罗'
#          , 'city_count_var', 'driver_count_var', 'type1_count_var'
#           ]]

feature = [x for x in dataset.columns if x not in ['userid', 'gender', 'province', 'age',
         'orderType', 'actionTime_future', 'action_time_future', 'orderTime', 'order_time',
         'time_1', 'time_2', 'time_2_c', 'gender_e', 'age_e', 'province_e', 'continent_e', 'season',

         '莫斯科', '胡志明市', 'tag_举牌迎接', 'tag_额外收费', '芭堤雅', '花莲', '芽庄', '苏梅岛', 'tag_驾驶平稳', 'province_e_23.0', 'tag_建筑师', 'tag_驾驶技术一般', 'tag_留学生', 'province_e_21.0', 'province_e_26.0', '菲律宾', '萨尔茨堡', 'province_e_20.0', '蒂卡波湖', '蒙特利尔', '西藏', 'tag_驾驶鲁莽', 'tag_接待地点不合理', '谢菲尔德', '贝尔法斯特', '考文垂', 'tag_临时换司导', '老挝', '科茨沃尔德', 'tag_车况有点旧', '瑞士', 'rating_min', 'tag_车旧/脏', 'continent_e_5.0', '盐湖城', '神户', 'tag_不协助搬运行李', 'continent_e_2.0', 'tag_非订单车辆', '福森', '科隆', 'tag_提前就终止了服务', '箱根', 'tag_频繁催促', '约克', '约翰内斯堡', 'province_e_30.0', 'province_e_29.0', 'province_e_28.0', '维多利亚', 'province_e_27.0', '缅甸', 'tag_不爱讲话', 'tag_路线不熟', '费城', 'age_e_3.0', 'province_e_5.0', 'province_e_10.0', 'tag_行前联系不上', 'province_e_8.0', '雅典', '雅加达', '雷克雅未克', '霍巴特', 'province_e_7.0', '静冈县', 'province_e_6.0', '首尔', '香港', '赫尔辛基', '马拉加', 'tag_未提前联系', '马赛', 'tag_行前未主动联系', '黄金海岸', '黑龙江', 'tag_沉默寡言', 'province_e_2.0', 'province_e_1.0', 'province_e_0.0', 'tag_没水/空调不足', 'tag_景点不熟悉', '阿里山', '珀斯', '阿维尼翁', 'tag_言语粗鲁', '越南', 'province_e_15.0', '轻井泽', 'action_56789_count_c', '达拉斯', '迈阿密', 'tag_提前结束行程', '那不勒斯', '都柏林', '里斯本', '里昂', 'province_e_14.0', 'actiontimespancount_8_9', 'tag_行程有点紧', '金泽', '金边', '釜山', '长崎', '阿姆斯特丹', '阿尔勒', 'province_e_12.0', 'tag_行程安排不合理', '北海道--小樽', 'action_6789_count', '爱尔兰', '俄罗斯', '宿务', '宜兰', '伯明翰', '奥斯陆', '奥兰多', '奈良', '冰岛', '爱丁堡', '大西洋城', '大叻', '多哈', '夏威夷茂宜岛', '夏威夷大岛', '墨西哥城', '富国岛', '富士山', '富士河口湖', '富山市', '富良野', '尼斯', '尼泊尔', '休斯敦', '岐阜县', '岘港', '巴伦西亚', '巴厘岛', '伊尔库茨克', '巴尔的摩', '巴斯', '巴西', '布达佩斯', '墨西哥', '冲绳市', '塞维利亚', '台南', '北海道--登别', '千叶', '千叶市', '华欣', '华沙', '北海道--函馆', '南投', '南非', '卡塔尔', '卡萨布兰卡', '卢塞恩', '印度尼西亚', '台中', '台北', '匈牙利', '塞班岛', '加德满都', '名古屋', '哈尔施塔特', '哥德堡', '嘉义', '利瓦绿洲', '因特拉肯', '圣保罗', '圣地亚哥', '圣彼得堡', '垦丁', '埃及', '基督城', '堪培拉', '布里斯班', '布鲁塞尔', '京都', '波士顿', '楠迪', '横滨', '比利时', '毛里求斯', '毛里求斯.1', '汉堡', '汉密尔顿', 'continent_driver_count_avg', 'histord_sum_cont4', '河内', 'histord_sum_cont5', 'histord_time_last_2_year', '法兰克福', '波兰', '波尔图', '柬埔寨', 'action_5678_count', 'histord_time_last_3_year', '洞爷湖', 'latest_1day_actionType9_count', '北海道--旭川', 'action_678_count', 'action_678_count_c', '渥太华', '温莎', 'age_00', 'action_789_count', 'action_78_count', '澳门', '熊本市', '格拉纳达', '柏林', '亚特兰大', '捷克', '广岛', '云顶高原', 'tag_车内有异味', '开普敦', '开罗', '御殿场市', '德国', '惠灵顿', '慕尼黑', '成田市', '戛纳', '丹佛', '拉科鲁尼亚', '挪威', '摩洛哥', 'histord_sum_all', '摩纳哥', '斯克兰顿（宾夕法尼亚州）', '中国香港', '新加坡.1', '新北', '新疆', '新竹', '新西兰', '中国澳门', '中国', '星野度假村', '上海', '暹粒', '曼彻斯特', '万象'
         , 'latest_2day_actionType9_count', 'continent_e_1.0', 'continent_type1_count_avg', 'continent_e_4.0', '高雄', 'province_e_17.0', 'gender_e_1.0', 'latest_7day_actionType9_count', 'age_e_1.0', 'province_e_25.0', 'latest_5day_actionType8_count', 'province_e_9.0', 'province_e_22.0', 'province_e_13.0', '韩国', 'latest_6day_actionType9_count', '宁夏', '青海', 'action_5678_count_c', '奥地利', '布拉格', '多伦多', '哥本哈根', '斐济', '斯德哥尔摩', '北海道--札幌', '槟城', '佛罗伦萨', '丹麦', '波尔多', 'action_56789_count', '海南', '牛津', 'rating_mean', '皇后镇', 'tag_车辆和订单显示不符', 'tag_车况脏乱', '米兰', 'tag_私聊不回复', 'tag_没做景点介绍', '芬兰', '英国', '荷兰', '葡萄牙', 'tag_未举牌服务', 'tag_景点不介绍', 'tag_主动热情', 'tag_路线熟悉'
          ]]


print("训练集：", train.shape)
print("测试集：", test.shape)

# train1 = train[train.count_all.isnull()]   # count_all为空   历史订单中无对应的
# train2 = train[train.count_all.notnull()]  # count_all不为空 历史订单中有对应的

# train1.fillna(-999, inplace=True)
# train2.fillna(-999, inplace=True)
# test.fillna(-999,inplace=True)

# train1_x, val1_x, train1_y, val1_y = train_test_split(train1[feature], train1["orderType"], test_size=0.2, random_state=1017, stratify = train1["orderType"])
# train2_x, val2_x, train2_y, val2_y = train_test_split(train2[feature], train2["orderType"], test_size=0.2, random_state=1017, stratify = train2["orderType"])
#
# val_x = val1_x.append(val2_x, ignore_index=True)
# val_y = val1_y.append(val2_y, ignore_index=True)
# train_x = train1_x.append(train2_x, ignore_index=True)
# train_y = train1_y.append(train2_y, ignore_index=True)

train.fillna(train.min(), inplace=True)
test.fillna(test.min(), inplace=True)
# train_x, val_x, train_y, val_y = train_test_split(train[feature], train["orderType"], test_size=0.2, random_state=0, stratify=train["orderType"])

# print(val_x.shape)
# print(val_y.shape)
# print(train_x.shape)
# print(train_y.shape)

# # specify the training parameters
k = '78'
x = '10'

X_train = np.array(train[feature])
y_train = np.array(train["orderType"])
X_test = np.array(test[feature])

X_train = np.nan_to_num(X_train)
X_train = preprocessing.scale(X_train)
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
y_train = preprocessing.scale(y_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
X_test = preprocessing.scale(X_test)
X_test = np.nan_to_num(X_test)

# prediction model
np.random.seed(314)
model_rf = ensemble.RandomForestClassifier(n_estimators=200, oob_score=True)
model_rf.fit(X_train, y_train)
y_train_pred = model_rf.predict(X_train)
y_train_predprob = model_rf.predict_proba(X_train).reshape((-1))
importances = model_rf.feature_importances_

print('score_AUC:', round(metrics.roc_auc_score(y_train, y_train_predprob), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, y_train_pred), 5))
scores_cross = model_selection.cross_val_score(model_rf, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

# test predict error obj
val_predict_prob = model_rf.predict_proba(train[feature])
val_preds_class = model_rf.predict(train[feature])
val_prob = train[['userid', 'count_all', 'count_1', 'ever_1', 'action_all', 'action_all_c']]
val_prob['pred_orderType_prob'] = val_predict_prob[:, 1]
val_prob['pred_orderType'] = val_preds_class
val_prob['orderType'] = train["orderType"]
val_prob['error'] = 0
val_prob['error'][val_prob['orderType'] != val_prob['pred_orderType']] = 1

val_prob = pd.DataFrame(val_prob)
print("验证集预测完成：", datetime.now())
val_prob.to_csv(dire + 'submit/RF/val/RF_val' + k + '_' + x + '.csv', index=False)

print("val_prob总个数：", len(val_prob))
count = val_prob['orderType'].value_counts()
print(count)
print("预测为1总个数：", len(val_prob[val_prob['pred_orderType'] == 1]))
print("预测为0总个数：", len(val_prob[val_prob['pred_orderType'] == 0]))
print("train错误总个数：", len(val_prob[(val_prob['error'] == 1)]))
count = val_prob[(val_prob['error'] == 1)]['pred_orderType'].value_counts()
print(count)

# make the prediction using the resulting model
predict_prob = model_rf.predict_proba(test[feature])
preds_class = model_rf.predict(test[feature])
print("1的数目：", np.sum(preds_class == 1))

prob = test[['userid']]
prob['orderType'] = predict_prob[:, 1]
sub = pd.DataFrame(prob)
print("训练完成：", datetime.now())
sub.to_csv(dire + 'submit/RF/sub_RF_prob' + k + '_' + x + '.csv', index=False)

importances = model_rf.feature_importances_
df_featImp = pd.DataFrame({'tags': feature, 'importance': importances})
df_featImp_sorted = df_featImp.sort_values(by=['importance'], ascending=False)
df_featImp_sorted.plot(x='tags', y='importance', kind='bar')
df_featImp_sorted.to_csv(dire + 'submit/RF/feature/feat_RF' + k + '_' + x + '.csv')
