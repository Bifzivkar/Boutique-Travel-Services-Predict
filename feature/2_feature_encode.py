# -*- encoding:utf-8 -*-
# ===================================================================== #
# 特征编码
# gender、province、age、city、country、continent
# ===================================================================== #
import pandas as pd

dire = '../../data/'
orderHistory_train = pd.read_csv(dire + 'train/orderHistory_train.csv', encoding='utf-8')
orderHistory_test = pd.read_csv(dire + 'test/orderHistory_test.csv', encoding='utf-8')
userProfile_train = pd.read_csv(dire + 'train/userProfile_train.csv', encoding='utf-8')
userProfile_test = pd.read_csv(dire + 'test/userProfile_test.csv', encoding='utf-8')

# 性别映射 {'男': 0, '女': 1}
gender_mapping = {lab: idx for idx, lab in enumerate(set(userProfile_train['gender'].dropna()))}
print(gender_mapping)
userProfile_train['gender_e'] = userProfile_train['gender'].map(gender_mapping)
userProfile_test['gender_e'] = userProfile_test['gender'].map(gender_mapping)

# 省份映射
province = userProfile_train['province'].append(userProfile_test['province'], ignore_index=True)
province_mapping = {lab: idx for idx, lab in enumerate(set(province.dropna()))}
print(province_mapping)
userProfile_train['province_e'] = userProfile_train['province'].map(province_mapping)
userProfile_test['province_e'] = userProfile_test['province'].map(province_mapping)

# age映射
age_mapping = {lab: idx for idx, lab in enumerate(set(userProfile_train['age'].dropna()))}
print(age_mapping)
userProfile_train['age_e'] = userProfile_train['age'].map(age_mapping)
userProfile_test['age_e'] = userProfile_test['age'].map(age_mapping)


# city映射
city = orderHistory_train['city'].append(orderHistory_test['city'], ignore_index=True)
city_mapping = {lab: idx for idx, lab in enumerate(set(city.dropna()))}
print(city_mapping)
orderHistory_train['city_e'] = orderHistory_train['city'].map(city_mapping)
orderHistory_test['city_e'] = orderHistory_test['city'].map(city_mapping)

# country映射
country = orderHistory_train['country'].append(orderHistory_test['country'], ignore_index=True)
country_mapping = {lab: idx for idx, lab in enumerate(set(country.dropna()))}
print(country_mapping)
orderHistory_train['country_e'] = orderHistory_train['country'].map(country_mapping)
orderHistory_test['country_e'] = orderHistory_test['country'].map(country_mapping)

# continent映射
continent = orderHistory_train['continent'].append(orderHistory_test['continent'], ignore_index=True)
continent_mapping = {lab: idx for idx, lab in enumerate(set(continent.dropna()))}
print(continent_mapping)
orderHistory_train['continent_e'] = orderHistory_train['continent'].map(continent_mapping)
orderHistory_test['continent_e'] = orderHistory_test['continent'].map(continent_mapping)

# print(userProfile_train)
# print(userProfile_test)
# print(orderHistory_train)
# print(orderHistory_test)

userProfile_train.to_csv(dire + 'train/userProfile_train.csv', index=False, encoding='utf-8')
userProfile_test.to_csv(dire + 'test/userProfile_test.csv', index=False, encoding='utf-8')
orderHistory_train.to_csv(dire + 'train/orderHistory_train.csv', index=False, encoding='utf-8')
orderHistory_test.to_csv(dire + 'test/orderHistory_test.csv', index=False, encoding='utf-8')
