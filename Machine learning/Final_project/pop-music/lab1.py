import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取用户行为数据
user_behavior = pd.read_csv('mars_tianchi_user_actions.csv')
# 读取歌曲艺人数据
song_artist = pd.read_csv('mars_tianchi_songs.csv')

# 转换日期格式
user_behavior['record_date'] = pd.to_datetime(user_behavior['record_date'], format='%Y%m%d')
song_artist['publish_time'] = pd.to_datetime(song_artist['publish_time'], format='%Y%m%d')

# 合并数据集
data = pd.merge(user_behavior, song_artist, on='song_id')

# 过滤过去6个月的数据
cutoff_date = pd.to_datetime('20150301', format='%Y%m%d')
filtered_data = data[data['record_date'] >= cutoff_date]

# 提取日期相关特征
filtered_data['year'] = filtered_data['record_date'].dt.year
filtered_data['month'] = filtered_data['record_date'].dt.month
filtered_data['day'] = filtered_data['record_date'].dt.day

# 聚合数据，计算每个艺人每天的播放数
artist_daily_play_count = filtered_data[filtered_data['user_behavior'] == 1].groupby(['artist_id', 'record_date']).size().reset_index(name='play_count')

# 数据平滑处理
artist_daily_play_count['play_count_smooth'] = artist_daily_play_count.groupby('artist_id')['play_count'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# 准备数据
artist_forecast = pd.DataFrame()
for artist in artist_daily_play_count['artist_id'].unique():
    artist_data = artist_daily_play_count[artist_daily_play_count['artist_id'] == artist].set_index('record_date')
    artist_data = artist_data.asfreq('D').fillna(0)  # 填补缺失值
    artist_data['play_count'] = artist_data['play_count_smooth']  # 使用平滑后的播放量数据

    # 创建特征
    artist_data['day_of_week'] = artist_data.index.dayofweek
    artist_data['day_of_month'] = artist_data.index.day
    artist_data['month'] = artist_data.index.month
    artist_data['lag_1'] = artist_data['play_count'].shift(1).fillna(0)
    artist_data['lag_7'] = artist_data['play_count'].shift(7).fillna(0)
    artist_data['lag_30'] = artist_data['play_count'].shift(30).fillna(0)

    X = artist_data[['day_of_week', 'day_of_month', 'month', 'lag_1', 'lag_7', 'lag_30']]
    y = artist_data['play_count']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 创建并训练模型
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # 预测未来60天
    last_row = X.iloc[-1].values.reshape(1, -1)
    predictions = []
    for _ in range(60):
        prediction = model.predict(last_row)
        predictions.append(prediction[0])
        # 更新last_row以便进行下一次预测
        last_row = np.roll(last_row, -1)
        last_row[0, -1] = prediction

    # 创建预测结果
    forecast_dates = pd.date_range(start='2015-09-01', periods=60)
    forecast_df = pd.DataFrame({'date': forecast_dates,
                                'predicted_play_count': predictions,
                                'artist_id': artist})

    artist_forecast = pd.concat([artist_forecast, forecast_df], ignore_index=True)

# 重置索引
artist_forecast.reset_index(drop=True, inplace=True)

# 将预测播放数四舍五入到整数
artist_forecast['predicted_play_count'] = artist_forecast['predicted_play_count'].round().astype(int)

# 输出结果
artist_forecast.to_csv('artist_forecast_xgboost.csv', index=False)



