import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

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

    # 训练ARIMA模型
    model = ARIMA(artist_data['play_count'], order=(5, 1, 0))  # ARIMA(p, d, q) 参数可以调整
    model_fit = model.fit()

    # 预测未来60天
    forecast = model_fit.forecast(steps=60)
    forecast_dates = pd.date_range(start='2015-09-01', periods=60)
    forecast_df = pd.DataFrame({'date': forecast_dates,
                                'predicted_play_count': forecast,
                                'artist_id': artist})

    artist_forecast = pd.concat([artist_forecast, forecast_df], ignore_index=True)

# 重置索引
artist_forecast.reset_index(drop=True, inplace=True)

# 将预测播放数四舍五入到整数
artist_forecast['predicted_play_count'] = artist_forecast['predicted_play_count'].round().astype(int)

# 输出结果
artist_forecast.to_csv('artist_forecast_arima.csv', index=False)

