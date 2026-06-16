import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Seaborn에서 제공하는 tips 데이터셋 로드
tips = sns.load_dataset('tips')

# Min-Max 스케일러 생성
scaler = MinMaxScaler()

# total_bill과 tip 열에 대해 Min-Max 스케일링 적용
scaled_data = scaler.fit_transform(tips[['total_bill', 'tip']])

# 스케일링된 결과를 원래 데이터프레임에 추가
tips['total_bill_scaled'] = scaled_data[:, 0]
tips['tip_scaled'] = scaled_data[:, 1]

# 결과 출력
print(tips.head())
