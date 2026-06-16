import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Seaborn에서 제공하는 tips 데이터셋 로드
tips = sns.load_dataset('tips')

# 라벨 인코더 생성
label_encoder = LabelEncoder()

# 범주형 변수들에 라벨 인코딩(숫자를 부여함 0, 1, 2, ...) 적용
tips['sex_encoded'] = label_encoder.fit_transform(tips['sex'])
tips['smoker_encoded'] = label_encoder.fit_transform(tips['smoker'])
tips['day_encoded'] = label_encoder.fit_transform(tips['day'])
tips['time_encoded'] = label_encoder.fit_transform(tips['time'])

# 인코딩된 데이터 확인
print(tips[['sex', 'sex_encoded', 'smoker', 'smoker_encoded', 'day', 'day_encoded', 'time', 'time_encoded']].head())
