import seaborn as sns
import numpy as np

# Seaborn에서 제공하는 tips 데이터셋 로드
tips = sns.load_dataset('tips')

# IQR 계산 함수
def calculate_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

# total_bill 열에 대해 IQR 계산
lower_bound, upper_bound = calculate_iqr(tips['total_bill'])

# 이상치 탐지 (IQR 기준을 벗어나는 값들)
outliers = tips[(tips['total_bill'] < lower_bound) | (tips['total_bill'] > upper_bound)]

# 이상치 처리 (이상치를 중앙값으로 대체)
median_total_bill = tips['total_bill'].median()
tips.loc[(tips['total_bill'] < lower_bound) | (tips['total_bill'] > upper_bound), 'total_bill'] = median_total_bill

# 결과 출력
print('발견된 이상치:')
print(outliers[['total_bill']])
print('\n이상치 처리 후 데이터:')
print(tips[['total_bill']].head(10))
