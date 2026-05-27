import seaborn as sns
import numpy as np

# Seaborn에서 제공하는 tips 데이터셋 로드
tips = sns.load_dataset('tips')

# Z-스코어 계산 함수
def calculate_z_scores(data):
    return (data - np.mean(data)) / np.std(data)

# total_bill 열에 대해 Z-스코어 계산
tips['total_bill_zscore'] = calculate_z_scores(tips['total_bill'])

# 이상치 탐지 (Z-스코어가 3 이상이거나 -3 이하인 경우)
outliers = tips[np.abs(tips['total_bill_zscore']) > 3]

# 이상치 처리 (이상치를 중앙값으로 대체)
median_total_bill = tips['total_bill'].median()
tips.loc[np.abs(tips['total_bill_zscore']) > 3, 'total_bill'] = median_total_bill

# 처리 후 Z-스코어 다시 계산
tips['total_bill_zscore'] = calculate_z_scores(tips['total_bill'])

# 결과 출력
print('발견된 이상치:')
print(outliers[['total_bill', 'total_bill_zscore']])
print('\n이상치 처리 후 데이터:')
print(tips[['total_bill', 'total_bill_zscore']].head(10))
