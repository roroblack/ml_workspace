import seaborn as sns
import pandas as pd

# Seaborn에서 제공하는 titanic 데이터셋 로드
titanic = sns.load_dataset('titanic')
# 데이터 구조 확인
print('----------------------------------------0-0')
print('titanic info : ')
print(titanic.info())

# 데이터셋의 결측치 확인
print('----------------------------------------0-1')
print('Original Data:')
print(titanic.isnull().sum())

# 1. 결측치가 있는 행(row) 삭제
print('----------------------------------------1')
titanic_dropped = titanic.dropna()
print('\nAfter Dropping Rows with Missing Values:')
print(titanic_dropped.isnull().sum())

# 2. 평균값으로 결측치 대체 (age 열에 대해)
print('----------------------------------------2')
print('age mean : ', titanic['age'].mean())
titanic['age_mean_filled'] = titanic['age'].fillna(titanic['age'].mean())
print('\nAfter Filling Missing Values with Mean in age Column:')
print(titanic['age_mean_filled'].isnull().sum())

# 3. 중앙값으로 결측치 대체 (fare 열에 대해)
print('----------------------------------------3')
titanic['fare_median_filled'] = titanic['fare'].fillna(titanic['fare'].median())
print('\nAfter Filling Missing Values with Median in fare Column:')
print(titanic['fare_median_filled'].isnull().sum())

# 4. 최빈값으로 결측치 대체 (embarked 열에 대해)
print('----------------------------------------4')
titanic['embarked_mode_filled'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
print('\nAfter Filling Missing Values with Mode in embarked Column:')
print(titanic['embarked_mode_filled'].isnull().sum())

# 5. 전진 채우기 (Forward Fill) (age 열에 대해)
print('----------------------------------------5')
titanic['age_ffill'] = titanic['age'].ffill()
print('\nAfter Forward Fill in age Column:')
print(titanic['age_ffill'].isnull().sum())

# 6. 후진 채우기 (Backward Fill) (age 열에 대해)
print('----------------------------------------6')
titanic['age_bfill'] = titanic['age'].bfill()
print('\nAfter Backward Fill in age Column:')
print(titanic['age_bfill'].isnull().sum())
