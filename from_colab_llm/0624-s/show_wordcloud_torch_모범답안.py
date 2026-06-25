# ============================================
# PyTorch 기반 한국어 워드 클라우드 예시
# ============================================

# 필요한 패키지 설치
# Colab 또는 Jupyter Notebook에서 처음 한 번만 실행합니다.
# !pip install torch pandas matplotlib wordcloud konlpy

# KoNLPy의 Okt가 Java JVM을 필요로 함
# https://jdk.java.net/archive/
# jdk 21 zip 파일 다운받음 > 압축풀어줌
# jdk-21.0.2 폴더를 복사해서 C:\Program Files\Java\jdk-21.0.2 위치가 되게 붙여넣기함
# 환경변수에 다음을 추가합니다.
# JAVA_HOME = C:\Program Files\Java\jdk-21.0.2
# 그리고 Path에 다음을 추가합니다.
# %JAVA_HOME%\bin
# 설정 후 PyCharm을 완전히 종료한 뒤 다시 실행합니다.

# 정규표현식을 사용하기 위한 re 모듈을 불러옵니다.
import re

# 단어 빈도 계산을 쉽게 하기 위해 Counter를 불러옵니다.
from collections import Counter

# PyTorch 텐서 처리를 위해 torch를 불러옵니다.
import torch

# 표 형태 데이터 처리를 위해 pandas를 불러옵니다.
import pandas as pd

# 그래프 출력을 위해 matplotlib을 불러옵니다.
import matplotlib.pyplot as plt

# 워드 클라우드 생성을 위해 WordCloud를 불러옵니다.
from wordcloud import WordCloud

# 한국어 형태소 분석을 위해 Okt 형태소 분석기를 불러옵니다.
from konlpy.tag import Okt


# --------------------------------------------
# 1. 예제 텍스트 준비
# --------------------------------------------

# 실제 수업에서는 txt 파일을 open()으로 읽을 수 있지만,
# 여기서는 코드가 바로 실행되도록 예제 문장을 직접 작성합니다.
text = """
인공지능은 데이터를 학습하여 예측과 분류를 수행한다.
딥러닝은 인공지능의 중요한 분야이며, 이미지 분석과 자연어 처리에 많이 사용된다.
자연어 처리는 문장을 분석하고 단어의 의미를 이해하는 기술이다.
워드 클라우드는 텍스트에서 자주 등장하는 단어를 크게 보여주는 시각화 방법이다.
데이터 분석에서는 단어 빈도를 계산하고 중요한 키워드를 찾는 과정이 필요하다.
머신러닝과 딥러닝 모델은 데이터 품질에 따라 성능이 크게 달라진다.
"""


# --------------------------------------------
# 2. 정규표현식으로 불필요한 문자 제거
# --------------------------------------------

# 한글과 공백을 제외한 모든 문자를 제거하는 정규표현식 패턴입니다.
# ^는 제외를 의미하고, ㄱ-ㅎㅏ-ㅣ가-힣은 한글 범위를 의미합니다.
clean_text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\s]", " ", text)

# 여러 개의 공백을 하나의 공백으로 정리합니다.
clean_text = re.sub(r"\s+", " ", clean_text).strip()

# 정제된 텍스트를 확인합니다.
print("정제된 텍스트:")
print(clean_text)


# --------------------------------------------
# 3. Okt 형태소 분석기 생성
# --------------------------------------------

# Okt 객체를 생성합니다.
# Okt는 한국어 문장에서 명사, 동사, 형용사 등을 분리할 수 있습니다.
okt = Okt()


# --------------------------------------------
# 4. 명사 추출
# --------------------------------------------

# nouns() 함수는 문장에서 명사만 추출합니다.
nouns = okt.nouns(clean_text)

# 한 글자 단어는 의미가 약한 경우가 많기 때문에 두 글자 이상만 남깁니다.
nouns = [word for word in nouns if len(word) > 1]

# 추출된 명사를 확인합니다.
print("\n추출된 명사:")
print(nouns)


# --------------------------------------------
# 5. 단어를 숫자 ID로 변환
# --------------------------------------------

# 중복을 제거한 단어 목록을 정렬하여 vocabulary를 만듭니다.
vocab = sorted(set(nouns))

# 단어를 숫자 인덱스로 바꾸기 위한 딕셔너리를 만듭니다.
word_to_id = {word: idx for idx, word in enumerate(vocab)}

# 각 명사를 숫자 ID로 변환합니다.
word_ids = [word_to_id[word] for word in nouns]

# 숫자 ID 리스트를 PyTorch 텐서로 변환합니다.
word_tensor = torch.tensor(word_ids, dtype=torch.long)


# --------------------------------------------
# 6. PyTorch로 단어 빈도 계산
# --------------------------------------------

# torch.bincount()는 각 숫자 ID가 몇 번 나왔는지 계산합니다.
word_counts_tensor = torch.bincount(word_tensor)

# PyTorch 텐서를 파이썬 딕셔너리 형태로 변환합니다.
word_freq = {
    vocab[i]: int(word_counts_tensor[i].item())
    for i in range(len(vocab))
}

# 빈도수가 높은 순서대로 정렬합니다.
word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

# 빈도 결과를 출력합니다.
print("\n단어 빈도:")
print(word_freq)


# --------------------------------------------
# 7. pandas로 상위 단어 확인
# --------------------------------------------

# 단어 빈도 딕셔너리를 pandas Series로 변환합니다.
freq_series = pd.Series(word_freq)

# 상위 10개 단어를 출력합니다.
print("\n상위 단어:")
print(freq_series.head(10))


# --------------------------------------------
# 8. 한글 폰트 경로 설정
# --------------------------------------------

# Windows 사용자는 보통 아래 경로를 사용할 수 있습니다.
# Colab에서는 별도 한글 폰트 설치가 필요할 수 있습니다.
font_path = "C:/Windows/Fonts/malgun.ttf"


# --------------------------------------------
# 9. 워드 클라우드 생성
# --------------------------------------------

# WordCloud 객체를 생성합니다.
wordcloud = WordCloud(
    font_path=font_path,        # 한글 깨짐 방지를 위한 폰트 경로
    width=900,                  # 워드 클라우드 이미지 너비
    height=600,                 # 워드 클라우드 이미지 높이
    background_color="white",   # 배경색
    max_words=100               # 최대 표시 단어 수
).generate_from_frequencies(word_freq)


# --------------------------------------------
# 10. 워드 클라우드 출력
# --------------------------------------------

# 그래프 크기를 설정합니다.
plt.figure(figsize=(12, 8))

# 워드 클라우드 이미지를 화면에 표시합니다.
plt.imshow(wordcloud, interpolation="bilinear")

# 축 눈금을 제거합니다.
plt.axis("off")

# 제목을 설정합니다.
plt.title("PyTorch 기반 단어 빈도 워드 클라우드", fontsize=18)

# 그래프를 출력합니다.
plt.show()