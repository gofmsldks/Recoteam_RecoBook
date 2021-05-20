# -*- coding: utf-8 -*-
"""네이버쇼핑_영화리뷰학습모델_(도서데이터도학습모델만들예정).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17UeaQKAYz2yoqz-CWOJWkHmeqlCbQD0D
"""

# 구글드라이브 연동
from google.colab import drive
drive.mount('/gdrive', force_remount=True)

# 구글 드라이브 파일 확인
!ls '/gdrive/MyDrive/likelion_Recoteam_RecoBook_project2/temp/'

# 반복되는 드라이브 경로 변수화
drive_path = '/gdrive/MyDrive/likelion_Recoteam_RecoBook_project2/temp/'

import os
os.chdir(drive_path)
print(os.getcwd())

!pip install nltk
import nltk
nltk.download('treebank')

!pip install konlpy

"""1. 컨텐츠 기반 추천


* 컨텐츠의 정보를 바탕으로 추천(가장 간단)
* ex) 인풋: 사용자가 귀멸의 칼날, 만화1, 만화2를 입력
* 먼저 입력한 책 이름을 바탕으로 도서 데이터 셋에서 장르, 감정점수, 평점을 찾아내고
* 비슷한 요소들을 연관도 분석을 통해 뽑아내고 여기서 유사한 장르, 감정, 평점을 가진 책을 뽑아내기


---------------


2. 사용자가 입력한 감정 요소들을 바탕으로 추천(될지 안될지 모르겠습니다)

* 한줄평을 감정분석하고 감정분석한것을 기준으로 군집화(긍정 군집, 부정 군집...etc)
* 인풋: 사용자가 '감동: 40, 재미: 60'를 원한다고 입력  
* '감동 군집'에서 평점 높은 책 4권 추출, '재미 군집'에서 평점 높은 책 6권 추출.  
* 아웃풋: 추출한 책들 댓글 감정 분석한거 퍼센트 비중 파이차트로 표현 하고 일치하는 책들 보여주기


---------------


3. 아이템 기반 협업 필터링(유저 개인에 특화된 추천방식)

* 입력: 사전에 사용자가 이전에 읽었던 국내 베스트 셀러의 이름을 몇가지 입력하고 긍정 부정 등등의 감정 퍼센트를 입력하게 함.  
* 기존 크롤링한 데이터를 데이터 프레임으로 만듦
*  행: userID , 열: 책 종류 (행열 바뀌어도 무관) 내용은 감정 분석한 점수로  
* 데이터 프레임에 사용자가 입력한 정보들 추가 
*  아웃풋:  surprise를 사용해서 구한 값
* 더 심화하려면 행렬 분해기반 잠재요인 협업 필터링


---------------


* 협업 필터링 (Collaborative Filtering, CF)이란 여러 사용자들로부터 얻은 기호 정보에 따라 다른 사용자들의 관심사를 예측하게 해주는 방법 이라고 정의됩니다.
 
* 협업 필터링에서 중요한 것은 "여러 사용자들로부터 얻은 정보"입니다. 
 
* 협업 필터링에는 크게 [사용자 기반 추천, 아이템 기반 추천] 두 가지가 있습니다.
 
* 1) 사용자 기반 추천 (User-based Recommendation)
비슷한 성향을 지닌 사용자를 기반으로 분석해서 추천해주는 방식입니다.
 
* A라는 사람이 [햄버거, 감자튀김, 콜라]를 구매하고,
* B라는 사람은 [햄버거, 콜라]를 구매하려 한다고 가정해보겠습니다.
 
* 이 둘의 구매목록을 보면 이 둘은 유사하다고 인식되어 B에게 감자튀김을 추천해줍니다.
 
* 2) 아이템 기반 추천 (Item-based Recommendation)
* 이전에 구매했던 아이템을 기반으로 유사한 상품을 추천하는 방식입니다.
 
* 예를 들어 기존에 [햄버거, 감자튀김]이 함께 구매되는 빈도가 많다고 분석되면 이 둘은 유사도가 높다고 판단됩니다.
* 따라서 C라는 사람이 햄버거를 구매하려고 하면 유사한 아이템으로 판단되는 감자튀김을 추천해주는 것입니다.

# 컨텐츠 기반 추천

## 1. 댓글 감정분석

방안1.
* 댓글의 감정분석을 위해서는 미리 학습 시킬 데이터가 필요.
* 1. 네이버 기사 댓글, 뉴스 댓글 사용해서 모델1을 학습
* 2. 학습된 모델1을 사용해서 도서 댓글 정보 라벨링
* 3. 라벨링된 도서 댓글 정보를 train, test로 나누어서 다시 한번 training 시키기


방안2.
* 1. 댓글 데이터와 평점 데이터를 바탕으로 토크나이징 및 k-means 군집화
* 2. 군집화를 통해 라벨링 된 데이터를 바탕으로 학습

# 네이버 영화 리뷰 긍정 부정 모델
"""

# Commented out IPython magic to ensure Python compatibility.
# 방안1

import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="movie_ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="movie_ratings_test.txt")

train_data = pd.read_table('movie_ratings_train.txt')
test_data = pd.read_table('movie_ratings_test.txt')

train_data.info()

print('훈련용 리뷰 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력
print(train_data[:5]) # train 상위 5개 출력

print('---------------------------------------------------------------------------------------------------\n')

print('테스트용 리뷰 개수 :',len(test_data)) # 테스트용 리뷰 개수 출력
print(test_data[:5]) # test 상위 5개 출력


print('---------------------------------------------------------------------------------------------------\n')

train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
print('총 샘플의 수 :',len(train_data))
train_data['label'].value_counts().plot(kind = 'bar')
print(train_data.groupby('label').size().reset_index(name = 'count'))

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print('null 존재?:', train_data.isnull().values.any()) # Null 값이 존재하는지 확인

print('---------------------------------------------------------------------------------------------------\n')

print('///한글과 공백 외의 것 모두 제거///')
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
print(train_data[:5])

train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

train_data = train_data.dropna(how = 'any') # nan 제거
print(len(train_data))

test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

X_train = []
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

print(X_train[:3])


X_test = []
for sentence in test_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)

import pickle
with open('movie_X_train.txt', 'wb') as f:
    pickle.dump(X_train, f)
    
with open('movie_X_test.txt', 'wb') as f:
    pickle.dump(X_test, f)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

print(tokenizer.word_index)

with open('movie_X_train.txt', 'rb') as f:
    movie_X_train = pickle.load(f) # 단 한줄씩 읽어옴
movie_X_train

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)



# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]


# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))

print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 30
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

"""## LSTM 모델 적용"""

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

loaded_model = load_model('naver_movie_best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

"""## 학습된 모델 적용"""

def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

# 테스트
sentiment_predict('이 책 아주 재밌어')
sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')
sentiment_predict('감독 뭐하는 놈이냐?')
sentiment_predict('이딴게 영화냐 ㅉㅉ')
sentiment_predict('이 영화 핵노잼 ㅠㅠ')

# 도서 리뷰 불러오기

"""## 영화 리뷰 토크나이징, 모델 만든것을 바로 적용해주게 통합
* 위의 학습 과정을 거치고 이 블록을 실행하면 됨.
* 추후에 도서 리뷰데이터를 적용하기 위해 하나의 프로세스로 만듦
* 이미 학습된 모델을 사용하여 기능을 활용할 수 있게 구성
* 이 모듈을 활용해서 각 도서 댓글을 감성분석하고 감성 분석 한 결과를 저장
* 이 결과를 토대로 사용자가 해당 도서를 입력했을 경우 해당 도서의 긍부정 퍼센트를 보여줌
"""

# Commented out IPython magic to ensure Python compatibility.
import os
os.chdir(drive_path)
print(os.getcwd())

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# 학습된 모델 사용

# 전처리만 하기(전체 데이터 대상)
train_data = pd.read_table('movie_ratings_train.txt')
test_data = pd.read_table('movie_ratings_test.txt')
train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
print('총 샘플의 수 :',len(train_data))


print(train_data.groupby('label').size().reset_index(name = 'count'))
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print('null 존재?:', train_data.isnull().values.any()) # Null 값이 존재하는지 확인

print('---------------------------------------------------------------------------------------------------\n')

print('///한글과 공백 외의 것 모두 제거///')
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
print(train_data[:5])
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

train_data = train_data.dropna(how = 'any') # nan 제거
print(len(train_data))
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 전처리 하고 토크나이징 한 파일 불러오기(전처리 + 토크나이징은 시간이 많이 걸려 작업 수행한 파일을 pickle에 저장한 후 불러옴)

with open('movie_X_train.txt', 'rb') as f:
    movie_X_train = pickle.load(f) # 단 한줄씩 읽어옴


with open('movie_X_test.txt', 'rb') as f:
    movie_X_test = pickle.load(f) # 단 한줄씩 읽어옴

tokenizer = Tokenizer()
tokenizer.fit_on_texts(movie_X_train)

print(tokenizer.word_index)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)



tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(movie_X_train)
X_train = tokenizer.texts_to_sequences(movie_X_train)
X_test = tokenizer.texts_to_sequences(movie_X_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])


drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]


# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))


def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))


max_len = 30
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

loaded_model = load_model('naver_movie_best_model.h5')


def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))

  return (score, 1-score)



# 도서 리뷰 불러오기

# 테스트
sentiment_predict('이 책 아주 재밌어')
sentiment_predict('와 개쩐다 정말 세계관 최강자들의 책이다')
sentiment_predict('작가 뭐하는 놈이냐?')
sentiment_predict('이딴게 책이냐 ㅉㅉ')
sentiment_predict('이 책 핵노잼 ㅠㅠ')

"""# 네이버 쇼핑 리뷰 감성 분석"""

# Commented out IPython magic to ensure Python compatibility.
# Colab에 Mecab 설치
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="shopping_ratings_total.txt")

total_data = pd.read_table('shopping_ratings_total.txt', names=['ratings', 'reviews'])
print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력

total_data['label'] = np.select([total_data.ratings > 3], [1], default=0)
total_data[:5]

train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
print('훈련용 리뷰의 개수 :', len(train_data))
print('테스트용 리뷰의 개수 :', len(test_data))

# 한글과 공백을 제외하고 모두 제거
train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data['reviews'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

test_data.drop_duplicates(subset = ['reviews'], inplace=True) # 중복 제거
test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['reviews'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

mecab = Mecab()
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']

train_data['tokenized'] = train_data['reviews'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
test_data['tokenized'] = test_data['reviews'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)

negative_word_count = Counter(negative_words)
print(negative_word_count.most_common(20))

positive_word_count = Counter(positive_words)
print(positive_word_count.most_common(20))

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
text_len = train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('긍정 리뷰의 평균 길이 :', np.mean(text_len))

text_len = train_data[train_data['label']==0]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
fig.suptitle('Words in texts')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('부정 리뷰의 평균 길이 :', np.mean(text_len))
plt.show()

X_train = train_data['tokenized'].values
y_train = train_data['label'].values
X_test= test_data['tokenized'].values
y_test = test_data['label'].values

# X_train, y_train, X_test, y_test 저장

import pickle
with open('shopping_X_train.txt', 'wb') as f:
    pickle.dump(X_train, f)
    
with open('shopping_y_train.txt', 'wb') as f:
    pickle.dump(y_train, f)

with open('shopping_X_test.txt', 'wb') as f:
    pickle.dump(X_test, f)
    
with open('shopping_y_test.txt', 'wb') as f:
    pickle.dump(y_test, f)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)


tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 80
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

import pickle
with open('shopping_X_train.txt', 'wb') as f:
    pickle.dump(X_train, f)
    
with open('shopping_X_test.txt', 'wb') as f:
    pickle.dump(X_test, f)

from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(GRU(128))
model.add(Dense(1, activation='sigmoid'))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)\

loaded_model = load_model('naver_shopping_best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

def sentiment_predict(new_sentence):
  new_sentence = mecab.morphs(new_sentence) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))

sentiment_predict('이 상품 진짜 좋아요... 저는 강추합니다. 대박')
sentiment_predict('진짜 배송도 늦고 개짜증나네요. 뭐 이런 걸 상품이라고 만듬?')
sentiment_predict('판매자님... 너무 짱이에요.. 대박나삼')
sentiment_predict('ㅁㄴㅇㄻㄴㅇㄻㄴㅇ리뷰쓰기도 귀찮아')

crawling_df = pd.read_csv('Copy of 순위별한줄평_종합.csv')
crawling_df = crawling_df['한줄평'] 

for str in crawling_df.iloc[: 20]:
    print( str ,'-----분석결과---->', end='')
    sentiment_predict(str)

    print()

"""## 쇼핑몰 리뷰 토크나이징, 모델 만든것을 바로 적용해주게 통합
* 위의 학습 과정을 거치고 이 블록을 실행하면 됨.
* 추후에 도서 리뷰데이터를 적용하기 위해 하나의 프로세스로 만듦
* 이미 학습된 모델을 사용하여 기능을 활용할 수 있게 구성
* 이 모듈을 활용해서 각 도서 댓글을 감성분석하고 감성 분석 한 결과를 저장
* 이 결과를 토대로 사용자가 해당 도서를 입력했을 경우 해당 도서의 긍부정 퍼센트를 보여줌

"""

# Colab에 Mecab 설치
#!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
#%cd Mecab-ko-for-Google-Colab
#!bash install_mecab-ko_on_colab190912.sh


# 학습된 모델 적용

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

os.chdir(drive_path)
print(os.getcwd())

total_data = pd.read_table('shopping_ratings_total.txt', names=['ratings', 'reviews'])
display(total_data)
print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력

total_data['label'] = np.select([total_data.ratings > 3], [1], default=0)

train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
print('훈련용 리뷰의 개수 :', len(train_data))
print('테스트용 리뷰의 개수 :', len(test_data))

# 한글과 공백을 제외하고 모두 제거
train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data['reviews'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

test_data.drop_duplicates(subset = ['reviews'], inplace=True) # 중복 제거
test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['reviews'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

mecab = Mecab()
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']

with open('shopping_X_train.txt', 'rb') as f:
    X_train = pickle.load(f) # 단 한줄씩 읽어옴


with open('shopping_X_test.txt', 'rb') as f:
    X_test = pickle.load(f) # 단 한줄씩 읽어옴

with open('shopping_y_train.txt', 'rb') as f:
    y_train = pickle.load(f) # 단 한줄씩 읽어옴


with open('shopping_y_test.txt', 'rb') as f:
    y_test = pickle.load(f) # 단 한줄씩 읽어옴


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)



# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)


tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 80
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)


from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



loaded_model = load_model('naver_shopping_best_model.h5')

def sentiment_predict(new_sentence):
  new_sentence = mecab.morphs(new_sentence) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    return score
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))
    return (1-score)



sentiment_predict('이 상품 진짜 좋아요... 저는 강추합니다. 대박')
sentiment_predict('진짜 배송도 늦고 개짜증나네요. 뭐 이런 걸 상품이라고 만듬?')
sentiment_predict('판매자님... 너무 짱이에요.. 대박나삼')
sentiment_predict('ㅁㄴㅇㄻㄴㅇㄻㄴㅇ리뷰쓰기도 귀찮아')

"""# 영화 리뷰 긍정 부정 모델을 바탕으로 긍 부정을 평가하고 이를 기반으로 컨텐츠 기반 필터링 추천 기법 사용"""

# 필요한 모듈과 데이터 불러오기
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

"""## 데이터 불러오기"""

book_senti_df = pd.read_csv(drive_path + '/도서감성데이터.csv') 
crawling_df = pd.read_csv(drive_path + '/Copy of 순위별한줄평_종합.csv')

display(book_senti_df)
display(crawling_df)
display(crawling_df.info())

# 크롤링 한 데이터 평점에 따라 0, 1 (부정, 긍정)으로 수정 및 저장
indexing = {'평점1점': 0, '평점2점': 0, '평점3점': 1, '평점4점': 1, '평점5점': 1}

# 한글과 공백을 제외하고 모두 제거
crawling_df['한줄평'] = crawling_df['한줄평'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
crawling_df['한줄평'].replace('', np.nan, inplace=True)
print(crawling_df.isnull().sum())
crawling_df = crawling_df.dropna(how='any') # Null 값 제거

senti = []
int_score = []
for score in crawling_df['평점']:
    
    if score == '평점5점':
        senti.append(1)
        int_score.append(5)
    if score == '평점4점':
        senti.append(1)
        int_score.append(4)
    elif score == '평점3점':
        senti.append(0)
        int_score.append(3)
    elif score == '평점2점':
        senti.append(0)
        int_score.append(2)
    elif score == '평점1점':
        senti.append(0)
        int_score.append(1)
    

crawling_df['감정'] = senti
crawling_df['평점'] = int_score

display(crawling_df)
# 도서 감성 데이터 0, 1로 나누기
# 둘이 합치기
# 네이버 쇼핑 후기 분석 한것 처럼 고치기

"""## 긍부정 데이터 추가"""

positive = []
negative = []

for str in crawling_df['한줄평']:
    p, n = sentiment_predict(str)
    print(p, n)
    positive.append(p)
    negative.append(n)
crawling_df['긍정확률'] = positive
crawling_df['부정확률'] = negative

display(crawling_df)

crawling_df.to_csv(drive_path + '도서긍부정확률포함.csv')

# 필요한 모듈과 데이터 불러오기
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from warnings import filterwarnings
import re
from sklearn.feature_extraction.text import CountVectorizer

filterwarnings('ignore')

"""## 분류 특성 데이터 정제후 분류열 속성 추가"""

data = pd.read_csv(drive_path + '도서긍부정확률포함.csv')

from sklearn.feature_extraction.text import CountVectorizer

data['분류열'] = data['분류'].apply( lambda x : x[1:-1].replace(',',' '))
# 특수 문제 제거
data["분류열"] = data["분류열"].str.replace(pat=r'[^\w]', repl=r' ', regex=True)


data = data.iloc[:, 2: ]
display(data)
display(data['분류열'])

"""## 책 id, name, 순위로 책을 그룹화 하고 긍부정확률 평균을 구함 """

#순위별평균확률 csv로 만들기
mean_data = data.groupby(['id', 'name', '순위'])['긍정확률','부정확률'].mean()
mean_data.reset_index(inplace=True)
mean_data.sort_values('순위')
mean_data.to_csv(drive_path + '순위별평균확률평균.csv')
display(mean_data)

"""## 분류를 기준으로 중복 제거"""

# 분류를 기준으로 중복 제거해주는 코드
ex_data = data.drop_duplicates(['name'])
ex_data = ex_data.iloc[:, :][['id', 'name', '분류열']]

ex_data.to_csv(drive_path + '분류를 기준으로 중복 제거.csv')
display(ex_data)

"""## 각 작품별 평가한 사람수 """

# 각 작품별 평가한 사람 수
count_score_data = data.groupby(['id','name','순위'])['한줄평'].count()
display(count_score_data)
count_score_data.to_csv(drive_path + '각작품별평가한사람수')

count_score_data = pd.read_csv(drive_path + '각작품별평가한사람수')
count_score_data.sort_values('순위')

"""## 각 작품별 평균 평점 구하기"""

# 각 작품별 평균 평점
mean_score_data = data.groupby(['id','name','순위'])['평점'].mean()
display(mean_score_data)
mean_score_data.to_csv(drive_path + '각작품별평균평점')

mean_score_data = pd.read_csv(drive_path + '각작품별평균평점')
mean_score_data.sort_values('순위')
mean_score_data

"""## 데이터 조인하기(일단 다 합친거라 오류가 있음)"""

# mean_score_data  +  count_score_data + ex_data + mean_data
merge_data1 = pd.merge(mean_score_data, count_score_data, how='inner', on = 'id')
merge_data2 = pd.merge(ex_data, mean_data, how='inner', on = 'id')
merge_data = pd.merge(merge_data1, merge_data2, how='inner', on = 'id')
display(merge_data)
merge_data.columns = ['id','name','순위_x' ,'평점','name_y_x','순위_y',	'한줄평','name_x_y','분류열','name_y_y','순위','긍정확률','부정확률']
merge_data = merge_data[['id', 'name', '한줄평', '순위', '분류열', '긍정확률', '부정확률', '평점']].sort_values('순위')

merge_data = merge_data.drop_duplicates(['name'])
merge_data['name'] = merge_data['name'].str.split().agg("".join)
merge_data.to_csv(drive_path +'가중치에필요한데이터모음.csv')
display(merge_data)

"""## 컨텐츠 기반 추천 부분
* 위의 조인한 데이터를 기반으로 진행
* 서비스로 올릴때는 조인한 데이터만 있으면 될것 같음.
* 일단 프론트단을 구현하고 시간이 남으면 이 데이터를 디비로 구축하고 배포하면 될듯.
"""

# 필요한 모듈과 데이터 불러오기
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from warnings import filterwarnings
import re
from sklearn.feature_extraction.text import CountVectorizer

filterwarnings('ignore')

merge_data = pd.read_csv(drive_path +'/가중치에필요한데이터모음.csv')

count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform( merge_data['분류열'] )
print(genre_mat.shape)

# 코사인 유사도 계산
# 반환된 코사인 유사도 행렬의 크기 및 앞 2개 데이터만 추출
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
genre_sim.shape

genre_sim[:2]
np.sort(genre_sim)[:, ::-1]


# 유사도가 높은 순으로 정리된 genre_sim 객체의 비교 행 위치 인덱스 값
# 값이 높은 순으로 정렬된 비교 대상 행의 유사도 값이 아니라
#  비교 대상 행의 위치 인덱스임에 주의
genre_sim_sorded_ind = genre_sim.argsort()[:, ::-1]
genre_sim_sorded_ind[:1]

# 장르 유사도에 따라 영화를 추천하는 함수를 생성
# movies_df DataFrame, 
# 레코드별 장르 코사인 유사도 인덱스를 가지는 genre_sim_sorted_ind
# 고객이 선정한 추천 기준이 되는 영화 제목
# 추천할 영화 건수
# return : 추천 영화 정보 DataFrame

# 정확한 책이름으로 찾기
def find_sim_book(df, sorted_ind, title_name, top_n=10):
  
  
  title_book = df[ df['name'].str.contains(title_name) ]
  title_book = title_book[title_book['평점'] == title_book['평점'].max() ]
  display(title_book)

  title_index = title_book.index.values
  sim_indexs = sorted_ind[title_index, :(top_n)]
  sim_indexs = sim_indexs.reshape(-1)
  sim_indexs = sim_indexs[sim_indexs != title_index]

  return df.iloc[sim_indexs].sort_values('weighted_vote', 
                                         ascending=False)[:top_n]

# 키워드로 찾기
def find_keyword_book(df, sorted_ind, keyword, top_n=10):
  
  keyword_book = pd.DataFrame()
  for key in keyword[1:]:

    temp = df[ df['분류열'].str.contains(key) ]
    keyword_book = pd.concat([temp,keyword_book])

  keyword_book.drop_duplicates(['id'])

  return keyword_book.sort_values('weighted_vote', 
                                         ascending=False)[:top_n]

# 가중평점=(𝑣/(𝑣+𝑚+p))∗𝑅+(𝑚/(𝑣+𝑚+p))∗𝐶 + (p/(𝑣+𝑚+p))*P*3

# v: 개별 영화에 평점을 투표한 횟수
# m: 평점을 부여하기 위한 최소 투표 횟수
# p: 긍정평가 확률
# R: 개별 책에 대한 평균 평점
# C: 전체 책에 대한 평균 평점
# P: 전체 책에 대한 평균 긍정

# 기존 평점을 가중 평점으로 변경하는 함수

C = merge_data['평점'].mean()
m = merge_data['한줄평'].quantile(0.6)
p = merge_data['긍정확률'].mean()
    
v = merge_data['한줄평']
R = merge_data['평점']
P = merge_data['긍정확률']

merge_data['weighted_vote'] = (v/(v+m+p))*R+(m/(v+m+p))*C + (p/(v+m+p))*P*5
display(merge_data)

# test code
choice = input("제목으로 찾기: 1번 or 키워드로 찾기: 2번 or 책 순위 보기 3번\n")

# 제목으로 찾아보기(제목이 정확해야함)
if choice == '1':

    user_book = input("정확한 책 제목을 입력하세요: ")
    # 띄어쓰기 제거
    user_book = user_book.replace(' ','')
    print(user_book)
    sim_books = find_sim_book(merge_data, genre_sim_sorded_ind, user_book, 10) # 책 이름을 입력

# 키워드로 찾아보기
elif choice == '2':

    user_book = input("책 키워드를 #로 입력하세요: ")
    user_book = user_book.replace(' ','')

    print(user_book)
    keyword = user_book.split('#')
    print(keyword)
    sim_books = find_keyword_book(merge_data, genre_sim_sorded_ind, keyword, 10) # 책 이름을 입력

elif choice == '3':
    bound = input('원하는 순위를 입력하세요. ex)1-10(10단위) \n')
    bound = bound.replace(' ', '')
    bound = bound.split('-')
    print(bound)
    merge_data['순위'] = merge_data['순위'].astype('int') 
    sim_books = merge_data[(merge_data['순위'] >= int(bound[0])) & (merge_data['순위'] <= int(bound[1]))]
else:
     print('잘못된 선택지 입력 다시 입력하세요.')

sim_books = sim_books[['id', 'name', '한줄평', '순위', '분류열', '긍정확률', '부정확률', '평점', 'weighted_vote']]
display(sim_books) # 추천된 책 보여줌

