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
from sklearn.metrics.pairwise import cosine_similarity


class recomodel:

    def __init__(self):
        filterwarnings('ignore')

        self.merge_data = pd.read_csv(drive_path +'temp/가중치에필요한데이터모음.csv')

        # 가중평점=(𝑣/(𝑣+𝑚+p))∗𝑅+(𝑚/(𝑣+𝑚+p))∗𝐶 + (p/(𝑣+𝑚+p))*P*3

        # v: 개별 영화에 평점을 투표한 횟수
        # m: 평점을 부여하기 위한 최소 투표 횟수
        # p: 긍정평가 확률
        # R: 개별 책에 대한 평균 평점
        # C: 전체 책에 대한 평균 평점
        # P: 전체 책에 대한 평균 긍정

        # 기존 평점을 가중 평점으로 변경하는 함수

        C = self.merge_data['평점'].mean()
        m = self.merge_data['한줄평'].quantile(0.6)
        p = self.merge_data['긍정확률'].mean()
            
        v = self.merge_data['한줄평']
        R = self.merge_data['평점']
        P = self.merge_data['긍정확률']

        self.merge_data['weighted_vote'] = (v/(v+m+p))*R+(m/(v+m+p))*C + (p/(v+m+p))*P*5
        display(self.merge_data)



    # 유사도가 높은 순으로 정리된 genre_sim 객체의 비교 행 위치 인덱스 값
    # 값이 높은 순으로 정렬된 비교 대상 행의 유사도 값이 아니라
    #  비교 대상 행의 위치 인덱스임에 주의

    def get_genre_sim(self):
        
        count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
        genre_mat = count_vect.fit_transform( self.merge_data['분류열'] )
        print(genre_mat.shape)

        # 코사인 유사도 계산
        # 반환된 코사인 유사도 행렬의 크기 및 앞 2개 데이터만 추출

        genre_sim = cosine_similarity(genre_mat, genre_mat)
        print(genre_sim.shape)

        #genre_sim[:2]
        np.sort(genre_sim)[:, ::-1]

        genre_sim_sorded_ind = genre_sim.argsort()[:, ::-1]
        return genre_sim_sorded_ind

    # 장르 유사도에 따라 영화를 추천하는 함수를 생성
    # movies_df DataFrame, 
    # 레코드별 장르 코사인 유사도 인덱스를 가지는 genre_sim_sorted_ind
    # 고객이 선정한 추천 기준이 되는 영화 제목
    # 추천할 영화 건수
    # return : 추천 영화 정보 DataFrame

    # 정확한 책이름으로 찾기
    def find_sim_book(self, sorted_ind, title_name, top_n=10):
    
        # 띄어쓰기 제거
        title_name = title_name.replace(' ','')
        title_book = self.merge_data[ self.merge_data['name'].str.contains(title_name) ]
        title_book = title_book[title_book['평점'] == title_book['평점'].max() ]
        display(title_book)

        title_index = title_book.index.values
        sim_indexs = sorted_ind[title_index, :(top_n)]
        sim_indexs = sim_indexs.reshape(-1)
        sim_indexs = sim_indexs[sim_indexs != title_index]

        return self.merge_data.iloc[sim_indexs].sort_values('weighted_vote', 
                                                ascending=False)[:top_n][['id', 'name', '한줄평', '순위', '분류열', '긍정확률', '부정확률', '평점', 'weighted_vote']]

    # 키워드로 찾기
    def find_keyword_book(self, sorted_ind, keyword, top_n=10):
        keyword = keyword.replace(' ','')
        keyword = keyword.split('#')

        keyword_book = pd.DataFrame()
        for key in keyword[1:]:

            temp = self.merge_data[ self.merge_data['분류열'].str.contains(key) ]
            keyword_book = pd.concat([temp,keyword_book])

        keyword_book.drop_duplicates(['id'])

        return keyword_book.sort_values('weighted_vote', 
                                                ascending=False)[:top_n][['id', 'name', '한줄평', '순위', '분류열', '긍정확률', '부정확률', '평점', 'weighted_vote']]

    # 단순 책 순위 보여주기
    def find_rank_book(self, bound):

        bound = bound.replace(' ', '')
        bound = bound.split('-')
        print(bound)
        self.merge_data['순위'] = self.merge_data['순위'].astype('int') 
        sim_books = self.merge_data[(self.merge_data['순위'] >= int(bound[0])) & (self.merge_data['순위'] <= int(bound[1]))]
        return sim_books


if __name__ == "__main__":

    rec = recomodel()

    # test code
    choice = input("제목으로 찾기: 1번 or 키워드로 찾기: 2번 or 책 순위 보기 3번\n")

    # 제목으로 찾아보기(제목이 정확해야함)
    if choice == '1':

        user_book = input("정확한 책 제목을 입력하세요: ")

        print(user_book)
        sim_books = rec.find_sim_book(rec.get_genre_sim(), user_book, 10) # 책 이름을 입력

    # 키워드로 찾아보기
    elif choice == '2':

        user_book = input("책 키워드를 #로 입력하세요: ")
        print(user_book)
            
        sim_books = rec.find_keyword_book(rec.get_genre_sim(), user_book, 10) # 책 이름을 입력

    elif choice == '3':
        bound = input('원하는 순위를 입력하세요. ex)1-10(10단위) \n')
        sim_books = rec.find_rank_book(bound)
    else:
        print('잘못된 선택지 입력 다시 입력하세요.')

    sim_books = sim_books[['id', 'name', '한줄평', '순위', '분류열', '긍정확률', '부정확률', '평점', 'weighted_vote']]
    display(sim_books) # 추천된 책 보여줌