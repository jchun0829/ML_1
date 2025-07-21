# medium_replace.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

# S1: ... (데이터 불러오기 및 price, milage 열 숫자 변환 코드) ...

df = pd.read_csv('used_cars.csv')

# 'price' 열: '$'와 ',' 제거 후 숫자(float)로 변환
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

# 'milage' 열: ' mi.'와 ',' 제거 후 숫자(int)로 변환
df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '').astype(int)

# # S2: 이상치 처리 (가장 중요!) ---
# # 모델링에 들어가기 전, 가격이 너무 높은 상위 1% 데이터를 제거합니다.
# price_98th = df['price'].quantile(0.98)
# df = df[df['price'] < price_98th]

# S3: 'engine' 열에서 마력(HP) 정보 추출하기 (피처 엔지니어링)
# 'HP' 앞에 있는 숫자(소수점 포함)를 뽑아내는 정규표현식 사용
df['horsepower'] = df['engine'].str.extract(r'(\d+\.?\d*)\s*HP').astype(float)

# 'engine' 열에서 추가 정보 추출
# 'L' 앞에 있는 숫자를 'engine_L'로 추출
df['engine_L'] = df['engine'].str.extract(r'(\d+\.?\d*)\s*L').astype(float)

# 'Cylinder' 또는 'V' 앞에 있는 숫자를 'cylinders'로 추출
df['cylinders'] = df['engine'].str.extract(r'(\d)\s*(?:Cylinder|V)').astype(float)

# 'model_year'를 이용해 'car_age' 피처 생성
current_year = 2025 # 현재 연도 (또는 데이터가 수집된 시점)
df['car_age'] = current_year - df['model_year']

# 불필요한 열 삭제
df.drop(['engine', 'model_year'], axis=1, inplace=True)

# S4: 숫자(float, int) 열의 결측치를 '중앙값(median)'으로 채우기
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    median_val = df[col].median()
    df.fillna({col : median_val}, inplace=True)

# 문자(object) 열의 결측치를 '최빈값(mode)'으로 채우기
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    mode_val = df[col].mode()[0] # 최빈값은 여러 개일 수 있어 첫 번째 값을 선택
    df.fillna({col : mode_val}, inplace=True)


# 만약 예측 목표인 'price' 열에 결측치가 있다면, 그 행은 예측할 수 없으므로 삭제하는 것이 좋습니다.
# df.dropna(subset=['price'], inplace=True)

# S5: 문자(object) 데이터를 원-핫 인코딩으로 변환

# 변환할 문자열 칼럼 리스트
categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']

df = pd.get_dummies(df, columns=categorical_features)

# 문제지와 정답지 분리

# 'price'를 제외한 모든 열 = X (문제지, Features)
X = df.drop('price', axis=1) 

# 'price' 열 = y (정답지, Target)
y = df['price'] 

print(y.mean())

# S6: 훈련용(Train)과 시험용(Test) 데이터 분리

# X와 y를 8:2 비율로 나눔
# random_state는 나눌 때의 규칙을 고정시켜서, 매번 같은 방식으로 나뉘게 함
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 결과 확인 (선택 사항)
# print("--- 데이터 분리 결과 ---")
# print("전체 데이터 개수:", len(X))
# print("훈련용 데이터(X_train) 개수:", len(X_train))
# print("시험용 데이터(X_test) 개수:", len(X_test))

#S7: 분석

# 비교할 모델들을 딕셔너리 형태로 준비합니다.
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)#,
    #'Huber': HuberRegressor()
}

# 각 모델을 훈련시키고 성능을 평가합니다.
for name, model in models.items():
    print(f"--- {name} 모델 훈련 및 평가 시작 ---")
    
    # 모델 훈련
    model.fit(X_train, y_train)
    
    # 예측
    predictions = model.predict(X_test)
    
    # 성능 평가 (RMSE)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"{name} 모델의 예측 오차(RMSE): ${rmse:.2f}\n")