# medium_dropna.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#... (데이터 불러오기 및 price, milage 열 숫자 변환 코드)...

df = pd.read_csv('used_cars.csv')

# --- 데이터 정제 및 피처 엔지니어링 시작 ---

# 'price' 열: '$'와 ',' 제거 후 숫자(float)로 변환
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

# 'milage' 열: ' mi.'와 ',' 제거 후 숫자(int)로 변환
df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '').astype(int)

# S2: 이상치 처리 (가장 중요!) ---
# 모델링에 들어가기 전, 가격이 너무 높은 상위 1% 데이터를 제거합니다.
price_98th = df['price'].quantile(0.98)
df = df[df['price'] < price_98th]

# 'engine' 열에서 마력(HP) 정보 추출하기 (피처 엔지니어링)
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

# 어떤 열에 결측치가 얼마나 있는지 확인
print("결측치 처리 전:")
print(df.isnull().sum())


# --- 결측치 처리 시작 (dropna 방식) ---

# dropna()를 사용하여 결측치가 하나라도 포함된 행을 모두 삭제합니다. [1, 3]
# 이 방식은 구현이 간단하지만, 결측치가 포함된 행의 정보가 모두 사라지는 데이터 손실이 발생할 수 있습니다. [1]
# inplace=True 옵션은 원본 데이터프레임을 직접 수정합니다. [1]
df.dropna(inplace=True)

# 만약 특정 열(예: 예측 목표인 'price')에 결측치가 있는 행만 선별적으로 삭제하고 싶다면,
# 'subset' 옵션을 사용할 수 있습니다. [1, 4]
# 예: df.dropna(subset=['price', 'horsepower'], inplace=True)

# 모든 결측치가 처리되었는지 최종 확인
print("\n결측치 처리 후:")
print(df.isnull().sum())

# 문자(object) 데이터를 원-핫 인코딩으로 변환

# 변환할 문자열 칼럼 리스트
categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']

df = pd.get_dummies(df, columns=categorical_features)

# 최종 데이터 확인
print("\n원-핫 인코딩 후 데이터 샘플:")
print(df.head())

print("\n결측치 처리 전 데이터 정보:")
print(df.info())

# 1. 문제지와 정답지 분리

# 'price'를 제외한 모든 열 = X (문제지, Features)
X = df.drop('price', axis=1) 

# 'price' 열 = y (정답지, Target)
y = df['price'] 

# 2. 훈련용(Train)과 시험용(Test) 데이터 분리

# X와 y를 8:2 비율로 나눔
# random_state는 나눌 때의 규칙을 고정시켜서, 매번 같은 방식으로 나뉘게 함
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 결과 확인 (선택 사항)
print("--- 데이터 분리 결과 ---")
print("전체 데이터 개수:", len(X))
print("훈련용 데이터(X_train) 개수:", len(X_train))
print("시험용 데이터(X_test) 개수:", len(X_test))

print("\ntotal mean : ", y.mean())

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

    # MAE 계산 (이 부분 추가!)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{name} 모델의 현실적인 평균 오차(MAE): ${mae:.2f}\n")