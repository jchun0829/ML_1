# 1st_ML_Project
🚗 중고차 가격 예측 모델 개발 프로젝트
1. 프로젝트 개요 (Overview)
이 프로젝트는 데이터 분석 및 머신러닝의 기초를 다지기 위한 첫 번째 개인 프로젝트입니다. Kaggle의 중고차 데이터를 활용하여, 차량의 다양한 특성(피처)을 기반으로 가격을 예측하는 회귀 모델을 개발하고 평가하는 전 과정을 담고 있습니다.

단순히 모델을 만드는 것을 넘어, 데이터 정제부터 피처 엔지니어링, 모델링, 성능 평가에 이르는 머신러닝 프로젝트의 전체 파이프라인을 직접 경험하고 이해하는 것을 목표로 합니다. 이를 통해 문제 해결 능력과 데이터 기반의 논리적 사고 역량을 증명하고자 합니다.

2. 개발 환경 및 사용 데이터
개발 환경: Python, Jupyter Notebook, Google Colab

주요 라이브러리: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

사용 데이터: Kaggle - Used Car Price Prediction (사용한 데이터셋의 정확한 링크를 여기에 삽입하세요)

3. 프로젝트 파이프라인 (Methodology)
본 프로젝트는 아래와 같은 체계적인 단계로 진행되었습니다.

데이터 탐색 (EDA): 데이터의 기본 구조와 통계적 특성을 파악하고, 변수 간의 관계를 시각화를 통해 탐색했습니다.

데이터 전처리: 모델 학습에 방해가 되는 결측치를 처리하고, 모델이 이해할 수 있도록 문자형 데이터(범주형)를 숫자형으로 변환했습니다.

피처 엔지니어링: 모델의 성능을 향상시키기 위해 기존 피처를 조합하여 '차량 나이(Car_Age)'와 같은 새로운 의미를 가진 파생 변수를 생성했습니다.

모델링: 선형 회귀, 랜덤 포레스트 등 다양한 회귀 모델을 학습하고 성능을 비교 분석했습니다.

평가: RMSE(Root Mean Squared Error)를 기준으로 각 모델의 예측 성능을 객관적으로 평가하고 최적의 모델을 선정했습니다.

4. 핵심 결과 (Results)
모델 성능 비교
다양한 모델을 학습시킨 결과, 다음과 같은 성능을 확인했습니다.

Model	RMSE (Root Mean Squared Error)
Linear Regression	(예: 4.51)
Random Forest	(예: 2.87)
Gradient Boosting	(예: 2.55)

Sheets로 내보내기
분석 결과, Gradient Boosting 모델이 가장 낮은 오류율(RMSE)을 보여 최적 모델로 선정되었습니다.

주요 피처 중요도
최종 모델인 Gradient Boosting에서 확인한 주요 피처 중요도입니다. '엔진 출력(Power)', '차량 나이(Car_Age)' 등이 가격 예측에 큰 영향을 미치는 것을 확인할 수 있었습니다.

(노트북에서 만든 피처 중요도 그래프 이미지를 캡처하여 여기에 추가하세요)

5. 결론 및 배운 점 (Conclusion & What I Learned)
결론
본 프로젝트를 통해 주행거리, 연식, 엔진 출력 등 주요 피처들이 중고차 가격에 중요한 영향을 미치는 것을 데이터 기반으로 확인할 수 있었으며, 이를 바탕으로 성공적인 예측 모델을 구축했습니다. 특히 피처 엔지니어링을 통해 '차량 나이'와 같은 새로운 변수를 생성하는 것이 모델 성능에 큰 영향을 미친다는 점을 실증적으로 확인했습니다.

배운 점 및 한계
배운 점:

이론으로만 배우던 데이터 전처리(결측치 처리, 인코딩)가 실제 분석에서 얼마나 중요하고 많은 시간을 차지하는지 체감할 수 있었습니다.

단순히 모델을 적용하는 것을 넘어, EDA와 피처 엔지니어링을 통해 데이터에 대한 깊은 이해가 선행되어야 모델의 성능을 제대로 이끌어낼 수 있다는 것을 배웠습니다.

GitHub를 통해 프로젝트 과정을 체계적으로 기록하고 문서화하는 것의 중요성을 깨달았습니다.

한계점 및 향후 과제 (Future Work):

현재 데이터에는 차량의 '사고 유무', '색상', '세부 옵션' 등 가격에 영향을 미칠 수 있는 중요 피처가 누락되어 있습니다. 이러한 데이터가 추가된다면 모델의 예측 성능이 더욱 개선될 것입니다.

이번 프로젝트에서는 기본 파라미터를 사용했지만, 향후 하이퍼파라미터 튜닝(Hyperparameter Tuning)을 통해 모델을 더욱 정교하게 최적화하여 성능을 추가로 향상시킬 계획입니다.

🇬🇧 README.md (English Version)
(This is the English version for your portfolio. You can copy and paste this as well.)

🚗 Used Car Price Prediction Model Project
1. Overview
This is a foundational personal project aimed at building basic skills in data analysis and machine learning. Using a used car dataset from Kaggle, this project covers the end-to-end process of developing and evaluating a regression model to predict vehicle prices based on various features.

The primary goal is not just to build a model, but to experience and understand the entire machine learning pipeline, from data cleaning and feature engineering to modeling and performance evaluation. This project serves to demonstrate problem-solving abilities and data-driven logical thinking.

2. Environment & Data
Environment: Python, Jupyter Notebook, Google Colab

Core Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

Dataset: Kaggle - Used Car Price Prediction (Insert the exact link to the dataset you used here)

3. Methodology / Project Pipeline
This project was conducted through the following systematic steps:

Exploratory Data Analysis (EDA): Investigated the basic structure and statistical properties of the data, and explored relationships between variables through visualization.

Data Preprocessing: Handled missing values and converted categorical features (text-based) into numerical formats suitable for the model.

Feature Engineering: Created new, meaningful derivative features, such as 'Car_Age', from existing ones to improve model performance.

Modeling: Trained and compared various regression models, including Linear Regression and Random Forest.

Evaluation: Objectively assessed the predictive performance of each model using RMSE (Root Mean Squared Error) to select the optimal model.

4. Results
Model Performance Comparison
After training several models, the following performance metrics were observed:

Model	RMSE (Root Mean Squared Error)
Linear Regression	(e.g., 4.51)
Random Forest	(e.g., 2.87)
Gradient Boosting	(e.g., 2.55)

Sheets로 내보내기
The analysis concluded that the Gradient Boosting model was optimal, exhibiting the lowest Root Mean Squared Error (RMSE).

Key Feature Importance
The feature importance plot from the final Gradient Boosting model is shown below. It indicates that features like 'Power' and 'Car_Age' have a significant impact on price prediction.

(Capture the feature importance graph from your notebook and insert the image here)

5. Conclusion & What I Learned
Conclusion
This project successfully built a predictive model and confirmed, through a data-driven approach, that key features like mileage, year, and engine power significantly influence used car prices. It was empirically verified that feature engineering, particularly the creation of new variables like 'Car_Age', plays a crucial role in enhancing model performance.

What I Learned & Future Work
What I Learned:

I gained a practical understanding of how critical and time-consuming data preprocessing (handling missing values, encoding) is in a real-world analysis.

I learned that a deep understanding of the data through EDA and feature engineering must precede modeling to achieve meaningful performance.

I realized the importance of systematically documenting the project process through tools like GitHub for clarity and reproducibility.

Limitations & Future Work:

The current dataset lacks important features that could affect price, such as accident history, color, and specific options. Including this data would likely improve model accuracy.

This project used default model parameters. Future work will involve hyperparameter tuning to further optimize the model and enhance its predictive power.