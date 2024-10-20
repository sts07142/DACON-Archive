## [DACON 소득 예측 AI 해커톤](https://dacon.io/competitions/official/236230/overview/description)
- 개인 특성 데이터를 활용하여 개인 소득 수준을 예측하는 AI 모델 개발
- 알고리즘 | 정형 | 회귀 | 사회 | RMSE
- 2024.03.11 ~ 2024.04.08

## 데이터셋

<details>
<summary>
<b>train.csv</b>
</summary>

    - 한 사람에 관련된 다양한 사회적, 경제적 정보
    - ID : 학습 데이터 고유 ID
    - Age
    - Gender
    - Education_Status
    - Employment_Status
    - Working_Week (Yearly)
    - Industry_Status
    - Occupation_Status
    - Race
    - Hispanic_Origin
    - Martial_Status
    - Household_Status
    - Household_summary
    - Citizenship
    - Birth_Country
    - Birth_Country (Father)
    - Birth_Country (Mother)
    - Tax_Status
    - Gains
    - Losses
    - Divdends
    - Incom_Status
    - Income : 예측 목표, 1시간 단위의 소득을 예측
</details>

<details>
<summary>
<b>test.csv</b>
</summary>

    - 한 사람에 관련된 다양한 사회적, 경제적 정보
    - ID : 학습 데이터 고유 ID
    - Age
    - Gender
    - Education_Status
    - Employment_Status
    - Working_Week (Yearly)
    - Industry_Status
    - Occupation_Status
    - Race
    - Hispanic_Origin
    - Martial_Status
    - Household_Status
    - Household_summary
    - Citizenship
    - Birth_Country
    - Birth_Country (Father)
    - Birth_Country (Mother)
    - Tax_Status
    - Gains
    - Losses
    - Divdends
    - Incom_Status
    - Income이 존재하지 않음
</details>

<details>
<summary>
<b>sample_submission.csv</b>
</summary>

    - ID : 테스트 데이터 고유 ID
    - Income : ID에 해당되는 Income을 예측하여 제출
</details>

</details>
<br>

# [상위 13%] CatBoost + StratifiedKFold + Optuna

## 1. 데이터 처리
### 1-a. Income = 0 인 경우
- Industry_Status : Not in universe
- Occupation_Status : Unknown
- Employment_Status : Not Working
- Education_Status : Children

train 데이터 관측 기준, Income 값이 0인 것 확인

### 1-b. Income 예측이 음수 값인 경우 0으로 변환

후처리 단계에서, Income 예측값 음수인 경우 처리

## 2. 파생변수 생성
```
for df in [train, test]:
    df['Working_Week_rate'] = df['Working_Week (Yearly)'] / 52

    df['Edu_Emp'] = df['Education_Status'] + '_' + df['Employment_Status']

    df['Ind_Occ'] = df['Industry_Status'] + '_' + df['Occupation_Status']

    df['Race_Hisp'] = df['Race'] + '_' + df['Hispanic_Origin']

    df['Tax_Income'] = df['Tax_Status'] + '_' + df['Income_Status']

    df['Birth_Country_F_M'] = df['Birth_Country (Father)'] + '_' + df['Birth_Country (Mother)']

    df['Age * Work%'] = df['Age'] * df['Working_Week_rate']
    
    df['log Gains'] = np.log(df['Gains']+1)

    df['log Losses'] = np.log(df['Losses']+1)

    df['log Dividends'] = np.log(df['Dividends']+1)
    
```
## 3. CatBoostRegressor 단일 모델 사용
```
from catboost import CatBoostRegressor
```

## 4. Optuna 하이퍼파라미터 튜닝
- Optuna
- best_params
```
{'depth': 7, 'learning_rate': 0.017583538296520786, 'random_strength': 0.0010018857421937968, 'border_count': 106, 'l2_leaf_reg': 85.22284049359163, 'leaf_estimation_iterations': 3, 'leaf_estimation_method': 'Newton', 'bootstrap_type': 'Bayesian', 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 30, 'one_hot_max_size': 1, 'random_state': 42, 'verbose': 0, 'iterations': 1000, 'loss_function': 'RMSE'}
```