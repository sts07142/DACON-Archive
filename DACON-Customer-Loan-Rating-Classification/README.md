## [DACON 고객 대출등급 분류 AI 해커톤](https://dacon.io/competitions/official/236214/overview/description)
- 고객의 대출등급을 예측하는 AI 알고리즘 개발
- 알고리즘 | 정형 | 분류 | 금융 | Macro F1 Score
- 2024.01.15 ~ 2024.02.05

## 데이터셋

<details>
<summary>
<b>train.csv</b>
</summary>

    - 고객 관련 금융 정보
    - ID : 대출 고객의 고유 ID
    - 대출금액
    - 대출기간
    - 근로기간
    - 주택소유상태
    - 연간소득
    - 부채_대비_소득_비율 
    - 총계좌수
    - 대출목적
    - 최근_2년간_연체_횟수
    - 총상환원금
    - 총상환이자
    - 총연체금액
    - 연체계좌수
    - 대출등급 : 예측 목표
</details>

<details>
<summary>
<b>test.csv</b>
</summary>

    - 고객 관련 금융 정보
    - ID : 대출 고객의 고유 ID
    - 대출금액
    - 대출기간
    - 근로기간
    - 주택소유상태
    - 연간소득
    - 부채_대비_소득_비율 
    - 총계좌수
    - 대출목적
    - 최근_2년간_연체_횟수
    - 총상환원금
    - 총상환이자
    - 총연체금액
    - 연체계좌수    
    - 대출등급이 존재하지 않음
</details>

<details>
<summary>
<b>sample_submission.csv</b>
</summary>

    - ID : 대출 고객의 고유 ID
    - Income : test.csv에서 제공된 고객의 대출등급을 예측하여 기입
</details>

</details>
<br>

# [상위 13%] CatBoost + StratifiedKFold + Wandb

## 1. 데이터 처리
### 1-a. 전처리
- 데이터 형식 숫자+문자 -> 숫자 추출

### 1-b. feature importance 기준 drop

## 2. 파생변수 생성
```
def add_var(train):
    train['총상환원금+총상환이자-총연체금액/대출금액'] = (train['총상환원금'] + train['총상환이자'] - train['총연체금액']) / train['대출금액'] * 100
    # train['대출금액/대출기간/연간소득 %'] = train['대출금액'] / train['대출기간'] / train['연간소득'] * 100
    train['총상환원금/대출금액'] = (train['총상환원금']) / train['대출금액'] * 100
    train['대출금액/대출기간'] = train['대출금액'] / train['대출기간'] * 100
    train['대출금액/연간소득'] = train['대출금액'] / train['연간소득'] * 100
    # train['총연체금액/대출금액 %'] = train['총연체금액'] / train['대출금액'] * 100
    train['총상환이자/총상환원금'] = train['총상환이자'] / train['총상환원금'] * 100
    train['근로기간/대출기간'] = train['근로기간'] / train['대출기간'] * 100
    train['연간소득/대출기간'] = train['연간소득'] / train['대출기간'] * 100
    train['최근_2년간_연체_횟수/대출기간'] = train['최근_2년간_연체_횟수'] / train['대출기간'] *12 * 100
    train['총상환원금/대출기간'] = train['총상환원금'] / train['대출기간'] * 100
    train['총상환이자/대출기간'] = train['총상환이자'] / train['대출기간'] * 100
    # train['총연체금액/대출기간 %'] = train['총연체금액'] / train['대출기간'] * 100
    train['근로기간*연간소득'] = train['근로기간'] * train['연간소득']
    train['주택소유상태_대출목적'] = train['주택소유상태'] + "_" + train['대출목적']
    # train['연체계좌수/총계좌수'] = train['연체계좌수'] / train['총계좌수']
    return train
```
## 3. CatBoost 단일 모델 사용
```
from catboost import CatBoostClassifier, CatBoostRegressor
```

## 4. Wandb 하이퍼파라미터 튜닝
- Wandb
- best_params
```
{
  "accuracy": 0.9121547401061985,
  "_timestamp": 1706900313.1919417,
  "_runtime": 4.466750621795654,
  "_step": 2,
  "learning_rate": 0.13828280590323885,
  "max_depth": 5,
  "_wandb": {
    "runtime": 3
  }
}
```