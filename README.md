-----

# cvm\_simple

**Single-Bounded Dichotomous Choice (SBDC) CVM Analysis Package**

`cvm_simple` is a Python package designed to perform Single-Bounded Contingent Valuation Method (CVM) analysis. It allows users to trace every step of the calculation—from log-transformation to Hessian matrix derivation—making it an excellent tool for educational purposes and cross-verification with Excel results.

---

## Data Preparation (Important)

**Before running the analysis, please exclude "Protest Responses" from your dataset.**

* **Protest Responses** refer to respondents who select "No" (0 WTP) not because they value the good at zero, but because they object to the payment vehicle (e.g., taxes) or the survey scenario itself.
* Including these invalid zeros can bias the WTP estimation.
* This package assumes that the input data consists only of **valid responses**.

-----

## Key Features

  * **Logic Replication**: Implements the exact "Log-Logit" model ($V = a + b \ln(Bid)$) commonly used in CVM tutorials.
  * **Traceable Process**: Provides access to intermediate calculation steps (Process 1\~6).
  * **Statistical Inference**: Calculates Hessian matrices, Variance-Covariance matrices, Standard Errors, t-values, and p-values.
  * **Bilingual Support**: All docstrings and comments are provided in both **English** and **Korean**.

## Installation

You can install the latest version of this package directly from PyPI:

```bash
pip install cvm-simple
```

For Google Colab or Jupyter Notebook users: Please add an exclamation mark (!) before the command:

```bash
!pip install cvm-simple
```

## Quick Start

Here is a simple example: estimating WTP for national park conservation.

```python
import pandas as pd
from cvm_simple import SingleBoundedLogit

# 1. Prepare Data (Annual donation for national park conservation)
# bid: suggested donation amount ($), yes/no: number of responses
df = pd.DataFrame({
    'bid': [5, 10, 20, 50, 100],
    'yes': [80, 65, 52, 30, 18],
    'no':  [20, 35, 48, 70, 82]
})

# 2. Initialize and Fit Model
model = SingleBoundedLogit()
model.fit(df, bid_col='bid', yes_col='yes', no_col='no', max_bid_integral=200)

# 3. Print Summary Report
model.summary()

# 4. Check Plotting Data (Real vs Estimate)
print(model.process_plot_data)

# 5. Calculate Confidence Intervals (Krinsky & Robb)
model.calculate_kr_confidence_interval(n_sim=1000)
```

## Traceable Processes

You can access intermediate steps to verify calculations.

| Property                                | Description | Excel Equivalent |
|:----------------------------------------| :--- | :--- |
| `model.process1_log_transformation`     | Log-transformed bids | `ln(Bid)` column |
| `model.process2_utility`                | Utility calculation ($V$) | Hidden Utility formula |
| `model.process3_probability`            | Probability calculation ($P$) | `Estimate` column |
| `model.process4_likelihood`             | Log-Likelihood contribution | `SumProduct` components |
| `model.process5_wtp`                    | Median & Truncated Mean WTP | WTP calculation area |
| `model.process6_statistics`             | Hessian & Inference | `Laa`, `Lbb`, `S.E`, `p-value` |
| `model.process_plot_data`                |Data for Plotting |   Real vs Estimate Table |


### Example: Verifying Statistics

```python
# Check the Hessian Matrix
print(model.process6_statistics)
```

## Parameter Guide

| Parameter | Description | Recommendation |
|:----------|:------------|:---------------|
| `bid_col` | Column name for bid amounts | Required |
| `yes_col` | Column name for "Yes" responses | Required |
| `no_col` | Column name for "No" responses | Required |
| `max_bid_integral` | Upper limit for truncated mean integration | **2~3x of max bid** (e.g., if max bid is $100, set to 200~300) |

-----

# [한국어] cvm\_simple

**로직을 구현한 단일양분선택형(SBDC) CVM 분석**

`cvm_simple`은 단일양분선택형 조건부 가치측정법(CVM) 분석을 위한 파이썬 패키지입니다. 결과값만 보여주는 일반적인 통계 패키지와 달리, 이 패키지는 로그 변환부터 헤시안 행렬 계산까지 분석의 모든 단계를 추적할 수 있어, 결과값을 검증하는 데 최적화되어 있습니다.

## 주의 사항
 * **지불 거부자 제외**: 분석을 수행하기 전에, 반드시 데이터에서 "지불거부자(Protest responses)"를 제외해야 합니다.
 * 지불거부자란? 해당 재화의 가치가 없어서가 아니라, 세금 납부 방식이나 설문 시나리오 자체에 대한 반감 때문에 '아니오(0원)'를 선택한 응답자를 말합니다. 이러한 응답자가 포함될 경우 지불용의액(WTP)이 과소 추정되는 등 결과에 편향(Bias)이 발생할 수 있습니다. 이 패키지는 지불거부자가 제거된 유효한 응답 데이터만을 입력으로 가정합니다.

## 주요 기능

  * **로직 완벽 구현**: 주로 사용되는 "로그-로짓(Log-Logit)" 모형($V = a + b \ln(Bid)$)을 그대로 따릅니다.
  * **과정 추적 기능**: 분석의 중간 과정(Process 1\~6)을 속성으로 제공합니다.
  * **통계적 추론**: 최적화 결과뿐만 아니라 헤시안 행렬, 공분산 행렬, 표준오차, t값, p값 등 상세 통계량을 제공합니다.
  * **이중 언어 지원**: 코드 내 모든 설명이 **한국어**와 **영어**로 병기되어 있습니다.

## 설치 방법

PyPI를 통해 최신 버전을 바로 설치할 수 있습니다.

```bash
pip install cvm-simple
```

Google Colab 또는 Jupyter Notebook 사용 시: 명령어 앞에 느낌표(!)를 붙여서 실행해 주세요:
```bash
!pip install cvm-simple
```

## 사용 예시

도시 공원 환경 개선을 위한 지불의사금액(WTP)을 추정하는 예시입니다.

```python
import pandas as pd
from cvm_simple import SingleBoundedLogit

# 1. 데이터 준비 (도시 공원 환경 개선을 위한 월 세금)
# 제시액: 월 추가 세금(원), 찬성/반대: 응답자 수
df = pd.DataFrame({
    '제시액': [1000, 3000, 5000, 10000, 20000],
    '찬성': [85, 70, 55, 35, 15],
    '반대': [15, 30, 45, 65, 85]
})

# 2. 모델 학습
model = SingleBoundedLogit()
model.fit(df, bid_col='제시액', yes_col='찬성', no_col='반대', max_bid_integral=50000)

# 3. 종합 결과 리포트 (AIC, 유의성 별 표시 포함)
model.summary()

# 4. 그래프용 데이터 확인 (실측치 vs 예측치)
print(model.process_plot_data)

# 5. 95% 신뢰구간 계산 (Krinsky & Robb 시뮬레이션)
model.calculate_kr_confidence_interval(n_sim=1000)
```

## 계산 과정 추적 

`model.processN` 속성을 호출하여 각 단계별 계산 값을 확인할 수 있습니다.

| 속성 (Property)                       | 설명                | 엑셀 대응 항목 |
|:------------------------------------|:------------------| :--- |
| `model.process1_log_transformation` | 제시액 로그 변환         | `ln(Bid)` 열 |
| `model.process2_utility`            | 효용 함수($V$) 계산 값   | 효용 계산 수식 |
| `model.process3_probability`        | 추정 구매 확률($P$)     | `Estimate` (추정 확률) 열 |
| `model.process4_likelihood`         | 로그우도 기여분          | `SumProduct` 내부 구성요소 |
| `model.process5_wtp`                | 중앙값 및 절사 평균 WTP   | 우측 WTP 계산 영역 |
| `model.process6_statistics`         | 헤시안 및 통계적 유의성     | `Laa`, `Lbb`, `표준오차`, `p값` |
| `model.process_plot_data`           | 시각화용 데이터          |  Real vs Estimate 표   |

### 예시: 통계량 검증

```python
print(model.process6_statistics)
```

## 파라미터 가이드

| 파라미터 | 설명 | 권장값 |
|:---------|:-----|:-------|
| `bid_col` | 제시액 컬럼명 | 필수 |
| `yes_col` | 찬성 응답수 컬럼명 | 필수 |
| `no_col` | 반대 응답수 컬럼명 | 필수 |
| `max_bid_integral` | 절사 평균 계산 시 적분 상한 | **최대 제시액의 2~3배** (예: 최대 제시액 20,000원 → 40,000~60,000) |