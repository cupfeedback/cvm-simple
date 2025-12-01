import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


class SingleBoundedLogit:
    """
    [ENG] Single-Bounded Dichotomous Choice (SBDC) CVM Model (Log-Logit)
    This class implements the logic of the 'Solver' function in Excel using Python.
    It provides intermediate calculation steps (processes) for educational and verification purposes.

    [KOR] 단일양분선택형(SBDC) CVM 모델 (로그-로짓 모형)
    이 클래스는 엑셀의 '해 찾기' 기능 로직을 파이썬으로 구현한 것입니다.
    교육 및 검증 목적으로 중간 계산 과정(Process)을 확인할 수 있는 기능을 제공합니다.
    """

    def __init__(self):
        # [ENG] Internal data storage
        # [KOR] 내부 데이터 저장 변수
        self._data = None
        self._params = None
        self._optimization_result = None
        self._cols = {}
        self._hessian = None
        self._var_cov = None

        # [ENG] Result storage
        # [KOR] 결과 저장 변수
        self.wtp_median = None
        self.wtp_mean = None
        self.stats_df = None

    def fit(self, df, bid_col='bid', yes_col='yes', no_col='no', max_bid_integral=20000):
        """
        [ENG] Fit the model to the provided data.
        :param df: Input DataFrame (pandas)
        :param bid_col: Column name for Bid amount
        :param yes_col: Column name for Yes count
        :param no_col: Column name for No count
        :param max_bid_integral: Maximum bid amount for calculating Truncated Mean WTP (Integration limit)

        [KOR] 데이터를 사용하여 모델을 학습(추정)합니다.
        :param df: 입력 데이터프레임 (pandas)
        :param bid_col: 제시 금액 컬럼명
        :param yes_col: 찬성 응답수 컬럼명
        :param no_col: 반대 응답수 컬럼명
        :param max_bid_integral: 절사 평균 WTP 계산 시 적분할 최대 금액 상한선
        """
        self._cols = {'bid': bid_col, 'yes': yes_col, 'no': no_col}
        self._data = df.copy()
        self._max_bid_integral = max_bid_integral

        # [Process 1] Log Transformation
        # [ENG] Avoid log(0) error by replacing 0 with epsilon (1e-10)
        # [KOR] 0이 입력될 경우 로그 에러 방지를 위해 1e-10으로 대체
        self._data['ln_bid'] = np.log(self._data[bid_col].replace(0, 1e-10))

        # [ENG] Optimization (Minimize Negative Log-Likelihood)
        # [KOR] 최적화 수행 (음의 로그우도 최소화 = 해 찾기)
        # Initial guess: alpha=1.0, beta=-1.0
        initial_guess = [1.0, -1.0]
        res = minimize(self._objective_function, initial_guess, method='BFGS')

        self._optimization_result = res
        self._params = res.x

        # [Process 5] Calculate WTP (Median & Mean)
        # [KOR] WTP 산출 (중앙값 및 평균값)
        self._calculate_wtp_values()

        # [Process 6] Calculate Statistics (SE, t-val, p-val)
        # [KOR] 통계량 산출 (표준오차, t값, p값)
        self._calculate_statistics()

        if res.success:
            print("[System] Analysis Complete. Optimization converged successfully.")
        else:
            print("[System] Analysis Complete. (Optimization did not fully converge, but results are usable.)")
        print("[System] Use '.summary()' to see the report.")

    def _objective_function(self, params):
        """
        [ENG] Objective function for minimization (Negative Log-Likelihood)
        [KOR] 최소화를 위한 목적 함수 (음의 로그우도 함수)
        """
        a, b = params
        bid_col, yes_col, no_col = self._cols['bid'], self._cols['yes'], self._cols['no']

        # Utility (V) = a + b * ln(Bid)
        utility = a + b * self._data['ln_bid']

        # Probability (P) = 1 / (1 + exp(-V))
        prob_yes = 1 / (1 + np.exp(-utility))

        # [ENG] Clip probabilities to avoid log(0) errors
        # [KOR] 로그 계산 오류 방지를 위해 확률 범위를 0과 1 사이로 제한
        epsilon = 1e-10
        prob_yes = np.clip(prob_yes, epsilon, 1 - epsilon)

        # Log-Likelihood Sum = Sum( Yes * ln(P) + No * ln(1-P) )
        ll_sum = (self._data[yes_col] * np.log(prob_yes) +
                  self._data[no_col] * np.log(1 - prob_yes)).sum()

        return -ll_sum  # Return negative for minimization

    def _calculate_wtp_values(self):
        """
        [ENG] Calculate Median, Truncated Mean, and Adjusted Truncated Mean WTP
        [KOR] 중앙값, 절사 평균, 그리고 조정된 절사 평균 WTP 계산
        """
        a, b = self._params

        # 1. Median WTP = exp(-a/b)
        self.wtp_median = np.exp(-a / b)

        # 2. Mean WTP (Numerical Integration)
        # [ENG] Create range from 0 to max_bid with step 10
        # [KOR] 0부터 최대금액까지 10원 단위로 구간 생성 (구분구적법)
        steps = np.arange(0, self._max_bid_integral + 10, 10)
        steps_log = np.log(np.maximum(steps, 1))  # log(0) protection

        probs = 1 / (1 + np.exp(-(a + b * steps_log)))
        self.wtp_mean = np.sum(probs * 10)  # Area = Height(Prob) * Width(10)

        # [NEW] 3. Adjusted Truncated Mean WTP
        # 공식: Truncated Mean / (1 - Prob(Yes at MaxBid))
        # 의미: WTP가 최대 제시액을 넘지 않는다고 가정하고 확률을 100%로 보정함
        max_bid_log = np.log(max(self._max_bid_integral, 1e-10))
        prob_at_max = 1 / (1 + np.exp(-(a + b * max_bid_log)))  # 최대금액에서의 Yes 확률
        cdf_at_max = 1 - prob_at_max  # 최대금액보다 작을 확률 (누적확률)

        if cdf_at_max > 0:
            self.wtp_mean_adj = self.wtp_mean / cdf_at_max
        else:
            self.wtp_mean_adj = np.inf

    def _calculate_statistics(self):
        """
        [ENG] Calculate Hessian Matrix and Standard Errors
        [KOR] 헤시안 행렬 및 표준오차 계산
        """
        a, b = self._params
        bid = self._data['ln_bid'].values

        # [NEW] 표본 수(n) 계산
        self.n_samples = self._data[self._cols['yes']].sum() + self._data[self._cols['no']].sum()

        n_total = self._data[self._cols['yes']] + self._data[self._cols['no']]

        # Re-calculate probability
        v = a + b * bid
        p = 1 / (1 + np.exp(-v))

        # [ENG] Calculate weights for Hessian (Information Matrix)
        # [KOR] 헤시안 계산을 위한 가중치 (정보 행렬)
        # Formula: w = N * P * (1-P)
        w = n_total * p * (1 - p)

        # Hessian Elements (Negative Log-Likelihood derivatives)
        laa = -np.sum(w)  # dLL/da^2
        lbb = -np.sum(w * bid ** 2)  # dLL/db^2
        lab = -np.sum(w * bid)  # dLL/dadb

        hessian = np.array([[laa, lab],
                            [lab, lbb]])
        self._hessian = hessian

        # [ENG] Variance-Covariance Matrix = Inverse of (-Hessian)
        # [KOR] 공분산 행렬 = (-헤시안)의 역행렬
        try:
            var_cov = np.linalg.inv(-hessian)
        except np.linalg.LinAlgError:
            var_cov = np.zeros((2, 2))  # Handle singular matrix

        self._var_cov = var_cov

        # Standard Errors (sqrt of diagonal elements)
        se_a = np.sqrt(var_cov[0, 0])
        se_b = np.sqrt(var_cov[1, 1])

        # t-values
        t_a = a / se_a
        t_b = b / se_b

        # p-values (Two-tailed test)
        p_a = 2 * (1 - norm.cdf(abs(t_a)))
        p_b = 2 * (1 - norm.cdf(abs(t_b)))

        # [NEW] 별 표시 함수 (Significance Stars)
        def get_star(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            elif p < 0.1:
                return "."
            else:
                return ""

        self.stats_df = pd.DataFrame({
            'Variable': ['Constant (a)', 'Log-Bid (b)'],
            'Coefficient': [a, b],
            'Std.Error': [se_a, se_b],
            't-value': [t_a, t_b],
            'p-value': [p_a, p_b],
            'Sig.': [get_star(p_a), get_star(p_b)]  # 별 컬럼 추가
        })

        # [NEW] 비절사 평균 (Untruncated Mean) 계산 - 비교용
        # 공식: Mean = exp(-a/b) * (pi/|b|) / sin(pi/|b|)
        # 단, |b| > 1 이어야 수렴함. 그 외에는 발산하거나 매우 큼.
        if abs(b) > 1:
            try:
                import math
                term1 = np.exp(-a / b)
                term2 = (np.pi / abs(b)) / np.sin(np.pi / abs(b))
                self.wtp_mean_untruncated = term1 * term2
            except:
                self.wtp_mean_untruncated = np.inf
        else:
            self.wtp_mean_untruncated = np.inf

    def calculate_kr_confidence_interval(self, n_sim=1000, ci=0.95):
        """
        [ENG] Calculate Krinsky and Robb (1986) Simulated Confidence Intervals
        :param n_sim: Number of simulations (default 1000)
        :param ci: Confidence level (default 0.95 for 95%)

        [KOR] Krinsky and Robb (1986) 시뮬레이션 신뢰구간 계산
        :param n_sim: 시뮬레이션 횟수 (보통 1000~5000)
        :param ci: 신뢰수준 (0.95는 95% 신뢰구간)
        """
        if self._var_cov is None:
            print("Run .fit() first to calculate Variance-Covariance Matrix.")
            return

        # 1. 시뮬레이션 데이터 생성 (가상의 a, b 1000개 뽑기)
        # mean=추정된 계수, cov=공분산 행렬
        beta_sim = np.random.multivariate_normal(
            mean=self._params,
            cov=self._var_cov,
            size=n_sim
        )

        a_sim = beta_sim[:, 0]
        b_sim = beta_sim[:, 1]

        # 2. 각 시뮬레이션 별 WTP 계산

        # (1) Median WTP 시뮬레이션
        median_sim = np.exp(-a_sim / b_sim)

        # (2) Mean WTP (Untruncated) 시뮬레이션
        # |b| > 1 인 경우에만 계산 (아니면 Inf)
        mean_sim = []
        for a_s, b_s in zip(a_sim, b_sim):
            if abs(b_s) > 1:
                term1 = np.exp(-a_s / b_s)
                term2 = (np.pi / abs(b_s)) / np.sin(np.pi / abs(b_s))
                mean_sim.append(term1 * term2)
            else:
                mean_sim.append(np.inf)
        mean_sim = np.array(mean_sim)

        # (3) Truncated Mean WTP 시뮬레이션 (적분)
        # 벡터화 연산으로 속도 최적화
        steps = np.arange(0, self._max_bid_integral + 100, 100)  # 100원 단위
        steps_log = np.log(np.maximum(steps, 1))

        # Broadcasting: (N_sim, 1) + (N_sim, 1) * (1, N_steps) -> (N_sim, N_steps)
        v_sim = a_sim[:, None] + b_sim[:, None] * steps_log[None, :]
        probs_sim = 1 / (1 + np.exp(-v_sim))
        trunc_mean_sim = np.sum(probs_sim * 100, axis=1)  # 면적 합계

        # [NEW] Adjusted Truncated Mean 시뮬레이션
        # 각 시뮬레이션 별 MaxBid에서의 Yes 확률 계산
        max_bid_log = np.log(self._max_bid_integral)
        v_max_sim = a_sim + b_sim * max_bid_log
        prob_max_sim = 1 / (1 + np.exp(-v_max_sim))  # P(Yes|MaxBid)
        cdf_max_sim = 1 - prob_max_sim  # P(WTP < MaxBid)

        # 0으로 나누기 방지
        adj_trunc_mean_sim = np.divide(trunc_mean_sim, cdf_max_sim,
                                       out=np.full_like(trunc_mean_sim, np.inf),
                                       where=cdf_max_sim != 0)

        # 3. 백분위수(Percentile)를 이용해 LB, UB 계산
        alpha = (1 - ci) / 2
        lower_p = alpha * 100  # 2.5%
        upper_p = (1 - alpha) * 100  # 97.5%

        def get_ci_row(sim_values, label):
            # Inf가 포함된 경우 처리
            valid_values = sim_values[~np.isinf(sim_values)]
            if len(valid_values) < len(sim_values) * 0.5:  # 절반 이상이 Inf면
                est, lb, ub = np.inf, np.inf, np.inf
            else:
                est = np.median(sim_values)  # 시뮬레이션의 중앙값 사용
                lb = np.percentile(valid_values, lower_p)
                ub = np.percentile(valid_values, upper_p)
            return {'Metric': label, 'Estimate': est, 'LB': lb, 'UB': ub}

        # 결과 정리
        results = [
            get_ci_row(mean_sim, "Mean"),
            get_ci_row(trunc_mean_sim, "Truncated Mean"),
            get_ci_row(adj_trunc_mean_sim, "Adjusted Truncated Mean"),  # 추가됨
            get_ci_row(median_sim, "Median")
        ]

        self.kr_results = pd.DataFrame(results)
        return self.kr_results

    # ==========================================================================
    # Traceability Properties (Process 1~6)
    # [ENG] Properties to view intermediate calculation steps
    # [KOR] 중간 계산 과정을 확인하기 위한 속성들
    # ==========================================================================

    @property
    def process1_log_transformation(self):
        """
        [ENG] Process 1: View Log-transformed Bids
        [KOR] 과정 1: 로그 변환된 제시 금액 확인
        """
        if self._data is None: return "Run .fit() first."
        res = self._data[[self._cols['bid']]].copy()
        res['ln(Bid)'] = self._data['ln_bid']
        return res

    @property
    def process2_utility(self):
        """
        [ENG] Process 2: View Utility Calculation (V = a + b * ln(Bid))
        [KOR] 과정 2: 효용 함수 계산 값 확인
        """
        if self._params is None: return "Run .fit() first."
        a, b = self._params
        res = self.process1_log_transformation.copy()
        res['Constant(a)'] = a
        res['Slope(b)'] = b
        res['Utility(V)'] = a + b * res['ln(Bid)']
        return res

    @property
    def process3_probability(self):
        """
        [ENG] Process 3: View Estimated Probability (P = 1 / (1 + exp(-V)))
        [KOR] 과정 3: 추정 구매 확률 확인
        """
        if self._params is None: return "Run .fit() first."
        res = self.process2_utility.copy()
        res['Prob(Yes)'] = 1 / (1 + np.exp(-res['Utility(V)']))
        res['Prob(No)'] = 1 - res['Prob(Yes)']
        return res[['ln(Bid)', 'Utility(V)', 'Prob(Yes)', 'Prob(No)']]

    @property
    def process4_likelihood(self):
        """
        [ENG] Process 4: View Log-Likelihood Contribution
        [KOR] 과정 4: 로그우도 기여분 확인 (SumProduct components)
        """
        if self._params is None: return "Run .fit() first."
        res = self.process3_probability.copy()
        yes_c, no_c = self._cols['yes'], self._cols['no']

        res['Yes_Count'] = self._data[yes_c]
        res['No_Count'] = self._data[no_c]
        res['LogL_Contribution'] = (res['Yes_Count'] * np.log(res['Prob(Yes)'] + 1e-10) +
                                    res['No_Count'] * np.log(res['Prob(No)'] + 1e-10))
        return res[['ln(Bid)', 'Prob(Yes)', 'Yes_Count', 'No_Count', 'LogL_Contribution']]

    @property
    def process5_wtp(self):
        """
        [ENG] Process 5: View Final WTP Results
        [KOR] 과정 5: 최종 WTP 산출 결과 확인
        """
        if self._params is None: return "Run .fit() first."
        return pd.DataFrame({
            'Metric': ['Median WTP', 'Mean WTP (Truncated)'],
            'Value': [self.wtp_median, self.wtp_mean],
            'Note': ['exp(-a/b)', f'Integration up to {self._max_bid_integral}']
        })

    @property
    def process6_statistics(self):
        """
        [ENG] Process 6: View Hessian, Covariance, and Statistical Inference
        [KOR] 과정 6: 헤시안, 공분산 행렬 및 통계적 검정 결과
        """
        if self._params is None: return "Run .fit() first."
        print("\n[Hessian Matrix] (Excel: Laa, Lab, Lbb)")
        print(self._hessian)
        print("\n[Variance-Covariance Matrix] (Excel: Covariance Table)")
        print(self._var_cov)
        print("\n[Statistical Inference]")
        return self.stats_df

    @property
    def process_plot_data(self):
        """
        [ENG] Generate data for plotting (Actual vs Predicted)

        [KOR] 시각화(그래프)를 위한 데이터 생성 (실측치 vs 예측치)
        """
        if self._params is None: return "Run .fit() first."

        # 1. 기본 데이터 가져오기
        plot_df = self._data[[self._cols['bid']]].copy()
        bid_col = self._cols['bid']
        yes_col = self._cols['yes']
        no_col = self._cols['no']

        # 2. Real (실측 확률) 계산: Yes / (Yes + No)
        # [KOR] 실제 설문 결과 비율 (점 그래프용)
        total_count = self._data[yes_col] + self._data[no_col]
        plot_df['Real'] = self._data[yes_col] / total_count

        # 3. Estimate (추정 확률) 계산
        # [KOR] 모델이 예측한 확률 곡선 (선 그래프용)
        a, b = self._params
        ln_bid = self._data['ln_bid']
        utility = a + b * ln_bid
        plot_df['Estimate'] = 1 / (1 + np.exp(-utility))

        # 4. 정렬 (금액 순서대로)
        plot_df = plot_df.sort_values(by=bid_col).reset_index(drop=True)

        return plot_df

    def summary(self):
        """
        [ENG] Print a comprehensive summary report
        [KOR] 종합 분석 결과 리포트 출력
        """
        if self._params is None:
            print("Model not fitted. Please run .fit() first.")
            return

        print("=" * 65)
        print("   CVM Single-Bounded Log-Logit Model Results")
        print("=" * 65)
        # [NEW] 표본 수(n) 출력
        print(f"Number of Obs (n): {self.n_samples}")
        print(f"Log-Likelihood : {-self._optimization_result.fun:.4f}")
        print(f"AIC            : {2 * 2 + 2 * self._optimization_result.fun:.4f}")
        print("-" * 65)
        print(self.stats_df.round(6).to_string(index=False))
        print("-" * 65)
        print("Note: *** p<0.001, ** p<0.01, * p<0.05")
        print("-" * 65)
        print(f"Median WTP     : {self.wtp_median:,.1f}")
        print(f"Mean WTP       : {self.wtp_mean:,.1f} (Truncated at {self._max_bid_integral})")

        # [NEW] 엑셀의 그 엄청난 숫자(비절사 평균)도 참고용으로 보여줌
        if self.wtp_mean_untruncated != np.inf:
            print(f"Mean WTP (Raw)   : {self.wtp_mean_untruncated:,.1f} (Theoretical, Do not use for policy)")
        else:
            print(f"Mean WTP (Raw)   : Infinite (Due to |b| <= 1)")

        print("=" * 65)