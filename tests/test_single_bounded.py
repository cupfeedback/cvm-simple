"""
pytest를 사용한 CVM 모델 테스트
실행: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from cvm_simple import SingleBoundedLogit


@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터: 해양 생태계 보전 WTP 설문"""
    return pd.DataFrame({
        'bid': [5, 10, 25, 50, 100],
        'yes': [82, 68, 51, 33, 15],
        'no':  [18, 32, 49, 67, 85]
    })


@pytest.fixture
def fitted_model(sample_data):
    """학습된 모델 fixture"""
    model = SingleBoundedLogit()
    model.fit(sample_data, bid_col='bid', yes_col='yes', no_col='no', max_bid_integral=200)
    return model


class TestModelFitting:
    """모델 학습 관련 테스트"""

    def test_model_fits_without_error(self, sample_data):
        """모델이 에러 없이 학습되는지 테스트"""
        model = SingleBoundedLogit()
        model.fit(sample_data, bid_col='bid', yes_col='yes', no_col='no')
        assert model._params is not None

    def test_custom_column_names(self):
        """커스텀 컬럼명으로 학습되는지 테스트"""
        df = pd.DataFrame({
            'price': [10, 20, 30],
            'accept': [80, 50, 20],
            'reject': [20, 50, 80]
        })
        model = SingleBoundedLogit()
        model.fit(df, bid_col='price', yes_col='accept', no_col='reject')
        assert model._params is not None


class TestWTPCalculation:
    """WTP 계산 관련 테스트"""

    def test_wtp_median_positive(self, fitted_model):
        """중앙값 WTP가 양수인지 테스트"""
        assert fitted_model.wtp_median > 0

    def test_wtp_mean_positive(self, fitted_model):
        """평균 WTP가 양수인지 테스트"""
        assert fitted_model.wtp_mean > 0

    def test_wtp_mean_adj_exists(self, fitted_model):
        """조정된 절사 평균이 계산되는지 테스트"""
        assert hasattr(fitted_model, 'wtp_mean_adj')
        assert fitted_model.wtp_mean_adj > 0


class TestCoefficients:
    """계수 관련 테스트"""

    def test_coefficients_have_expected_signs(self, fitted_model):
        """계수의 부호가 예상대로인지 테스트 (b < 0)"""
        a, b = fitted_model._params
        # 금액이 높을수록 찬성 확률이 낮아지므로 b는 음수여야 함
        assert b < 0

    def test_statistics_calculated(self, fitted_model):
        """통계량이 계산되는지 테스트"""
        assert fitted_model.stats_df is not None
        assert len(fitted_model.stats_df) == 2  # a, b 두 개의 계수

    def test_sample_count_calculated(self, fitted_model):
        """표본 수가 계산되는지 테스트"""
        assert hasattr(fitted_model, 'n_samples')
        assert fitted_model.n_samples == 500  # 100*5 제시액 그룹


class TestProcessProperties:
    """중간 과정 속성 테스트"""

    def test_process1_log_transformation(self, fitted_model):
        """과정 1: 로그 변환 테스트"""
        result = fitted_model.process1_log_transformation
        assert result is not None
        assert 'ln(Bid)' in result.columns

    def test_process2_utility(self, fitted_model):
        """과정 2: 효용 계산 테스트"""
        result = fitted_model.process2_utility
        assert result is not None
        assert 'Utility(V)' in result.columns

    def test_process3_probability(self, fitted_model):
        """과정 3: 확률 계산 테스트"""
        result = fitted_model.process3_probability
        assert result is not None
        assert 'Prob(Yes)' in result.columns
        # 확률은 0~1 사이
        assert (result['Prob(Yes)'] >= 0).all()
        assert (result['Prob(Yes)'] <= 1).all()

    def test_process4_likelihood(self, fitted_model):
        """과정 4: 로그우도 테스트"""
        result = fitted_model.process4_likelihood
        assert result is not None
        assert 'LogL_Contribution' in result.columns

    def test_process5_wtp(self, fitted_model):
        """과정 5: WTP 결과 테스트"""
        result = fitted_model.process5_wtp
        assert result is not None
        assert len(result) == 2  # Median, Mean

    def test_process6_statistics(self, fitted_model):
        """과정 6: 통계량 테스트"""
        result = fitted_model.process6_statistics
        assert result is not None
        assert 'Coefficient' in result.columns
        assert 'p-value' in result.columns

    def test_process_plot_data(self, fitted_model):
        """시각화용 데이터 테스트"""
        result = fitted_model.process_plot_data
        assert result is not None
        assert 'Real' in result.columns
        assert 'Estimate' in result.columns


class TestConfidenceInterval:
    """Krinsky & Robb 신뢰구간 테스트"""

    def test_kr_confidence_interval_returns_dataframe(self, fitted_model):
        """신뢰구간 계산이 DataFrame을 반환하는지 테스트"""
        result = fitted_model.calculate_kr_confidence_interval(n_sim=100)
        assert isinstance(result, pd.DataFrame)

    def test_kr_confidence_interval_has_required_columns(self, fitted_model):
        """신뢰구간 결과에 필요한 컬럼이 있는지 테스트"""
        result = fitted_model.calculate_kr_confidence_interval(n_sim=100)
        assert 'Metric' in result.columns
        assert 'LB' in result.columns
        assert 'UB' in result.columns

    def test_kr_confidence_interval_lb_less_than_ub(self, fitted_model):
        """하한이 상한보다 작은지 테스트"""
        result = fitted_model.calculate_kr_confidence_interval(n_sim=100)
        for _, row in result.iterrows():
            if not np.isinf(row['LB']) and not np.isinf(row['UB']):
                assert row['LB'] <= row['UB']


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_model_before_fit(self):
        """fit() 전 모델 상태 테스트"""
        model = SingleBoundedLogit()
        assert model._params is None
        assert model.wtp_median is None

    def test_process_before_fit(self):
        """fit() 전 process 속성 접근 테스트"""
        model = SingleBoundedLogit()
        result = model.process1_log_transformation
        assert result == "Run .fit() first."
