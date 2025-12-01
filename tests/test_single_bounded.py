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
    """테스트용 샘플 데이터"""
    return pd.DataFrame({
        'bid': [3000, 5000, 8000, 12000, 20000],
        'yes': [57, 63, 45, 36, 29],
        'no':  [18, 11, 27, 33, 43]
    })


class TestSingleBoundedLogit:

    def test_model_fits_without_error(self, sample_data):
        """모델이 에러 없이 학습되는지 테스트"""
        model = SingleBoundedLogit()
        model.fit(sample_data, bid_col='bid', yes_col='yes', no_col='no')
        assert model._params is not None

    def test_wtp_median_positive(self, sample_data):
        """중앙값 WTP가 양수인지 테스트"""
        model = SingleBoundedLogit()
        model.fit(sample_data)
        assert model.wtp_median > 0

    def test_wtp_mean_positive(self, sample_data):
        """평균 WTP가 양수인지 테스트"""
        model = SingleBoundedLogit()
        model.fit(sample_data)
        assert model.wtp_mean > 0

    def test_coefficients_have_expected_signs(self, sample_data):
        """계수의 부호가 예상대로인지 테스트 (b < 0)"""
        model = SingleBoundedLogit()
        model.fit(sample_data)
        a, b = model._params
        # 금액이 높을수록 찬성 확률이 낮아지므로 b는 음수여야 함
        assert b < 0

    def test_statistics_calculated(self, sample_data):
        """통계량이 계산되는지 테스트"""
        model = SingleBoundedLogit()
        model.fit(sample_data)
        assert model.stats_df is not None
        assert len(model.stats_df) == 2  # a, b 두 개의 계수

    def test_process_properties_work(self, sample_data):
        """중간 과정 속성들이 작동하는지 테스트"""
        model = SingleBoundedLogit()
        model.fit(sample_data)

        assert model.process1_log_transformation is not None
        assert model.process2_utility is not None
        assert model.process3_probability is not None
        assert model.process4_likelihood is not None
        assert model.process5_wtp is not None
