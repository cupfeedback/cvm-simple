"""
CVM Single-Bounded Logit Model Demo Script

Example: Estimating WTP for wetland conservation program
(습지 보전 프로그램에 대한 지불의사금액 추정 예시)
"""
import pandas as pd
from cvm_simple import SingleBoundedLogit

# Sample data: Wetland conservation program (annual household tax)
# 습지 보전을 위한 연간 가구당 추가 세금 설문 결과
sample_data = pd.DataFrame({
    'bid': [2000, 5000, 10000, 15000, 30000],
    'yes': [78, 62, 45, 28, 12],
    'no':  [22, 38, 55, 72, 88]
})

print("=" * 50)
print("Sample Data:")
print("=" * 50)
print(sample_data)
print()

# 모델 생성 및 학습
# max_bid_integral: 절사 평균 계산 시 적분 상한 (보통 최대 제시액의 2~3배)
model = SingleBoundedLogit()
model.fit(sample_data, bid_col='bid', yes_col='yes', no_col='no', max_bid_integral=60000)

# 결과 요약 출력
print()
model.summary()

# 중간 과정 확인 (선택)
print("\n[Process 1] Log Transformation:")
print(model.process1_log_transformation)
print("\n[Process 2] utility:")
print(model.process2_utility)
print("\n[Process 3] probability:")
print(model.process3_probability)
print("\n[Process 4] likelihood:")
print(model.process4_likelihood)
print("\n[Process 5] WTP Results:")
print(model.process5_wtp)
print("\n[Process 6] statistics:")
print(model.process6_statistics)

# 1. 그래프용 표 데이터 확인
print("\n[그래프용 데이터 (Excel: Real vs Estimate)]")
plot_data = model.process_plot_data
print(plot_data)

# 2. Krinsky & Robb 신뢰구간 계산 (1000번 시뮬레이션)
print("\n[Krinsky & Robb Simulated Confidence Intervals]")
kr_df = model.calculate_kr_confidence_interval(n_sim=1000, ci=0.95)
print(kr_df.round(1))

# 3. 그래프 그려보기 (matplotlib 필요)
try:
    import matplotlib.pyplot as plt

    # 데이터 준비
    bids = plot_data[sample_data.columns[0]]  # 제시액
    real = plot_data['Real']  # 실측치 (점)
    est = plot_data['Estimate']  # 추정치 (선)

    # 그래프 그리기
    plt.figure(figsize=(8, 5))

    # (1) 모델 추정선 (부드러운 곡선을 위해 점을 더 촘촘하게 찍을 수도 있지만, 여기선 직선 연결)
    plt.plot(bids, est, 'b-', label='Estimate (Model)', marker='o')  # 파란 선

    # (2) 실제 데이터 점
    plt.plot(bids, real, 'rs', label='Real (Observed)')  # 빨간 네모 점

    plt.title('CVM: Wetland Conservation WTP')
    plt.xlabel('Bid Amount (KRW)')
    plt.ylabel('Probability of Yes')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("\n[System] 그래프가 팝업되었습니다.")

except ImportError:
    print("\n[System] matplotlib가 설치되지 않아 그래프를 그릴 수 없습니다.")