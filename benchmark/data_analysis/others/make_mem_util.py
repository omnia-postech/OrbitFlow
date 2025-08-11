import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_graph(csv_file_path, output_path, method_name):
    # CSV 파일 읽기
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"오류: 파일 {csv_file_path}이(가) 존재하지 않습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"CSV 파일 읽기 중 오류 발생: {e}")
        sys.exit(1)

    # step_num으로 정렬 (없으면 인덱스 사용)
    if 'step_num' in df.columns:
        df = df.sort_values('step_num')
        x = df['step_num']
    else:
        df = df.reset_index()
        x = df.index

    # Utilization 계산
    df['utilization'] = df['used_num'] / df['total_num']
    mean_util = df['utilization'].mean()
    max_util = df['utilization'].max()

    # ───────────────────────────────────────────────
    # 🎨 Figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # 선 + fill_between 시각화
    ax.plot(x, df['utilization'], color='#1f77b4', label='Utilization', linewidth=2)
    ax.fill_between(x, df['utilization'], color='#1f77b4', alpha=0.3)

    ax.set_xlabel('Step')
    ax.set_ylabel('Utilization\n(used / total)')
    ax.set_title(f'{method_name} - Memory Utilization\n(Avg: {mean_util:.3f})')
    ax.grid(True)
    ax.legend()

    # 평균값 텍스트로 추가 (옵션)
    ax.text(0.99, 0.01,
            f'Avg: {mean_util:.3f}',
            transform=ax.transAxes,
            fontsize=12,
            ha='right',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"평균 Utilization: {mean_util:.3f} (최대: {max_util:.3f})")
    print(f"✅ 저장 완료: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("사용법: python parse_log_to_csv.py <입력_로그_파일_디렉토리>")
        sys.exit(1)

    input_dir = sys.argv[1]
    csv_file_path = os.path.join(input_dir, "mem_util_output.csv")
    output_path = os.path.join(input_dir, "mem_util_output.png")

    if not os.path.exists(csv_file_path):
        print(f"오류: 입력 파일 {csv_file_path}이(가) 존재하지 않습니다.")
        sys.exit(1)

    # 메서드 이름 추출: 마지막 디렉토리 이름
    method_name = os.path.basename(input_dir.strip('/'))

    plot_graph(csv_file_path, output_path, method_name)
