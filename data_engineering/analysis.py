import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Matplotlib 폰트 설정 코드
try:
    plt.rcParams['font.family'] = 'NanumGothic' # Windows
except:
    plt.rcParams['font.family'] = 'AppleGothic' # Mac
plt.rcParams['axes.unicode_minus'] = False

# 클래스 분포 분석
def analyze_class_distribution(json_path, output_image_path):
    """
    COCO 형식의 JSON 파일에서 클래스 분포를 분석하고 시각화합니다.

    Args:
        json_path (str): 분석할 JSON 파일 경로.
        output_image_path (str): 생성된 차트를 저장할 이미지 파일 경로.
    """
    if not os.path.exists(json_path):
        print(f"오류: '{json_path}' 파일을 찾을 수 없습니다.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 데이터 로드 및 DataFrame 변환
    annotations_df = pd.DataFrame(data.get('annotations', []))
    categories_df = pd.DataFrame(data.get('categories', []))

    if annotations_df.empty or categories_df.empty:
        print("오류: JSON 파일에 'annotations' 또는 'categories' 정보가 부족합니다.")
        return

    # 가독성을 위해 카테고리 DataFrame의 컬럼명 변경
    categories_df.rename(columns={'id': 'category_id', 'name': 'category_name'}, inplace=True)

    # 어노테이션과 카테고리 정보 병합
    class_distribution = annotations_df.merge(categories_df, on='category_id')

    # 클래스별 객체 수 계산 및 정렬
    class_counts = class_distribution['category_name'].value_counts().reset_index()
    class_counts.columns = ['클래스', '객체 수']
    class_counts_sorted = class_counts.sort_values(by='객체 수', ascending=False)

    print("--- 📊 클래스별 객체 수 분석 결과 ---")
    print(class_counts_sorted.to_string(index=False))
    print("-" * 40)

    # 시각화
    plt.figure(figsize=(12, 16)) # 클래스 수가 많으므로 세로 길이를 늘림
    sns.barplot(x='객체 수', y='클래스', data=class_counts_sorted, palette='viridis_r')
    
    plt.title('정리된 데이터의 클래스별 객체 수 분포', fontsize=18, pad=20)
    plt.xlabel('객체(Bounding Box) 수', fontsize=14)
    plt.ylabel('클래스 (알약 종류)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # 각 막대 위에 숫자 표시
    for index, value in enumerate(class_counts_sorted['객체 수']):
        plt.text(value, index, f' {value}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_image_path)
    
    print(f"✅ 클래스 분포 시각화 차트를 '{output_image_path}' 파일로 저장했습니다.")

# 데이터셋 유효성 검사
def precise_validate_final_dataset(json_path):
    """최종 데이터셋을 정밀 검증하는 함수."""
    # (이전과 동일한 검증 함수)
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    images_df = pd.DataFrame(data['images'])
    annotations_df = pd.DataFrame(data['annotations'])
    actual_counts = annotations_df['image_id'].value_counts().reset_index()
    actual_counts.columns = ['id', 'actual_count']
    analysis_df = pd.merge(images_df, actual_counts, on='id', how='left')
    analysis_df['actual_count'] = analysis_df['actual_count'].fillna(0).astype(int)
    analysis_df['expected_count'] = analysis_df['file_name'].apply(get_expected_count_from_filename)
    mismatched_df = analysis_df[analysis_df['actual_count'] != analysis_df['expected_count']]
    
    print("--- 🔬 정밀 무결성 검사 결과 ---")
    if mismatched_df.empty:
        print("✅ 완벽합니다! 모든 이미지의 실제 바운딩 박스 개수가 파일명과 일치합니다.")
    else:
        print(f"🚨 오류: {len(mismatched_df)}개 이미지에서 개수 불일치가 발견되었습니다.")
        # ... 오류 출력 ...
    print("-" * 50)

def get_expected_count_from_filename(filename):
    """파일명에서 기대 알약 개수를 반환합니다."""
    if filename.startswith('synthetic_'):
        return 4 # 합성 이미지는 4개로 가정
    try:
        key_part = filename.split('_')[0]
        num_parts = len(key_part.split('-'))
        return num_parts - 1
    except Exception:
        return 0