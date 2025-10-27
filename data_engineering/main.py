from data_pipeline import (
    merge_coco_annotations,
    create_clean_annotations,
    build_pill_library,
    clean_annotations_from_library,
    synthesize_images,
    package_final_dataset
)
from analysis import precise_validate_final_dataset
import os

def main():
    """전체 데이터 엔지니어링 파이프라인을 실행합니다."""
    
    # === Phase 1: 데이터 통합 및 정제 ===
    print("--- Phase 1: 데이터 통합 및 정제 ---")
    merge_coco_annotations('data/train_annotations', 'merged_annotations.json')
    clean_data = create_clean_annotations('merged_annotations.json', 'cleaned_annotations.json')
    
    # === Phase 2: 재료 준비 (알약 라이브러리) ===
    print("\n--- Phase 2: 재료 준비 (알약 라이브러리) ---")
    build_pill_library('merged_annotations.json', 'data/train_images', 'cropped_pills')
    
    # (선택) 여기서 cropped_pills 폴더를 수동으로 정리할 수 있습니다.
    # print("\n알약 라이브러리(cropped_pills) 수동 검수 후 Enter를 누르세요...")
    # input()
    
    clean_annotations_from_library('cropped_pills', 'merged_annotations.json', 'library_annotations.json')

    # === Phase 3: 데이터 합성 ===
    print("\n--- Phase 3: 데이터 합성 ---")
    synthesize_images(
        cleaned_json_path='cleaned_annotations.json',
        library_json_path='library_annotations.json',
        library_folder='cropped_pills',
        backgrounds_folder='backgrounds',
        output_folder='synthetic_dataset_final',
        target_count=200
    )

    # === Phase 4: 최종 패키징 ===
    print("\n--- Phase 4: 최종 패키징 ---")
    package_final_dataset(
        clean_data=clean_data, # Phase 1에서 생성된 올바른 데이터 사용
        synthetic_json_path='synthetic_dataset_final/synthetic_annotations_final.json',
        source_original_images_folder='data/train_images',
        source_synthetic_images_folder='synthetic_dataset_final/images',
        output_folder='final_dataset'
    )

    # === Phase 5: 최종 검증 ===
    print("\n--- Phase 5: 최종 검증 ---")
    precise_validate_final_dataset('final_dataset/final_annotations.json')
    
    print("\n 모든 데이터 엔지니어링 과정이 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main()