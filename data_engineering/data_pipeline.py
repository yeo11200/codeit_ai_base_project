import os
import json
import shutil
import cv2
import random
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

# merge_coco_annotations
def merge_coco_annotations(root_folder, output_file):
    """
    지정된 폴더와 그 하위 폴더의 모든 COCO JSON 파일을 병합합니다.

    Args:
        root_folder (str): 검색을 시작할 최상위 폴더 경로
        output_file (str): 병합된 결과를 저장할 파일 이름
    """
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_ids = set()
    image_ids = set()
    annotation_id_counter = 1

    # 지정된 폴더 및 하위 폴더 탐색
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.json'):
                json_path = os.path.join(dirpath, filename)
                print(f"처리 중: {json_path}")

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 카테고리 병합 (중복 방지)
                    if 'categories' in data:
                        for category in data['categories']:
                            if category['id'] not in category_ids:
                                merged_data['categories'].append(category)
                                category_ids.add(category['id'])

                    # 이미지 정보 병합 (중복 방지)
                    if 'images' in data:
                        for image in data['images']:
                            if image['id'] not in image_ids:
                                merged_data['images'].append(image)
                                image_ids.add(image['id'])

                    # 어노테이션 병합 (ID 재설정)
                    if 'annotations' in data:
                        for ann in data['annotations']:
                            ann['id'] = annotation_id_counter
                            merged_data['annotations'].append(ann)
                            annotation_id_counter += 1

                except Exception as e:
                    print(f"파일 처리 중 오류 발생 '{json_path}': {e}")

    # 카테고리 이름순으로 정렬 (선택 사항)
    merged_data['categories'] = sorted(merged_data['categories'], key=lambda x: x['name'])

    # 병합된 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print("-" * 30)
    print(f"병합 완료! 총 {len(merged_data['images'])}개의 이미지, {len(merged_data['annotations'])}개의 어노테이션을")
    print(f"'{output_file}' 파일에 저장했습니다.")
    print(f"총 클래스 수: {len(merged_data['categories'])}")
    
# "깨끗한 어노테이션 생성" 함수
def create_clean_annotations(original_merged_json, clean_json_output):
    """
    원본 어노테이션에서 파일명 규칙과 bbox 개수가 일치하는 '깨끗한' 데이터만 선별하여
    새로운 어노테이션 파일을 생성합니다.
    """
    print("\n-> 올바른 로직으로 깨끗한 원본 데이터를 선별합니다...")
    with open(original_merged_json, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)

    images_df = pd.DataFrame(merged_data['images'])
    annotations_df = pd.DataFrame(merged_data['annotations'])
    
    actual_counts = annotations_df['image_id'].value_counts().reset_index()
    actual_counts.columns = ['id', 'actual_count']
    analysis_df = pd.merge(images_df, actual_counts, on='id', how='left')
    analysis_df['actual_count'] = analysis_df['actual_count'].fillna(0).astype(int)
    analysis_df['expected_count'] = analysis_df['file_name'].apply(
        lambda f: len(f.split('_')[0].split('-')) - 1
    )
    
    mismatched_df = analysis_df[analysis_df['actual_count'] != analysis_df['expected_count']]
    
    # 이미지의 'id' 컬럼을 기준으로 올바른 ID 집합을 만듭니다.
    correct_image_ids = set(analysis_df['id']) - set(mismatched_df['id'])
    
    clean_data = {
        "images": [img for img in merged_data['images'] if img['id'] in correct_image_ids],
        "annotations": [ann for ann in merged_data['annotations'] if ann['image_id'] in correct_image_ids],
        "categories": merged_data['categories']
    }
    
    with open(clean_json_output, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=4)

    print(f"-> {len(clean_data['images'])}개의 깨끗한 원본을 선별하여 '{clean_json_output}'에 저장했습니다.")
    return clean_data # 후속 작업을 위해 데이터 반환

# build_pill_library
def build_pill_library(json_path, image_folder_path, output_base_folder):
    """
    어노테이션 정보를 바탕으로 원본 이미지에서 개별 알약 이미지를 잘라내어
    클래스별로 정리된 '알약 라이브러리'를 구축합니다.
    """
    if not os.path.exists(json_path):
        print(f"오류: '{json_path}' 파일을 찾을 수 없습니다.")
        return
    if not os.path.exists(image_folder_path):
        print(f"오류: 이미지 폴더 '{image_folder_path}'를 찾을 수 없습니다.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 빠른 조회를 위해 이미지 ID와 파일명을 매핑하는 딕셔너리 생성
    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    # 카테고리 ID와 클래스명을 매핑하는 딕셔너리 생성
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    # 기본 출력 폴더 생성
    os.makedirs(output_base_folder, exist_ok=True)
    print(f"'{output_base_folder}' 폴더에 알약 라이브러리를 생성합니다.")

    # 각 카테고리(클래스)별로 하위 폴더 생성
    for cat_name in category_id_to_name.values():
        os.makedirs(os.path.join(output_base_folder, cat_name), exist_ok=True)

    # 모든 어노테이션을 순회하며 이미지 자르기
    for ann in tqdm(data['annotations'], desc="알약 이미지 추출 중"):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox'] # [x, y, width, height]

        # 이미지 파일명과 클래스명 조회
        filename = image_id_to_filename.get(image_id)
        cat_name = category_id_to_name.get(category_id)

        if not filename or not cat_name:
            continue

        try:
            # 원본 이미지 열기
            img_path = os.path.join(image_folder_path, filename)
            with Image.open(img_path) as img:
                # BBox 좌표를 PIL의 crop 형식 (left, upper, right, lower)으로 변환
                x, y, w, h = bbox
                cropped_img = img.crop((x, y, x + w, y + h))

                # 잘라낸 이미지 저장 (파일 이름은 어노테이션 ID로 하여 중복 방지)
                output_filename = f"pill_{ann['id']}.png"
                output_path = os.path.join(output_base_folder, cat_name, output_filename)
                cropped_img.save(output_path)
                
        except FileNotFoundError:
            # print(f"경고: 이미지 파일을 찾을 수 없습니다 - {img_path}")
            pass
        except Exception as e:
            print(f"이미지 처리 중 오류 발생 '{filename}': {e}")
            
    print("\n--- ✅ Phase 1 완료 ---")
    print("모든 바운딩 박스를 개별 이미지로 추출하여 '디지털 알약 라이브러리'를 성공적으로 구축했습니다.")

# clean_annotations_from_library
def clean_annotations_from_library(library_base_folder, original_json_path, cleaned_json_path):
    """
    실제 파일 시스템에 존재하는 크롭된 알약 이미지들을 기준으로
    원본 어노테이션 파일을 정리합니다.
    """
    print("--- Phase 1: 어노테이션 파일 정리 시작 ---")
    
    # 1. 실제 존재하는 알약 파일로부터 유효한 어노테이션 ID 목록 생성
    valid_annotation_ids = set()
    for dirpath, _, filenames in os.walk(library_base_folder):
        for filename in filenames:
            if filename.startswith('pill_') and filename.endswith('.png'):
                try:
                    # 'pill_{ann_id}.png' 형식에서 ann_id 추출
                    ann_id = int(filename.split('_')[1].split('.')[0])
                    valid_annotation_ids.add(ann_id)
                except (ValueError, IndexError):
                    continue
    
    print(f"'{library_base_folder}' 폴더에서 유효한 알약 이미지 {len(valid_annotation_ids)}개를 확인했습니다.")

    # 2. 원본 어노테이션 파일 로드
    with open(original_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 3. 유효한 ID를 가진 어노테이션만 필터링
    cleaned_annotations = [
        ann for ann in tqdm(data['annotations'], desc="어노테이션 필터링 중") 
        if ann['id'] in valid_annotation_ids
    ]
    
    # 4. 정리된 어노테이션에 사용된 이미지 ID들만 추림
    valid_image_ids = {ann['image_id'] for ann in cleaned_annotations}
    
    # 5. 유효한 이미지 ID를 가진 이미지 정보만 필터링
    cleaned_images = [img for img in data['images'] if img['id'] in valid_image_ids]
    
    # 6. 최종 데이터 구성
    cleaned_data = {
        "images": cleaned_images,
        "annotations": cleaned_annotations,
        "categories": data['categories']
    }
    
    with open(cleaned_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 어노테이션 정리 완료! '{cleaned_json_path}' 파일로 저장했습니다.")
    print(f"최종 유효 어노테이션 수: {len(cleaned_annotations)}")
    print("-" * 40)

# 이미지 생성 알고리즘(빈 배경 이미지를 사등분하고, 각 구역에서 랜덤으로 배치)
def synthesize_images(cleaned_json_path, library_json_path, library_folder, backgrounds_folder, output_folder, target_count=200):
    """
    준비된 배경 이미지 위에, 각 클래스별 목표치에 도달하도록 4개의 귀퉁이 사분면 내에서 
    랜덤하게 알약을 배치하여 이미지를 합성합니다. (경계선 오류 해결)
    """
    print("--- 🎯 목표 지향적 이미지 합성 시작 (v3.2: 경계선 오류 해결) ---")
    pills_per_image = 4

    # --- 1. 클래스 분포 분석 및 필요 개수 계산 ---
    with open(cleaned_json_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)
        
    annotations_df = pd.DataFrame(cleaned_data['annotations'])
    current_counts = annotations_df['category_id'].value_counts()
    cat_id_to_name = {cat['id']: cat['name'] for cat in cleaned_data['categories']}
    
    needed_counts = {}
    print("\n[Step 1] 클래스별 필요 개수 계산:")
    for cat_id, cat_name in cat_id_to_name.items():
        needed = max(0, target_count - current_counts.get(cat_id, 0))
        if needed > 0:
            print(f"- {cat_name}: {current_counts.get(cat_id, 0)}개 -> {needed}개 추가 필요")
            needed_counts[cat_id] = needed
            
    if not needed_counts:
        print("\n모든 클래스가 이미 목표 개수를 충족합니다. 합성을 종료합니다.")
        return

    class_pool, class_weights = list(needed_counts.keys()), list(needed_counts.values())
    num_images_to_create = math.ceil(sum(class_weights) / pills_per_image)
    print(f"\n총 {sum(class_weights)}개의 알약 추가를 위해 약 {num_images_to_create}개의 이미지 생성이 필요합니다.")

    # --- 2. 알약 라이브러리 로드 ---
    with open(library_json_path, 'r', encoding='utf-8') as f:
        library_data = json.load(f)
        
    pills_by_category = {}
    for ann in library_data['annotations']:
        cat_id = ann['category_id']
        pill_filename = f"pill_{ann['id']}.png"
        pill_path = os.path.join(library_folder, cat_id_to_name[cat_id], pill_filename)
        if os.path.exists(pill_path):
            pills_by_category.setdefault(cat_id, []).append(pill_path)

    # --- 3. 이미지 합성 준비 ---
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    background_files = [os.path.join(backgrounds_folder, f) for f in os.listdir(backgrounds_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not background_files:
        print(f"\n오류: '{backgrounds_folder}' 폴더에서 배경 이미지를 찾을 수 없습니다.")
        return

    synthetic_coco = {"images": [], "annotations": [], "categories": library_data['categories']}
    annotation_id_counter = 1
    
    # --- 4. 이미지 합성 루프 ---
    print("\n[Step 2] 이미지 합성 시작...")
    
    for image_id in tqdm(range(num_images_to_create), desc="이미지 합성 중"):
        bg_path = random.choice(background_files)
        # 한글 경로 처리 및 배경 로드 실패 시 방어 코드
        try:
            stream = open(bg_path, "rb")
            bytes_data = bytearray(stream.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            background = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            stream.close()
            if background is None:
                print(f"\n[경고] 배경 파일을 읽을 수 없습니다. 건너뜁니다: {bg_path}")
                continue
        except Exception as e:
            print(f"\n[경고] 배경 파일 처리 중 오류 발생. 건너뜁니다: {bg_path}, 오류: {e}")
            continue

        bg_h, bg_w, _ = background.shape
        
        selected_category_ids = random.choices(class_pool, weights=class_weights, k=pills_per_image)
        while len(set(selected_category_ids)) < pills_per_image:
             selected_category_ids = random.choices(class_pool, weights=class_weights, k=pills_per_image)

        quadrants = [
            (0, 0, bg_w//2, bg_h//2), (bg_w//2, 0, bg_w, bg_h//2),
            (0, bg_h//2, bg_w//2, bg_h), (bg_w//2, bg_h//2, bg_w, bg_h)
        ]
        
        for i, category_id in enumerate(selected_category_ids):
            q_x1, q_y1, q_x2, q_y2 = quadrants[i]
            
            pill_path = random.choice(pills_by_category[category_id])
            
            try:
                stream = open(pill_path, "rb")
                bytes_data = bytearray(stream.read())
                numpyarray = np.asarray(bytes_data, dtype=np.uint8)
                pill = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                stream.close()
                if pill is None:
                    print(f"\n[경고] 알약 파일을 읽을 수 없습니다. 건너뜁니다: {pill_path}")
                    continue
            except Exception as e:
                print(f"\n[경고] 알약 파일 처리 중 오류 발생. 건너뜁니다: {pill_path}, 오류: {e}")
                continue

            mask = pill[:,:,3] if pill.shape[2] == 4 else cv2.threshold(cv2.cvtColor(pill, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY_INV)[1]
            pill, (ph, pw) = pill[:,:,:3], pill.shape[:2]

            # ✅ [수정된 부분] 안전한 위치 계산 로직
            margin = 10
            
            # 알약 중심의 유효한 X, Y 좌표 범위 계산
            min_cx = q_x1 + pw//2 + margin
            max_cx = q_x2 - pw//2 - margin
            min_cy = q_y1 + ph//2 + margin
            max_cy = q_y2 - ph//2 - margin
            
            # 유효한 범위가 없을 경우 (알약이 사분면보다 클 때)
            if min_cx >= max_cx or min_cy >= max_cy:
                # 사분면의 중앙을 기준으로 하되, 배경 이미지 전체 경계를 벗어나지 않도록 강제 조정
                center_x = max(pw//2, min((q_x1 + q_x2)//2, bg_w - pw//2))
                center_y = max(ph//2, min((q_y1 + q_y2)//2, bg_h - ph//2))
            else:
                center_x = random.randint(min_cx, max_cx)
                center_y = random.randint(min_cy, max_cy)

            background = cv2.seamlessClone(pill, background, mask, (center_x, center_y), cv2.NORMAL_CLONE)
            new_bbox = [center_x - pw//2, center_y - ph//2, pw, ph]
            
            synthetic_coco['annotations'].append({
                "id": annotation_id_counter, "image_id": image_id, "category_id": category_id,
                "bbox": new_bbox, "area": pw * ph, "iscrowd": 0
            })
            annotation_id_counter += 1

        output_filename = f"synthetic_final_{image_id:06d}.png"
        output_filepath = os.path.join(output_folder, 'images', output_filename)
        
        extension = os.path.splitext(output_filepath)[1]
        result, encoded_img = cv2.imencode(extension, background)
        if result:
            with open(output_filepath, mode='w+b') as f:
                encoded_img.tofile(f)
                
        synthetic_coco['images'].append({"id": image_id, "width": bg_w, "height": bg_h, "file_name": output_filename})

    # --- 5. 최종 결과 저장 ---
    output_json_path = os.path.join(output_folder, 'synthetic_annotations_final.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(synthetic_coco, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 이미지 합성 완료! '{output_folder}' 폴더에 {len(synthetic_coco['images'])}개의 이미지와 어노테이션 파일을 생성했습니다.")

# 최종 패키징 함수
def package_final_dataset(clean_data, synthetic_json_path, source_original_images_folder, source_synthetic_images_folder, output_folder):
    """
    깨끗한 원본 데이터와 합성 데이터를 최종적으로 통합하여 패키징합니다.
    """
    final_images_dir = os.path.join(output_folder, 'images')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(final_images_dir)
    
    # 1. 이미지 파일 복사
    clean_filenames = [img['file_name'] for img in clean_data['images']]
    for filename in tqdm(clean_filenames, desc="원본 이미지 복사"):
        shutil.copy(os.path.join(source_original_images_folder, filename), final_images_dir)
    
    synth_filenames = os.listdir(source_synthetic_images_folder)
    for filename in tqdm(synth_filenames, desc="합성 이미지 복사"):
        shutil.copy(os.path.join(source_synthetic_images_folder, filename), final_images_dir)

    # 2. 어노테이션 병합
    with open(synthetic_json_path, 'r', encoding='utf-8') as f:
        synth_data = json.load(f)

    final_coco = {"images": [], "annotations": [], "categories": clean_data['categories']}
    final_coco['images'].extend(clean_data['images'])
    final_coco['annotations'].extend(clean_data['annotations'])
    
    # ID 재설정
    image_id_offset = max(img['id'] for img in clean_data['images']) + 1 if clean_data['images'] else 0
    ann_id_offset = max(ann['id'] for ann in clean_data['annotations']) + 1 if clean_data['annotations'] else 0
    
    synth_image_id_map = {img['id']: img['id'] + image_id_offset for img in synth_data['images']}
    
    for synth_img in synth_data['images']:
        synth_img['id'] = synth_image_id_map[synth_img['id']]
        final_coco['images'].append(synth_img)

    for synth_ann in synth_data['annotations']:
        synth_ann['id'] += ann_id_offset
        synth_ann['image_id'] = synth_image_id_map[synth_ann['image_id']]
        final_coco['annotations'].append(synth_ann)
    
    # 최종 파일 저장
    output_json_path = os.path.join(output_folder, 'final_annotations.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_coco, f, ensure_ascii=False, indent=4)