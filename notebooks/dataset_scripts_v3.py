import torch
import torchvision.transforms as T
from torchvision.ops import nms
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
import shutil
import xml.etree.ElementTree as ET

# model=get_model() 학습시킨 모델이 문제있던 850장의 데이터를 입력받고 예측값을 새로 라벨 데이터로 사용하므로 모델 입력 필수
# device = 'cuda'

# 학습시킨 모델로 라벨 데이터 만들기(세미오토 라벨링)
def predict_and_create_review_json(
    problematic_json_path, 
    source_images_folder, 
    model, 
    device, 
    output_json_path,
    score_threshold=0.2,  # 낮은 점수도 일단 탐지하도록 설정
    iou_threshold=0.3    # NMS를 위한 겹침 기준
):
    """
    [v3] 모델 예측 후 NMS를 적용하고, 'score'까지 함께 COCO JSON 파일로 저장합니다.
    """
    print("--- 🤖 Step 1: 모델 초벌 라벨링 (NMS + Score 저장) 시작 ---")
    model.eval()
    
    with open(problematic_json_path, 'r', encoding='utf-8') as f:
        problematic_data = json.load(f)
        
    try:
        cat_id_to_label_map = {cat['id']: i + 1 for i, cat in enumerate(problematic_data['categories'])}
        label_to_cat_id_map = {v: k for k, v in cat_id_to_label_map.items()}
        print("-> 원본 카테고리 ID <-> 모델 라벨 매핑 완료.")
    except Exception as e:
        print(f"오류: 카테고리 맵 생성 중 문제 발생: {e}")
        return

    review_coco = {
        "images": problematic_data['images'],
        "annotations": [],
        "categories": problematic_data['categories']
    }
    
    transform = T.ToTensor()
    annotation_id_counter = 1
    
    for img_info in tqdm(problematic_data['images'], desc="모델 예측 중"):
        img_path = os.path.join(source_images_folder, img_info['file_name'])
        if not os.path.exists(img_path): continue

        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).to(device).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        # 1. score_threshold를 넘는 박스만 먼저 거릅니다.
        keep_by_score = prediction['scores'] > score_threshold
        high_score_boxes = prediction['boxes'][keep_by_score]
        high_score_labels = prediction['labels'][keep_by_score]
        high_score_scores = prediction['scores'][keep_by_score]

        # 2. NMS를 적용하여 겹치는 박스를 제거합니다.
        keep_by_nms = nms(high_score_boxes, high_score_scores, iou_threshold)
        
        final_boxes = high_score_boxes[keep_by_nms]
        final_labels = high_score_labels[keep_by_nms]
        final_scores = high_score_scores[keep_by_nms] # 👈 NMS를 통과한 최종 점수들

        for box, label, score in zip(final_boxes, final_labels, final_scores):
            box_np = box.cpu().numpy()
            x, y, xmax, ymax = box_np
            w, h = xmax - x, ymax - y
            original_cat_id = label_to_cat_id_map.get(int(label.cpu()), -1)

            new_ann = {
                "id": annotation_id_counter,
                "image_id": img_info['id'],
                "category_id": original_cat_id,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "score": float(score.cpu().numpy()) # ✅ 점수를 저장합니다.
            }
            review_coco['annotations'].append(new_ann)
            annotation_id_counter += 1

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(review_coco, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 초벌 라벨링 (NMS + Score 저장) 완료! '{output_json_path}' 파일을 저장했습니다.")


# 직접 검수용 데이터 시각화(이미지 파일화)
def visualize_all_predictions_with_score(
    review_json_path, 
    source_images_folder, 
    output_folder
):
    """
    [v3] '검토용' 어노테이션 파일 전체를 시각화하며, 'score'도 함께 표시합니다.
    """
    print(f"--- 🖼️ '초벌 라벨링' 전체 시각화 (Score 표시) 시작 ---")
    
    if not os.path.exists(review_json_path):
        print(f"오류: 검토용 JSON 파일 '{review_json_path}'를 찾을 수 없습니다.")
        return

    os.makedirs(output_folder, exist_ok=True)

    with open(review_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        cat_id_to_name_map = {
            cat['id']: cat['name'] for cat in data['categories']
        }
        print("-> 원본 카테고리 ID와 이름 매핑 완료.")
    except Exception as e:
        print(f"오류: 카테고리 맵 생성 중 문제 발생: {e}")
        return

    annos_by_img_id = {}
    for ann in data['annotations']:
        annos_by_img_id.setdefault(ann['image_id'], []).append(ann)
        
    print(f"총 {len(data['images'])}개의 이미지 시각화를 시작합니다...")
    
    for img_info in tqdm(data['images'], desc="시각화 진행 중"):
        img_path = os.path.join(source_images_folder, img_info['file_name'])
        if not os.path.exists(img_path): continue
            
        stream = open(img_path, "rb")
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        stream.close()

        annotations = annos_by_img_id.get(img_info['id'], [])
        
        for ann in annotations:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cat_id = ann['category_id']
            cat_name = cat_id_to_name_map.get(cat_id, f"ID:{cat_id}")

            score = ann.get('score', -1.0) # 1단계에서 저장한 score를 가져옵니다.
            
            if score >= 0:
                label_text = f"{cat_name}: {score:.2f}" # 예: "보령부스파정: 0.34"
            else:
                label_text = cat_name # score가 없는 경우 대비
            
            # 점수에 따라 박스 색상 변경 (선택 사항)
            color = (0, 255, 0) # 기본값 (녹색)
            if score < 0.5:
                color = (0, 255, 255) # 낮은 점수는 노란색으로 표시
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 결과 이미지 저장
        output_path = os.path.join(output_folder, img_info['file_name'])
        extension = os.path.splitext(output_path)[1]
        result, encoded_img = cv2.imencode(extension, image)
        if result:
            with open(output_path, mode='w+b') as f:
                encoded_img.tofile(f)
                
    print(f"✅ 검수용 시각화 완료! '{output_folder}' 폴더에서 850개의 이미지를 확인하세요.")


# 새로 라벨링 한 xml파일과 깨끗한 원본 데이터(639장)의 어노테이션 병합

# xml 파일을 coco 데이터 양식으로 변환
def parse_xml_to_coco_v2(
    xml_path, 
    name_to_cat_id_map, 
    categories_list, 
    next_new_cat_id
):
    """
    단일 XML 파일을 읽고, 신규 카테고리를 발견하면 자동으로 추가하며 변환합니다.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    image_info = {
        "file_name": root.find('filename').text,
        "width": int(root.find('size/width').text),
        "height": int(root.find('size/height').text)
    }
    
    annotations = []
    for obj in root.findall('object'):
        label_name = obj.find('name').text
        
        # --- ✅ [핵심 수정] 신규 카테고리 감지 및 추가 로직 ---
        if label_name not in name_to_cat_id_map:
            print(f"\n[신규 카테고리 감지] '{label_name}'을(를) 목록에 추가합니다.")
            new_id = next_new_cat_id
            name_to_cat_id_map[label_name] = new_id # '번역 사전' 업데이트
            categories_list.append({ # '카테고리 목록' 업데이트
                "supercategory": "pill",
                "id": new_id,
                "name": label_name
            })
            next_new_cat_id += 1 # 다음 ID 준비
            
        category_id = name_to_cat_id_map[label_name]
        # --- [수정 끝] ---
            
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        
        annotations.append({
            "category_id": category_id,
            "bbox": bbox,
            "area": (xmax - xmin) * (ymax - ymin),
            "iscrowd": 0
        })
        
    # 변경된 'next_new_cat_id' 값을 반환하여 다음 XML 파싱에 사용
    return image_info, annotations, next_new_cat_id

# 원본 데이터와의 병합
def package_dataset_with_xml_v2(
    clean_json_path,
    xml_folder_path,
    source_images_folder,
    output_folder
):
    """
    COCO(JSON)와 PASCAL VOC(XML)를 병합하며,
    XML에만 존재하는 신규 카테고리를 자동으로 추가합니다.
    """
    print("--- 🚀 최종 데이터셋 (v_clean) 재구성 시작 (신규 카테고리 감지) ---")
    
    # --- Step 1: 폴더 초기화 ---
    final_images_dir = os.path.join(output_folder, 'images')
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(final_images_dir)
    print(f"\n[Phase 1] '{output_folder}' 폴더를 깨끗하게 준비했습니다.")

    # --- Step 2: 깨끗한 원본 데이터 로드 및 '번역 사전' 생성 ---
    with open(clean_json_path, 'r', encoding='utf-8') as f:
        clean_data = json.load(f)
    
    # [수정] 이 변수들이 parse_xml_to_coco_v2 함수에 의해 '직접' 수정될 것입니다.
    categories_list = clean_data['categories']
    name_to_cat_id_map = {cat['name']: cat['id'] for cat in categories_list}
    
    # [수정] 신규 ID 할당을 위한 기준점 설정
    try:
        max_cat_id = max(cat['id'] for cat in categories_list)
        next_new_cat_id = max_cat_id + 1
    except ValueError:
        next_new_cat_id = 1 # 카테고리가 아예 비어있을 경우
        
    print(f"-> 기존 {len(categories_list)}개 카테고리 로드. 신규 ID는 {next_new_cat_id}부터 시작합니다.")

    # 최종 COCO 데이터 구조 (깨끗한 639장을 기본으로)
    final_coco = {
        "images": clean_data['images'],
        "annotations": clean_data['annotations'],
        "categories": categories_list # 👈 수정될 수 있는 'categories_list'를 연결
    }
    
    image_id_offset = max(img['id'] for img in clean_data['images']) + 1 if clean_data['images'] else 1
    ann_id_offset = max(ann['id'] for ann in clean_data['annotations']) + 1 if clean_data['annotations'] else 1

    # --- Step 3: XML 변환 및 병합 (v2) ---
    print(f"\n[Phase 2] '{xml_folder_path}' 폴더에서 XML 파일을 변환 및 병합합니다...")
    xml_files = [f for f in os.listdir(xml_folder_path) if f.endswith('.xml')]
    newly_added_filenames = set()
    
    for xml_file in tqdm(xml_files, desc="XML 변환 중"):
        xml_path = os.path.join(xml_folder_path, xml_file)
        
        # [수정] v2 함수 호출
        new_image_info, new_annotations, next_new_cat_id = parse_xml_to_coco_v2(
            xml_path, 
            name_to_cat_id_map,  # '번역 사전' (수정 가능)
            categories_list,     # '카테고리 목록' (수정 가능)
            next_new_cat_id      # '다음 ID'
        )
        
        new_image_id = image_id_offset
        new_image_info['id'] = new_image_id
        final_coco['images'].append(new_image_info)
        newly_added_filenames.add(new_image_info['file_name'])
        
        for ann in new_annotations:
            ann['id'] = ann_id_offset
            ann['image_id'] = new_image_id
            final_coco['annotations'].append(ann)
            ann_id_offset += 1
            
        image_id_offset += 1
        
    print(f"-> {len(xml_files)}개의 XML 파일을 성공적으로 병합했습니다.")
    print(f"-> 최종 카테고리 수: {len(final_coco['categories'])}개")

    # --- Step 4: 이미지 파일 통합 ---
    clean_filenames = {img['file_name'] for img in clean_data['images']}
    all_filenames = clean_filenames.union(newly_added_filenames)
    
    print(f"\n[Phase 3] 총 {len(all_filenames)}개의 원본 이미지를 복사합니다...")
    for filename in tqdm(all_filenames, desc="전체 원본 이미지 복사"):
        source_path = os.path.join(source_images_folder, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, final_images_dir)
        else:
            print(f"경고: 원본 이미지 '{filename}'을 찾을 수 없습니다!")

    # --- Step 5: 최종 저장 ---
    output_json_path = os.path.join(output_folder, 'final_clean_annotations.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_coco, f, ensure_ascii=False, indent=4)
    
    print("\n--- ✅ 최종 (v_clean) 재구성 완료! ---")
    print(f"'{output_folder}'에 총 {len(final_coco['images'])}개의 이미지와 {len(final_coco['annotations'])}개의 어노테이션이 저장되었습니다.")


# 원본데이터 + 신규 라벨링 데이터 병합
def merge_final_datasets(
    base_clean_json_path,    # 👈 100% 신뢰하는 645장 데이터
    review_json_path,      # 👈 모델이 예측한 844장 데이터
    source_images_folder,  # 👈 1489장 원본 이미지 폴더
    output_folder
):
    """
    100% 신뢰하는 '베이스' 데이터셋과,
    베이스와 겹치지 않는 '모델 예측' 데이터셋을 병합하여
    최종 1489장 데이터셋을 패키징합니다.
    """
    print("--- 🚀 최종 데이터셋 (v_final) 재구성 시작 (Clean 645 + Predicted 844) ---")
    
    # --- Step 1: 폴더 초기화 ---
    final_images_dir = os.path.join(output_folder, 'images')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(final_images_dir)
    print(f"\n[Phase 1] '{output_folder}' 폴더를 깨끗하게 준비했습니다.")

    # --- Step 2: [소스 A] 100% 신뢰하는 베이스(645장) 로드 ---
    with open(base_clean_json_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    # 베이스에 포함된 파일명 목록 (이것을 기준으로 예측 데이터를 필터링)
    base_filenames = {img['file_name'] for img in base_data['images']}
    print(f"-> 베이스 데이터 {len(base_filenames)}장 로드 완료.")

    # 최종 COCO 데이터 구조 (베이스를 기본으로)
    final_coco = {
        "images": base_data['images'],
        "annotations": base_data['annotations'],
        "categories": base_data['categories']
    }
    
    # ID 재설정을 위한 기준점(offset) 계산
    image_id_offset = max(img['id'] for img in base_data['images']) + 1 if base_data['images'] else 1
    ann_id_offset = max(ann['id'] for ann in base_data['annotations']) + 1 if base_data['annotations'] else 1

    # --- Step 3: [소스 B] 모델 예측(844장) 필터링 및 병합 ---
    print(f"\n[Phase 2] '{review_json_path}'에서 모델 예측 데이터를 필터링 및 병합합니다...")
    with open(review_json_path, 'r', encoding='utf-8') as f:
        review_data = json.load(f)

    # [핵심] 베이스(645장)에 포함되지 않은 844개의 이미지만 필터링
    images_to_add = [
        img for img in review_data['images'] 
        if img['file_name'] not in base_filenames
    ]
    image_ids_to_add = {img['id'] for img in images_to_add}
    
    # 844개 이미지에 연결된 어노테이션만 필터링
    annotations_to_add = [
        ann for ann in review_data['annotations'] 
        if ann['image_id'] in image_ids_to_add
    ]
    print(f"-> {len(images_to_add)}개의 순수 예측 데이터(850 - {850 - len(images_to_add)})를 병합합니다.")

    # ID 재설정 및 최종 병합
    review_id_map = {img['id']: img['id'] + image_id_offset for img in images_to_add}
    
    for img in images_to_add:
        img['id'] = review_id_map[img['id']]
        final_coco['images'].append(img)

    for ann in annotations_to_add:
        ann['id'] += ann_id_offset
        ann['image_id'] = review_id_map[ann['image_id']]
        final_coco['annotations'].append(ann)
        ann_id_offset += 1
    
    image_id_offset += len(images_to_add)

    # --- Step 4: 이미지 파일 통합 ---
    all_filenames = {img['file_name'] for img in final_coco['images']}
    print(f"\n[Phase 3] 총 {len(all_filenames)}개의 원본 이미지를 복사합니다...")
    if len(all_filenames) != 1489:
        print(f"경고: 최종 이미지 수가 1489장이 아닙니다! (현재: {len(all_filenames)}장)")
        
    for filename in tqdm(all_filenames, desc="전체 원본 이미지 복사"):
        source_path = os.path.join(source_images_folder, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, final_images_dir)
        else:
            print(f"경고: 원본 이미지 '{filename}'을 찾을 수 없습니다!")

    # --- Step 5: 최종 저장 ---
    output_json_path = os.path.join(output_folder, 'final_1489_annotations.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_coco, f, ensure_ascii=False, indent=4)
    
    print("\n--- ✅ 최종 (v_final) 재구성 완료! ---")
    print(f"'{output_folder}'에 총 {len(final_coco['images'])}개의 이미지와 {len(final_coco['annotations'])}개의 어노테이션이 저장되었습니다.")
    print(f"최종 카테고리 수: {len(final_coco['categories'])}개")


# 데이터셋 무결성 검사
def get_expected_count_from_filename(filename):
    """
    파일명의 약품 코드 부분을 분석하여 기대되는 알약 개수를 반환합니다.
    예: 'K-001-002-003_...' -> 3
    """
    try:
        # 파일명에서 첫번째 '_' 앞부분을 추출 (예: 'K-001900-010224-016551')
        key_part = filename.split('_')[0]
        # '-'를 기준으로 분리하여 개수를 셈
        num_parts = len(key_part.split('-'))
        # 'K' 부분을 제외한 개수가 알약의 개수
        expected_count = num_parts - 1
        return expected_count
    except Exception:
        # 예상치 못한 형식의 파일명일 경우 0을 반환
        return 0

def verify_annotation_counts_by_filename(merged_json_path):
    """
    병합된 COCO 파일을 분석하여, 파일명 규칙에 따른 어노테이션 개수를 검증합니다.
    """
    if not os.path.exists(merged_json_path):
        print(f"오류: '{merged_json_path}' 파일을 찾을 수 없습니다.")
        print("스크립트와 같은 경로에 파일이 있는지 확인해주세요.")
        return

    try:
        with open(merged_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"오류: '{merged_json_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return


    images_df = pd.DataFrame(data.get('images', []))
    annotations_df = pd.DataFrame(data.get('annotations', []))

    if images_df.empty:
        print("오류: JSON 파일에 이미지 정보('images' key)가 없습니다.")
        return

    if annotations_df.empty:
        print("경고: JSON 파일에 어노테이션 정보('annotations' key)가 없습니다. 모든 실제값이 0으로 표시됩니다.")
        # 어노테이션이 없는 경우를 대비해 빈 DataFrame 생성
        actual_counts = pd.DataFrame(columns=['id', 'actual_count'])
    else:
        # 1. 각 이미지별 실제 어노테이션 개수 계산
        actual_counts = annotations_df['image_id'].value_counts().reset_index()
        actual_counts.columns = ['id', 'actual_count']

    # 2. 이미지 정보와 실제 어노테이션 개수 병합
    analysis_df = pd.merge(images_df, actual_counts, on='id', how='left')
    analysis_df['actual_count'] = analysis_df['actual_count'].fillna(0).astype(int)

    # 3. 파일명에서 기대 어노테이션 개수 추출
    analysis_df['expected_count'] = analysis_df['file_name'].apply(get_expected_count_from_filename)

    # 4. 기대값과 실제값이 다른 경우 필터링
    # (단, 파일명 규칙이 없는 이미지(expected_count=0)는 검증에서 제외)
    mismatched_df = analysis_df[
        (analysis_df['actual_count'] != analysis_df['expected_count']) &
        (analysis_df['expected_count'] > 0)
    ].copy() # SettingWithCopyWarning 방지를 위해 .copy() 사용

    # --- 결과 리포트 ---
    total_images_to_check = len(analysis_df[analysis_df['expected_count'] > 0])
    total_mismatched = len(mismatched_df)
    total_correct = total_images_to_check - total_mismatched

    print("--- 🔬 어노테이션 개수 무결성 검증 결과 ---")
    if total_images_to_check == 0:
        print("검증할 이미지를 찾지 못했습니다. 파일명 규칙을 다시 확인해주세요.")
        return

    print(f"총 {total_images_to_check}개 이미지를 검증했습니다.")
    print(f"✅ {total_correct}개 이미지는 파일명과 어노테이션 개수가 일치합니다.")

    if not mismatched_df.empty:
        print(f"\n🚨 {total_mismatched}개 이미지에서 개수 불일치가 발견되었습니다:")
        # 보기 편하게 열 이름 변경하여 출력
        mismatched_df.rename(columns={
            'file_name': '파일명',
            'expected_count': '기대 개수',
            'actual_count': '실제 개수'
        }, inplace=True)
        report = mismatched_df[['파일명', '기대 개수', '실제 개수']]
        print(report.to_string(index=False))
    else:
        print("\n🎉 완벽합니다! 모든 이미지의 어노테이션 개수가 파일명과 정확히 일치합니다.")
    print("-" * 50)



#----------실행 코드-----------

if __name__ == '__main__':
    # 사전 학습 모델로 라벨 데이터 예측
    predict_and_create_review_json(
    problematic_json_path='problematic_annotations.json', # 850장의문제 있는 데이터의 어노테이션 파일
    source_images_folder='data/train_images', # 원본 학습 이미지 폴더
    model=model, #사전 학습시킨 모델 로드
    device=device, #cuda or cpu
    output_json_path='annotations_for_review.json' #결과물을 json으로 저장
    )

    # 검수를 위한 시각화+이미지 파일화
    visualize_all_predictions_with_score(
    'annotations_for_review.json', # 세미오토 라벨링을 위한, 모델 예측 결과
    'data/train_images', # 원본 학습 이미지 폴더
    'review_visualizations_1' #신규 폴더
    )

    # 직접 라벨링한 데이터와 원본 깨끗한 데이터 병합(639장 + 6장)
    package_dataset_with_xml_v2(
    clean_json_path='cleaned_annotations.json', # 639장의 어노테이션 파일
    xml_folder_path='new_xml_labels', # 새로 직접 만든 라벨데이터 6장 폴더
    source_images_folder='data/train_images', #원본 학습이미지 폴더
    output_folder='final_dataset_v_clean'
    )

    # 깨끗한 데이터(639+6장)와 신규 라벨링 데이터(844장) 최종 병합
    merge_final_datasets(
    base_clean_json_path='final_dataset_v_clean/final_clean_annotations.json', # 👈 1순위 (645장)
    review_json_path='annotations_for_review.json',                       # 👈 2순위 (850장)
    source_images_folder='data/train_images',                             # 👈 1489장 원본 폴더
    output_folder='final_dataset_1489'                                    # 👈 최종 결과물 폴더
    )

    # 데이터 무결성 검사
    json_file_path = 'final_dataset_1489/final_1489_annotations.json' 
    verify_annotation_counts_by_filename(json_file_path)