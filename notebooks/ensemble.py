import os
import csv

origin_path = r"C:\Users\main\Downloads\origin.csv"
path = r"C:\Users\main\Downloads\re"
head = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']

def bbox_iou(box1, box2, value=0.8):
    # (x, y, w, h) → (x1, y1, x2, y2) 변환
    if len(box2) == 6:
        mean = box2[-1]
        box2 = [box2[0]/mean, box2[1]/mean, box2[2]/mean, box2[3]/mean]
    x1_1, y1_1, x2_1, y2_1 = box1[0]-box1[2]/2, box1[1]-box1[3]/2, box1[0]+box1[2]/2, box1[1]+box1[3]/2
    x1_2, y1_2, x2_2, y2_2 = box2[0]-box2[2]/2, box2[1]-box2[3]/2, box2[0]+box2[2]/2, box2[1]+box2[3]/2

    inter_x1, inter_y1, inter_x2, inter_y2 = max(x1_1, x1_2), max(y1_1, y1_2), min(x2_1, x2_2), min(y2_1, y2_2)

    inter_w, inter_h = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
    intersection = inter_w*inter_h

    # 각 영역 면적
    area1 = (x2_1-x1_1)*(y2_1-y1_1)
    area2 = (x2_2-x1_2)*(y2_2-y1_2)
    union = area1+area2-intersection

    # IoU 계산
    iou = intersection/union if union > 0 else 0.0
    return iou >= value


or_f = open(origin_path, "r")
or_data = list(csv.reader(or_f))[1:]
or_dict = {}
for i in or_data:
    i = list(map(float,i))
    image_id = i[1]
    if image_id in or_dict:
        or_dict[image_id].append(i[2:])
    else:
        or_dict[image_id] = [i[2:]]

for pat in os.listdir(path):
    
    count = 0
    count_1 = 0
    f = open(os.path.join(path, pat), "r")
    data = list(csv.reader(f))[1:]
    for i in data:
        i = list(map(float,i))
        image_id = i[1]
        class_id = i[2]
        bbox = i[3:7]
        if image_id in or_dict:
            for idx, anno_list in enumerate(or_dict[image_id]):
                or_class_id = anno_list[0]
                if bbox_iou(bbox, anno_list[1:5]):
                    count_1 += 1
                else:
                    count += 1
                if class_id == or_class_id and class_id in anno_list and bbox_iou(bbox, anno_list[1:5]):
                    # iou 비교 0.8이상 반영
                    anno_list[1] += i[3]
                    anno_list[2] += i[4]
                    anno_list[3] += i[5]
                    anno_list[4] += i[6]
                    if len(anno_list) == 6:
                        anno_list.append(2)
                    else:
                        anno_list[6] += 1
                    or_dict[image_id][idx] = anno_list
            # or_dict[image_id].append(i[2:7])
        else:
            or_dict[image_id] = [i[2:]]
            
    print(count_1)
    print(count)
rows = []

count = 1
for key in or_dict.keys():
    for value in or_dict[key]:
        print(value)
        meann = value[-1]
        conf = value[-2]
        cate_id = int(value[0])
        value = [int(value[1]/meann), int(value[2]/meann), int(value[3]/meann), int(value[4]/meann)]
        rows.append([count, int(key), cate_id, *value, conf])
        count += 1

OUTPUT_CSV = "ensemble_output_last.csv"   

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
    writer.writerows(rows)
    
    print('')