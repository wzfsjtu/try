import glob
import json
import pycocotools.mask
import pycocotools.coco
import argparse
import collections
import datetime
import os
import os.path as osp
import sys
import uuid

import numpy as np
import PIL.Image

import labelme

## ==================
import matplotlib.pyplot as plt

def labelme2coco():
    # https://github.com/wkentaro/labelme/blob/master/examples/instance_segmentation/labelme2coco.py
    # It generates:
    #   - data_dataset_coco/JPEGImages
    #   - data_dataset_coco/annotations.json
    # Terminal: labelme2coco.py data_annotated data_dataset_coco --labels labels.txt
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
    print('Creating dataset:', args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            data_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    out_ann_file = osp.join(args.output_dir, 'annotations.json')
    label_files = glob.glob(osp.join(args.input_dir, '*.json'))
    for image_id, label_file in enumerate(label_files):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]
        out_img_file = osp.join(
            args.output_dir, 'JPEGImages', base + '.jpg'
        )

        img_file = osp.join(
            osp.dirname(label_file), label_data['imagePath']
        )
        img = np.asarray(PIL.Image.open(img_file).convert('RGB'))
        PIL.Image.fromarray(img).save(out_img_file)
        data['images'].append(dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            id=image_id,
        ))

        masks = {}
        segmentations =collections.defaultdict(list)
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            shape_type = shape.get('shape_id')
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            points = np.asarray(points).flatten().tolist()
            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data['annotations'].append(dict(
                id=len(data['annotations']),
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[instance],
                area=area,
                bbox=bbox,
                iscrowd=0,
            ))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    #labelme2coco()
    print()

    # 下面这一段不知道有什么用
    '''
    fp = 'E:\\科研\\研究生\\小麦\\样本数据\\2020.1.15\\用于称重\\背面正常备份\\'
    for i in range(0, 14):
        for j in range(0, 28):
            src_path1 = fp + str(i * 28 + j) + '.jpg'
            drc_path1 = fp + '-' + str(i * 28 + 27 - j) + '.jpg'
            src_path2 = fp + str(i * 28 + j) + '.json'
            drc_path2 = fp + '-' + str(i * 28 + 27 - j) + '.json'
            os.rename(src_path1, drc_path1)
            os.rename(src_path2, drc_path2)

    for i in range(0, 14):
        for j in range(0, 28):
            drc_path1 = fp + '-' + str(i * 28 + 27 - j) + '.jpg'
            result_path1 = fp + str(i * 28 + 27 - j) + '.jpg'
            drc_path2 = fp + '-' + str(i * 28 + 27 - j) + '.json'
            result_path2 = fp + str(i * 28 + 27 - j) + '.json'
            os.rename(drc_path1, result_path1)
            os.rename(drc_path2, result_path2)
    '''