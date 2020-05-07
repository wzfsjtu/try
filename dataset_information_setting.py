#   这个文件用于生成数据文件txt，包括序号，正面图像地址，背面图像地址，重量，所属数据集
#   txt文件格式
#   line0:      'pos_folder_path:'positive_folder_path
#   line1:      'neg_folder_path:'negative_folder_path
#   line2:      'weight_path:'weight_excel_path
#   line3:      'txt_path:'txt_path
#   line4:      'pos,neg,weight,set_type'
#   line5:      positive_order0,negative_order0,weight0,set_type0
#   line6:      positive_order1,negative_order1,weight1,set_type1
#   ...         ...
#   导入相关包
import xlrd
import json
import numpy as np
import skimage.io
import torch
import torchvision
import matplotlib.pyplot as plt


def start(positive_folder_path, negative_folder_path, weight_excel_path, txt_path, black_list=[]):
    #   输入：
    #   positive_folder_path:  正面图像文件夹路径，文件夹包括JPEGImages和annotation.json
    #   negative_folder_path:  反面图像文件夹路径，文件夹包括JPEGImages和annotation.json
    #   weight_excel_path:     重量excel文件路径，包括正常粒重量矩阵
    #   txt_path:              txt文件路径,建议命名为info.txt
    #   black_list:            黑名单，type=List（tuple（row，column）），row和column与重量excel对应，翻转在start中实现
    #   WARNING!!!!!!!!!!!     正、反面图像矩阵和重量矩阵没有一一对应，需要进行调整

    #   输出：
    #   boolean:              txt创建完成标志

    #   weight import start

    EXCEPTION_WEIGHT = 100

    print('weight import start.')
    data = xlrd.open_workbook(weight_excel_path)
    c_sheet = data.sheet_by_name('完善粒(mg)')
    N = c_sheet.ncols * c_sheet.nrows
    c_data = np.zeros([c_sheet.nrows, c_sheet.ncols])
    # 同task1，对重量矩阵沿Z轴旋转180度
    for i in range(0, c_sheet.nrows):
        A = c_sheet.row_values(c_sheet.nrows - 1 - i)
        c_data[i] = A
        c_data[i] = c_data[i][::-1]
    weight_data = c_data.flatten()
    print('weight import done.')
    #   weight import done

    rows = c_sheet.nrows    # 行
    cols = c_sheet.ncols    # 列

    #   image import start
    print('image import start.')
    #   正面0~27 《=》 背面27~0，正面28~55 《=》 背面55~28
    #   用positive_negative_list来进行一一对应，positive_negative_list[positive_index]= negative_index
    positive_negative_list = [i for i in range(N)]
    for i in range(len(positive_negative_list)):
        #   i + positive_negative_list[i]  + 1 = row * cols
        positive_negative_list[i] = cols - 1 - i + 2 * cols * (i // 28)
    print('image import done.')
    #   image import done

    #   area import start
    print('area import start.')
    pos_val = json.load(open(positive_folder_path + r'/annotations.json', 'r'))
    image_dict = {int(image['file_name'][11:][:-4]): image['id'] for image in pos_val['images']}
    area_dict = {annotation['image_id']: annotation['area'] for annotation in pos_val['annotations']}
    areas = {i: area_dict[image_dict[i]] for i in image_dict.keys()}
    pos_area_list = [areas[i] for i in sorted(areas)]

    neg_val = json.load(open(negative_folder_path + r'/annotations.json', 'r'))
    image_dict = {int(image['file_name'][11:][:-4]): image['id'] for image in neg_val['images']}
    area_dict = {annotation['image_id']: annotation['area'] for annotation in neg_val['annotations']}
    areas = {i: area_dict[image_dict[i]] for i in image_dict.keys()}
    neg_area_list = [areas[i] for i in sorted(areas)]
    print('area import done.')
    #   area import done

    #   set_list generate start         生成训练集（A）,验证集（V）和测试集（E）标记    A:V:E =  6:2:2（数据量小时）
    print('set_list generate start.')
    set_list = []
    for i in range(N):
        if int(i) < int(N * 0.6):
            set_list.append('A')
        elif int(N * 0.6) <= i < int(N * (0.6 + 0.2)):
            set_list.append('V')
        else:
            set_list.append('E')
    print('set_list generate done.')
    #   set_list generate done.

    #   black_list start
    for pair in black_list:
        ind = (14 - pair[0]) * cols + cols - pair[1]
        # set black_weight -100
        weight_data[ind] = -EXCEPTION_WEIGHT
        set_list[ind] = 'N'
    #   black_list done

    #   calculate weight_data.max and weight_data.min start
    weight_data_copy = list(set(weight_data))
    weight_data_copy.sort()
    weight_max = weight_data_copy[1]
    weight_min = weight_data_copy[-1]
    print(weight_max, weight_min)
    #   calculate weight_data.max and weight_data.min done

    #   txt generation start
    print('txt generation start.')
    with open(txt_path, 'w') as f:
        f.write('pos_folder_path: {}\n'.format(positive_folder_path))
        f.write('neg_folder_path: {}\n'.format(negative_folder_path))
        f.write('weight_path: {}\n'.format(weight_excel_path))
        f.write('txt_path: {}\n'.format(txt_path))
        f.write('weight_max: {},weight_min: {}\n'.format(weight_max, weight_min))
        f.write('pos,neg,weight,posarea,negarea,set\n')
        for i in range(N):
            # img1 = skimage.io.imread(positive_folder_path + r'\JPEGImages' + '\{}.jpg'.format(i))
            # img2 = skimage.io.imread(negative_folder_path + r'\JPEGImages' + '\{}.jpg'.format(positive_negative_list[i]))
            # plt.figure(1)
            # plt.subplot(211)
            # plt.imshow(img1)
            # plt.subplot(212)
            # plt.imshow(img2)
            # plt.show()
            f.write('{},{},{},{},{},{}\n'.format(
                i,
                positive_negative_list[i],
                weight_data[i],
                pos_area_list[i],
                neg_area_list[positive_negative_list[i]],
                set_list[i],
            ))
    f.close()
    print('txt generation done.')
    return True


if __name__ == '__main__':
    start(r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\outputc',
          r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\outputc1',
          r'E:\科研\研究生\小麦\样本数据\2020.1.15\工作簿1.xlsx',
          r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\info1.txt',
          black_list=[(1, 25), (2, 22), (7, 23), (9, 23), (13, 15)])
