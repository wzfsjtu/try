import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import labelme
import pycocotools.coco
import pycocotools.mask
import labelme2coco
import labelme
from skimage import io
import sklearn
import sklearn.preprocessing
import torch
import torch.utils.data as Data
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


def weight_matrix_rotation(weight_matrix):
    weight_matrix_copy = weight_matrix.reshape(weight_matrix.size)
    weight_matrix_copy = weight_matrix_copy[::-1]
    weight_matrix_copy = weight_matrix_copy.reshape(weight_matrix.shape)
    return weight_matrix_copy.copy()


def weight_import():
    file = u'E:/科研/研究生/小麦/样本数据/2020.1.15/工作簿1.xlsx'
    data = xlrd.open_workbook(file)
    c_sheet = data.sheet_by_name('完善粒(mg)')
    p_sheet = data.sheet_by_name('破碎粒(mg)')
    print("C_sheet.nrows: {}".format(c_sheet.nrows))
    print("P_sheet.nrows: {}".format(p_sheet.nrows))
    ## 默认14*28
    N = 14*28
    c_data = np.zeros([14, 28])
    p_data = np.zeros([14, 28])
    for i in range(0, 14):
        A = c_sheet.row_values(i)
        c_data[i] = A
        A = p_sheet.row_values(i)
        p_data[i] = A
    x = np.arange(0, 14 * 28, 1)
    print('weight import successfully')
    # 原来重量和图像不匹配，需要将重量矩阵沿Z轴旋转180度

    c_data = weight_matrix_rotation(c_data)
    p_data = weight_matrix_rotation(p_data)
    c_data = np.reshape(c_data, [-1, 1])
    p_data = np.reshape(p_data, [-1, 1])
    return c_data, p_data


def area_import(annotation_json_path):
    # 输入：annotation_json_path
    # 输出：area_array
    json_file = annotation_json_path
    val = json.load(open(json_file, 'r'))
    # 取出图像序号和面积
    image_dict = {int(image['file_name'][11:][:-4]): image['id'] for image in val['images']}
    area_dict = {annotation['image_id']: annotation['area'] for annotation in val['annotations']}
    areas = {i: area_dict[image_dict[i]] for i in image_dict.keys()}
    return np.array([areas[i] for i in sorted(areas)])


def image_import(image_folder_path, num_of_image):
    # 注意，skimage.io.imread 似乎不能用with，因为不适用上下文管理器
    # intrans = torchvision.transforms.ToPILImage() PILIMAGE -> TENSOR
    # trans = torchvision.transforms.ToTensor() TENSOR -> PILIMAGE
    # 先读入一张图片，得到图片的CHW信息
    images = []
    for i in range(num_of_image):
        fp = image_folder_path + '\{}.jpg'.format(i)
        try:
            image = io.imread(fp)  # HWC,RGB
            image_trans = torchvision.transforms.ToTensor()
            image = image_trans(image)  # Tensor, CHW, RGB
            images.append(image)
        except:
            print("image {} read failed".format(i))
        else:
            # plt.imshow(image)
            # plt.show()
            # print("image {} read successfully".format(i))
            # print("type: {}".format(type(image)))
            # print("size: {}".format(np.shape(image)))
            pass
    print("image import done")
    return images


class MyDataSet(Data.Dataset):
    def __init__(self, info_path, set_type='A'):
        super(MyDataSet, self).__init__()
        self.info_path = info_path
        self.set_type = set_type
        self.pos_image_list = []
        self.neg_image_list = []
        self.weight_list = []
        self.pos_area_list = []
        self.neg_area_list = []
        with open(info_path, 'r') as f:
            self.pos_folder_path = f.readline().strip().split(' ')[1]
            self.neg_folder_path = f.readline().strip().split(' ')[1]
            self.weight_path = f.readline().strip().split(' ')[1]
            self.txt_path = f.readline().strip().split(' ')[1]
            W = f.readline().strip().split(',')
            self.weight_max = float(W[0].split(' ')[1])
            self.weight_min = float(W[1].split(' ')[1])
            self.titles = f.readline()
            for line in f:
                l = line.rstrip().split(',')
                if l[-1] == self.set_type:
                    self.pos_image_list.append(int(l[0]))
                    self.neg_image_list.append(int(l[1]))
                    self.weight_list.append(float(l[2]))
                    self.pos_area_list.append(float(l[3]))
                    self.neg_area_list.append(float(l[4]))
        f.close()
        self.dataset_size = len(self.weight_list)
        self.weight_list = torch.tensor(self.weight_list, dtype=torch.float32).unsqueeze(dim=1)
        # print(self.pos_folder_path)
        # print(self.neg_folder_path)
        # print(self.weight_path)
        # print(self.txt_path)
        # print(self.info_path)
        # print(self.set_type)
        # print(self.pos_list)
        # print(self.neg_list)
        # print(self.weight_list)

        if self.txt_path == self.info_path:
            print('txt_path and info_path is the same.')
        else:
            print('txt_path and info_path is DIFFERENT!')

    def __getitem__(self, item):
        # 导入正面、背面图像pos_img和neg_img
        pos_img = io.imread(self.pos_folder_path + r'\JPEGImages\{}.jpg'.format(self.pos_image_list[item]))
        neg_img = io.imread(self.neg_folder_path + r'\JPEGImages\{}.jpg'.format(self.neg_image_list[item]))
        # 注意！！转换为float32
        # plt.figure(1)
        # plt.subplot(121)
        # plt.imshow(pos_img)
        # plt.subplot(122)
        # plt.imshow(neg_img)
        # plt.show()
        # print(pos_img)
        # print(neg_img)
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),        # 对PIL图像（HWC）resize，默认双线性插值
            # augmentation
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomVerticalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.1),
            torchvision.transforms.ToTensor(),

            #
        ])
        pos_img = trans(pos_img)
        neg_img = trans(neg_img)
        # tensor2img_trans = torchvision.transforms.ToPILImage()
        # pos_img = tensor2img_trans(pos_img)
        # neg_img = tensor2img_trans(neg_img)
        # plt.figure(1)
        # plt.imshow(pos_img)
        # plt.show()

        # 导入重量
        weight = self.weight_list[item]
        #   数据归一化 (x - xmin) / (xmax - xmin)
        weight = (weight - self.weight_min) / (self.weight_max - self.weight_min)

        return pos_img, neg_img, weight

    def __len__(self):
        return self.dataset_size


class CNN(nn.Module):

    def __init__(self, n_classes):
        super(CNN, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3,
        #               out_channels=32,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.Conv2d(in_channels=32,
        #               out_channels=32,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.5),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.Conv2d(in_channels=64,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.Conv2d(in_channels=64,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.5),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.linear1 = nn.Linear(64 * 56 * 56, 512)
        # self.linear2 = nn.Linear(512, 128)
        # self.output = nn.Linear(128, n_classes)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear1 = nn.Linear(64 * 56 * 56, 256)
        self.linear2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, n_classes)

    def forward(self, x: 'torch.Tensor'):
        # print(torch.isnan(x))
        x = self.conv1(x)
        # print(torch.isnan(x))
        x = self.conv2(x)
        # print(torch.isnan(x))
        x = x.view(x.size(0), -1)
        # print(torch.isnan(x))
        x = self.linear1(x)
        # print(torch.isnan(x))
        x = self.linear2(x)
        # print(torch.isnan(x))
        output = self.output(x)
        # print(torch.isnan(output))
        return output


def task1():
    c_annotation_json_path = r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\outputc\annotations.json'
    c1_annotation_json_path = r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\outputc1\annotations.json'
    # P_annotation_json_path = r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\outputp\annotations.json'
    # P1_annotation_json_path = r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\outputp1\annotations.json'
    c_weight, p_weight = weight_import()  # 完善、破碎粒重量，N*1
    c_area = area_import(c_annotation_json_path)  # 完善粒正面面积，N*1
    c1_area = area_import(c1_annotation_json_path)  # 完善粒背面面积，N*1
    # p_area = area_import(P_annotation_json_path)                  # 破碎粒正面面积，N*1
    # p1_area = area_import(P1_annotation_json_path)                  # 破碎粒背面面积，N*1

    # ============================= #
    # 原来标注时发生错误，需要对背面面积的序号进行重新对应，0~27 -> 27~0， 28~55 -> 55~28，...
    c1_area = np.reshape(c1_area, [14, 28])
    for i in range(0, 14):
        c1_area[i] = c1_area[i][::-1]
    c1_area = np.reshape(c1_area, [-1, ])

    c_area = np.reshape(c_area, [-1, 1])
    c1_area = np.reshape(c1_area, [-1, 1])
    c_weight = np.reshape(c_weight, [-1, 1])
    c_weight_copy = c_weight

    # 将输入的数据分为训练集和测试集，比例为7:3,并将面积进行归一化,Tensor为dtype=float32
    M = 275  # 训练集个数
    area_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    c_area_norm = torch.tensor(area_scaler.fit_transform(c_area), dtype=torch.float32)

    # c_weight_copy 重量归一化
    weight_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    c_weight_norm = torch.tensor(weight_scaler.fit_transform(c_weight_copy), dtype=torch.float32)
    areas_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    c_areas_norm = torch.tensor(areas_scaler.fit_transform((c_area + c1_area) / 2), dtype=torch.float32)
    plt.figure(1)
    plt.scatter(c_area, c1_area)
    plt.xlabel('positive_area')
    plt.ylabel('negative_area')
    plt.figure(2)
    plt.scatter(c_areas_norm, c_weight_norm)
    plt.xlabel('normalized areas')
    plt.ylabel('normalized weight')
    plt.show()


def task2():
    EPOCH = 100
    LR = 0.0001
    BATCH_SIZE = 20
    train_data = MyDataSet(info_path=r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\info1.txt', set_type='A')
    validate_data = MyDataSet(info_path=r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\info1.txt', set_type='V')
    test_data = MyDataSet(info_path=r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\info1.txt', set_type='E')
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    validate_loader = Data.DataLoader(dataset=validate_data, batch_size=1, shuffle=False)

    net = CNN(1)
    # net = VGG16net(1)
    print(net)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    if use_gpu:
        net.cuda()
        loss_func.cuda()

    train_epoch_loss = []
    validate_epoch_loss = []
    test_grounds = []
    test_preds = []

    try:
        for epoch in range(EPOCH):
            for step, (pos_img, neg_img, train_weight) in enumerate(train_loader):
                if use_gpu:
                    pos_img = pos_img.cuda()
                    train_weight = train_weight.cuda()
                train_output = net(pos_img)
                loss = loss_func(train_output, train_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 5 == 0:
                print("Epoch {}".format(epoch))
                torch.save(torch.nn.Module.state_dict(net), '.\epoch_{}.pth'.format(epoch))
                net.eval()
                with torch.no_grad():
                    validate_loss = 0
                    validate_count = 0
                    for step, (validate_pos_img, validate_neg_img, validate_weight) in enumerate(validate_loader):
                        torch.cuda.empty_cache()
                        if use_gpu:
                            validate_pos_img = validate_pos_img.cuda()
                            validate_weight = validate_weight.cuda()
                        validate_output = net(validate_pos_img)
                        loss = loss_func(validate_output, validate_weight)
                        validate_loss += loss.clone().detach().cpu().data * validate_output.size(0)
                        validate_count += validate_output.size(0)
                    print("Epoch: {}, validate_loss: {}".format(epoch, validate_loss / validate_count))
                    validate_epoch_loss.append(validate_loss / validate_count)
                net.train()

        net.eval()
        with torch.no_grad():
            print("ground|test\n")
            test_loss = 0
            test_count = 0
            for step, (test_pos_img, test_neg_img, test_weight) in enumerate(test_loader):
                torch.cuda.empty_cache()
                if use_gpu:
                    test_pos_img = test_pos_img.cuda()
                    test_weight = test_weight.cuda()
                test_output = net(test_pos_img)
                loss = loss_func(test_output, test_weight)
                print(test_output.clone().detach().cpu().data, "|", test_weight.clone().detach().cpu().data, "\n")
                test_grounds.append(float(test_weight.clone().detach().cpu().data[0][0]))
                test_preds.append(float(test_output.clone().detach().cpu().data[0][0]))
                test_loss += loss.clone().detach().cpu().data * test_output.size(0)
                test_count += test_output.size(0)

            print('loss: {}'.format(test_loss / test_count))


        # plt.figure('loss')
        # plt.plot(range(len(train_epoch_loss)), train_epoch_loss, color='blue')
        # plt.plot(range(len(validate_epoch_loss)), validate_epoch_loss, color='red')
        # plt.legend(['train_loss', 'validate_loss'])
        # plt.axis([0, EPOCH, 0, 2])
        # # plt.show()
        # plt.savefig(r'E:\科研\研究生\小麦\loss1.png')
        # print(test_grounds)
        # print(test_preds)
        # plt.figure('pred')
        # plt.scatter(test_grounds, test_preds)
        # plt.grid()
        # plt.plot([0, 1], [0, 1])
        # # plt.show()
        # plt.savefig(r'E:\科研\研究生\小麦\pred1.png')
        # Err = [abs(test_preds[i] - test_grounds[i]) / test_grounds[i] for i in range(len(test_grounds))]
        # print("Error:", sum(Err) / len(Err))

        net.eval()
        train_loader = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i in range(EPOCH // 5):
                print('Epoch {}'.format(i * 5 + 4))
                torch.nn.Module.load_state_dict(net, torch.load(r'.\epoch_{}.pth'.format(i * 5 + 4)))
                # train
                train_loss = 0
                train_count = 0
                train_err = []
                plt.figure('train_{}'.format(i * 5 + 4))
                plt.plot([15, 65], [15, 65])
                plt.grid()
                plt.axis([0, 70, 0, 70])
                plt.xlabel('ground')
                plt.ylabel('pred')
                for step, (train_pos_img, train_neg_img, train_weight) in enumerate(train_loader):
                    if use_gpu:
                        train_pos_img = train_pos_img.cuda()
                        train_weight = train_weight.cuda()
                    train_output = net(train_pos_img)
                    loss = loss_func(train_output, train_weight)
                    train_count += 1
                    train_loss += loss.clone().detach().cpu().data
                    print('train {}, loss:'.format(step), loss.clone().detach().cpu().data)
                    pred = train_output.clone().detach().cpu().requires_grad_(False).tolist()[0][0]
                    ground = train_weight.clone().detach().cpu().requires_grad_(False).tolist()[0][0]

                    pred = pred * (63 - 19) + 19
                    ground = ground * (63 - 19) + 19
                    err = abs(pred - ground) / ground
                    print('pred: {}, ground: {}, err: {}'.format(pred, ground, err))
                    train_err.append(err)
                    plt.scatter(ground, pred, color='green')
                print("loss: {}".format(train_loss / train_count / 2))
                print(train_err)
                print("mean err:", sum(train_err) / len(train_err))

                # plt.show()
                plt.savefig(r'E:\科研\研究生\小麦\Result\train_{}.png'.format(i * 5 + 4))

                # validate
                validate_loss = 0
                validate_count = 0
                validate_err = []
                plt.figure('validate_{}'.format(i * 5 + 4))
                plt.plot([15, 65], [15, 65])
                plt.grid()
                plt.axis([0, 70, 0, 70])
                plt.xlabel('ground')
                plt.ylabel('pred')
                for step, (validate_pos_img, validate_neg_img, validate_weight) in enumerate(validate_loader):
                    if use_gpu:
                        validate_pos_img = validate_pos_img.cuda()
                        validate_weight = validate_weight.cuda()
                    validate_output = net(validate_pos_img)
                    loss = loss_func(validate_output, validate_weight)
                    validate_count += 1
                    validate_loss += loss.clone().detach().cpu().data
                    print('validate {}, loss:'.format(step), loss.clone().detach().cpu().data)
                    pred = validate_output.clone().detach().cpu().requires_grad_(False).tolist()[0][0]
                    ground = validate_weight.clone().detach().cpu().requires_grad_(False).tolist()[0][0]

                    pred = pred * (63 - 19) + 19
                    ground = ground * (63 - 19) + 19
                    err = abs(pred - ground) / ground
                    print('pred: {}, ground: {}, err: {}'.format(pred, ground, err))
                    validate_err.append(err)
                    plt.scatter(ground, pred, color='blue')
                print("loss: {}".format(validate_loss / validate_count / 2))
                print(validate_err)
                print("mean err:", sum(validate_err) / len(validate_err))

                # plt.show()
                plt.savefig(r'E:\科研\研究生\小麦\Result\validate_{}.png'.format(i * 5 + 4))
                # test
                test_loss = 0
                test_count = 0
                test_err = []
                plt.figure('test_{}'.format(i * 5 + 4))
                plt.plot([15, 65], [15, 65])
                plt.grid()
                plt.axis([0, 70, 0, 70])
                plt.xlabel('ground')
                plt.ylabel('pred')
                for step, (test_pos_img, test_neg_img, test_weight) in enumerate(test_loader):
                    if use_gpu:
                        test_pos_img = test_pos_img.cuda()
                        test_weight = test_weight.cuda()
                    test_output = net(test_pos_img)
                    loss = loss_func(test_output, test_weight)
                    test_count += 1
                    test_loss += loss.clone().detach().cpu().data
                    print('test {}, loss:'.format(step), loss.clone().detach().cpu().data)
                    pred = test_output.clone().detach().cpu().requires_grad_(False).tolist()[0][0]
                    ground = test_weight.clone().detach().cpu().requires_grad_(False).tolist()[0][0]

                    pred = pred * (63 - 19) + 19
                    ground = ground * (63 - 19) + 19
                    err = abs(pred - ground) / ground
                    print('pred: {}, ground: {}, err: {}'.format(pred, ground, err))
                    test_err.append(err)
                    plt.scatter(ground, pred, color='red')
                print("loss: {}".format(test_loss / test_count / 2))
                print(test_err)
                print("mean err:", sum(test_err) / len(test_err))
                plt.savefig(r'E:\科研\研究生\小麦\Result\test_{}.png'.format(i * 5 + 4))


    except BaseException as exception:
        print('Exception: {}'.format(exception))


class VGG16net(nn.Module):
    def __init__(self, n_classes: "int"):
        super(VGG16net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
        )
        # self.linear1 = nn.Linear(28 * 28 * 512, 4096)
        # self.linear2 = nn.Linear(4096, 4096)
        # self.out = nn.Linear(4096, n_classes)
        self.out = nn.Linear(7 * 7 * 512, n_classes)
        # self.linear1 = nn.Linear(7 * 7 * 512, 2)
        # self.out = nn.Linear(2, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)
        # out = self.linear1(out)
        # out = self.linear2(out)
        # out = self.linear1(out)
        out = self.out(out)
        return out


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    # labelme2coco.labelme2coco()
    # task1()
    task2()
    # L = [0.47727271914482117, 0.20454545319080353, 0.25, 0.3181818127632141, 0.6136363744735718, 0.2954545319080353, 0.2954545319080353, 0.3181818127632141, 0.20454545319080353, 0.5909090638160706, 0.3181818127632141, 0.27272728085517883, 0.5, 0.4545454680919647, 0.5909090638160706, 0.47727271914482117, 0.3636363744735718, 0.6136363744735718, 0.3181818127632141, 0.5681818127632141, 0.6136363744735718, 0.7727272510528564, 0.25, 0.5454545617103577, 0.5227272510528564, 0.15909090638160706, 0.5, 0.22727273404598236, 0.6363636255264282, 0.5454545617103577, 0.40909090638160706, 0.3863636255264282, 0.4545454680919647, 0.25, 0.6136363744735718, 0.5, 0.5, 0.5227272510528564, 0.47727271914482117, 0.40909090638160706, 0.3636363744735718, 0.4318181872367859, 0.15909090638160706, 0.5227272510528564, 0.09090909361839294, 0.27272728085517883, 0.6363636255264282, 0.5227272510528564, 0.6136363744735718, 0.3863636255264282, 0.47727271914482117, 0.3636363744735718, 0.40909090638160706, 0.22727273404598236, 0.22727273404598236, 0.27272728085517883, 0.22727273404598236, 0.7045454382896423, 0.3863636255264282, 0.5681818127632141, 0.47727271914482117, 0.4545454680919647, 0.4545454680919647, 0.4318181872367859, 0.4545454680919647, 0.5909090638160706, 0.3636363744735718, 0.6818181872367859, 0.40909090638160706, 0.4545454680919647, 0.6136363744735718, 0.2954545319080353, 0.5681818127632141, 0.27272728085517883, 0.7045454382896423, 0.5, 0.4318181872367859]
    # L1 = [0.4643567204475403, 0.26885706186294556, 0.6171239614486694, 0.6708859205245972, 0.7056723833084106, 0.4405507445335388, 0.3268565237522125, 0.3990173935890198, 0.3032139539718628, 0.6218065619468689, 0.6284602880477905, 0.462285578250885, 0.6271972060203552, 0.7471458315849304, 0.5575829744338989, 0.5200176239013672, 0.561244010925293, 0.4122476577758789, 0.35733580589294434, 0.469486802816391, 0.5669162273406982, 0.7422673106193542, 0.4681692123413086, 0.43936559557914734, 0.43245142698287964, 0.2475883811712265, 0.35561585426330566, 0.3065759837627411, 0.7418251633644104, 0.6344946622848511, 0.5930187106132507, 0.5161446928977966, 0.7329273819923401, 0.6082422137260437, 0.7847831845283508, 0.5898382663726807, 0.5485522747039795, 0.4756850004196167, 0.7030618786811829, 0.6060481071472168, 0.5550587773323059, 0.5107734799385071, 0.38902539014816284, 0.7591414451599121, 0.2491343766450882, 0.24795778095722198, 0.6254265308380127, 0.5727943181991577, 0.4777734875679016, 0.6717519164085388, 0.5090293288230896, 0.17114344239234924, 0.48825347423553467, 0.32278305292129517, 0.4395012855529785, 0.4865431785583496, 0.39442625641822815, 0.729850709438324, 0.6406647562980652, 0.8138948082923889, 0.5029466152191162, 0.42038512229919434, 0.5987384915351868, 0.5384516716003418, 0.40832310914993286, 0.7349493503570557, 0.4136729836463928, 0.6986573934555054, 0.5785363912582397, 0.4634820818901062, 0.4778715670108795, 0.3328378200531006, 0.4463181495666504, 0.22212979197502136, 0.708525538444519, 0.47413408756256104, 0.5954564213752747]
    # plt.scatter(range(len(L)), L)
    # plt.scatter(range(len(L)), L1)
    # plt.axis([0, 100, 0, 2])
    # plt.show()
    # plt.figure(2)
    # Err = [(L1[i] - L[i]) / L[i] for i in range(len(L))]
    # print(sum(Err) / len(Err))



    # net = CNN(1)
    # net.load_state_dict(torch.load(r'.\epoch_4.pth'))
    # net.cuda()
    # loss_func = nn.MSELoss()
    # loss_func.cuda()
    # EPOCH = 10
    # LR = 0.0001
    # BATCH_SIZE = 20
    # train_data = MyDataSet(info_path=r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\info1.txt', set_type='A')
    # validate_data = MyDataSet(info_path=r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\info1.txt', set_type='V')
    # test_data = MyDataSet(info_path=r'E:\科研\研究生\小麦\样本数据\2020.1.15\用于称重\info1.txt', set_type='E')
    # train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    # validate_loader = Data.DataLoader(dataset=validate_data, batch_size=1, shuffle=False)
    # net.eval()
    # with torch.no_grad():
    #     for step, (validate_pos_img, validate_neg_img, validate_weight) in enumerate(validate_loader):
    #         if use_gpu:
    #             validate_pos_img = validate_pos_img.cuda()
    #             validate_weight = validate_weight.cuda()
    #         validate_output = net(validate_pos_img)
    #         loss = loss_func(validate_output, validate_weight)
    #         print('validate {}'.format(step), loss.detach().cpu().data)
