import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
from get_cls_map import get_cls_map
import warnings
warnings.filterwarnings("ignore")



import time
from SSFAN import SSFAN
import spectral
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))#  精度小降低OA  ,dtype=np.float16
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):


    X_test, X_train, y_test, y_train = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 100

def create_data_loader():
    # 地物类别
    # class_num = 16
    # 读入数据

    #Indine
    # X = sio.loadmat(r"E:\zc_paper\five\code_main\data\Indian\Indian_pines_corrected.mat")['indian_pines_corrected']  # X.shape=(145,145,200)
    # y = sio.loadmat(r"E:\zc_paper\five\code_main\data\Indian\Indian_pines_gt.mat")['indian_pines_gt']#[145,145]

    # ## PaviaU
    # X = sio.loadmat(r"E:\zc_paper\five\code_main\data\PaviaU\PaviaU.mat")[
    #     'paviaU']  # X.shape=(145,145,200)
    # y = sio.loadmat(r"E:\zc_paper\five\code_main\data\PaviaU\PaviaU_gt.mat")['paviaU_gt']  # [145,145]

    # #HanChuan
    # X = sio.loadmat(r"E:\zc_paper\five\code_main\data\HanChuan\WHU_Hi_HanChuan.mat")['WHU_Hi_HanChuan']  # X.shape=(145,145,200)
    # y = sio.loadmat(r"E:\zc_paper\five\code_main\data\HanChuan\WHU_Hi_HanChuan_gt.mat")['WHU_Hi_HanChuan_gt']  # [145,145]         #X = sio.loadmat('D:\wodewenjian\HSI_dada\PaviaU')['paviaU']


    ##LongKou
    X = sio.loadmat(r"E:\zc_paper\five\code_main\data\LongKou\WHU_Hi_LongKou.mat")['WHU_Hi_LongKou']  # X.shape=(145,145,200)
    y = sio.loadmat(r"E:\zc_paper\five\code_main\data\LongKou\WHU_Hi_LongKou_gt.mat")['WHU_Hi_LongKou_gt']  # [145,145]         #X = sio.loadmat('D:\wodewenjian\HSI_dada\PaviaU')['paviaU']


    test_ratio = 0.90

    patch_size = 15

    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)

    print('\n... ... create train & test data ... ...')
    Xtest, Xtrain, ytest, ytrain = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len



if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')

import torch.nn.functional as F
class NormalizedGeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()



class NGCEandNCEandLab(torch.nn.Module):
    def __init__(self, alpha, beta,num_classes, q=0.7):
        super(NGCEandNCEandLab, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.nce(pred, labels)

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    # 网络放到GPU上
    net = SSFAN().to(device)
    # 交叉熵损失函数
    criterion = NGCEandNCEandLab(alpha=1.0, beta=1.0, num_classes=9, q=0.7)#############################################3
    #criterion = nn.CrossEntropyLoss()
    #criterion = NormalizedCrossEntropy(num_classes=9,scale=0.1)
    #criterion = NormalizedGeneralizedCrossEntropy(num_classes=9,scale=1.0,q=0.7)
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):#
        net.train()
        for i, (data, target) in enumerate(train_loader):#[128, 1, 30, 9, 9]) target: torch.Size([128]
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            print(data.shape)
            outputs = net(data)

            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

        if ((epoch + 1) % 5 == 0 and (epoch + 1) >= 60):
            y_pred_test, y_test = test(device, net, test_loader)
            classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
            #print('oa: ', oa), print('aa: ', aa), print('kappa: ', kappa)
            if(oa>95.5):
               print('classification: ', classification)

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    # target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
    #     , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
    #                 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
    #                 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
    #                 'Stone-Steel-Towers']

    #Pavia
    # target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
    #                 'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']

    # #HanChuan
    # target_names = ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum'
    #     , 'Water spinach', 'Watermelon', 'Greens',
    #                 'Trees', 'Grass', 'Red roof', 'Gray roof',
    #                 'Plastic', 'Bare soil', 'Road', 'Bright object',
    #                 'Water']

    #LongKou
    target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybeam'
        , 'Narrow-leaf soybeam', 'Rice', 'Water',
                    'Roads and houses', 'Mixed weed']

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=100)

    # 只保存模型参数
    #Indian
    # torch.save(net.state_dict(), r'E:\zc_paper\five\code_main\cls_params\Indian\SSFAN\param.pth')

    #PaviaU
    # torch.save(net.state_dict(), r'E:\zc_paper\five\code_main\cls_params\PaviaU\SSFAN\param.pth')

    # #HanChuan
    # torch.save(net.state_dict(), r'E:\zc_paper\five\code_main\cls_params\HanChuan\SSFAN\param.pth')

    #LongKou
    #torch.save(net.state_dict(), r'E:\zc_paper\five\code_main\cls_params\LongKou\SSFAN\param.pth')

    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    print(classification)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    ##Indina
    # file_name = r"E:\zc_paper\five\code_main\cls_result\Indian\SSFAN\classification_report.txt"

    # # #PaviaU
    # file_name = r"E:\zc_paper\five\code_main\cls_result\PaviaU\SSFAN\classification_report.txt"

    # #HanChuan
    # file_name = r"E:\zc_paper\five\code_main\cls_result\HanChuan\SSFAN\classification_report.txt"

    #LongKou
    file_name = r"E:\zc_paper\five\code_main\cls_result\LongKou\SSFAN\classification_report.txt"


    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
    #print('y_pred_test: ',y_pred_test),print('y_pred_test_size: ',y_pred_test.size)

    #print('y_test: ',y_test),print('y_test_size: ',y_test.size)
    print('oa: ', oa), print('aa: ', aa), print('kappa: ', kappa)

    get_cls_map(net, device, all_data_loader, y_all)




