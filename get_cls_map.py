import numpy as np
import matplotlib.pyplot as plt

def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    # #indian
    # for index, item in enumerate(x_list):
    #     if item == 0:
    #         y[index] = np.array([0, 0, 0]) / 255.
    #     if item == 1:
    #         y[index] = np.array([147, 67, 46]) / 255.
    #     if item == 2:
    #         y[index] = np.array([0, 0, 255]) / 255.
    #     if item == 3:
    #         y[index] = np.array([255, 100, 0]) / 255.
    #     if item == 4:
    #         y[index] = np.array([0, 255, 123]) / 255.
    #     if item == 5:
    #         y[index] = np.array([164, 75, 155]) / 255.
    #     if item == 6:
    #         y[index] = np.array([101, 174, 255]) / 255.
    #     if item == 7:
    #         y[index] = np.array([118, 254, 172]) / 255.
    #     if item == 8:
    #         y[index] = np.array([60, 91, 112]) / 255.
    #     if item == 9:
    #         y[index] = np.array([255, 255, 0]) / 255.
    #     if item == 10:
    #         y[index] = np.array([255, 255, 125]) / 255.
    #     if item == 11:
    #         y[index] = np.array([255, 0, 255]) / 255.
    #     if item == 12:
    #         y[index] = np.array([100, 0, 255]) / 255.
    #     if item == 13:
    #         y[index] = np.array([0, 172, 254]) / 255.
    #     if item == 14:
    #         y[index] = np.array([0, 255, 0]) / 255.
    #     if item == 15:
    #         y[index] = np.array([171, 175, 80]) / 255.
    #     if item == 16:
    #         y[index] = np.array([101, 193, 60]) / 255.
    #
    # return y


    # #HanChuan
    # for index, item in enumerate(x_list):
    #     if item == 0:
    #         y[index] = np.array([0, 0, 0]) / 255.
    #     if item == 1:
    #         y[index] = np.array([167, 41, 96]) / 255.
    #     if item == 2:
    #         y[index] = np.array([98, 255, 255]) / 255.
    #     if item == 3:
    #         y[index] = np.array([246, 0, 253]) / 255.
    #     if item == 4:
    #         y[index] = np.array([160, 32, 238]) / 255.
    #     if item == 5:
    #         y[index] = np.array([147, 255, 213]) / 255.
    #     if item == 6:
    #         y[index] = np.array([140, 255, 38]) / 255.
    #     if item == 7:
    #         y[index] = np.array([61, 206, 28]) / 255.
    #     if item == 8:
    #         y[index] = np.array([78, 255, 38]) / 255.
    #     if item == 9:
    #         y[index] = np.array([38, 140, 15]) / 255.
    #     if item == 10:
    #         y[index] = np.array([241, 0, 8]) / 255.
    #     if item == 11:
    #         y[index] = np.array([214, 191, 216]) / 255.
    #     if item == 12:
    #         y[index] = np.array([243, 122, 82]) / 255.
    #     if item == 13:
    #         y[index] = np.array([152, 79, 47]) / 255.
    #     if item == 14:
    #         y[index] = np.array([255, 255, 255]) / 255.
    #     if item == 15:
    #         y[index] = np.array([212, 110, 213]) / 255.
    #     if item == 16:
    #         y[index] = np.array([61, 21, 253]) / 255.
    #
    #
    # return y



    #PaviaU
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([10, 17, 241]) / 255.
        if item == 2:
            y[index] = np.array([10, 247, 11]) / 255.
        if item == 3:
            y[index] = np.array([10, 247, 241]) / 255.
        if item == 4:
            y[index] = np.array([10, 133, 11]) / 255.
        if item == 5:
            y[index] = np.array([240, 17, 241]) / 255.
        if item == 6:
            y[index] = np.array([159, 91, 48]) / 255.
        if item == 7:
            y[index] = np.array([126, 17, 127]) / 255.
        if item == 8:
            y[index] = np.array([240, 17, 11]) / 255.
        if item == 9:
            y[index] = np.array([240, 247, 11]) / 255.
    return y

    # # #LongKou
    # for index, item in enumerate(x_list):
    #     if item == 0:
    #         y[index] = np.array([0, 0, 0]) / 255.
    #     if item == 1:
    #         y[index] = np.array([241, 0, 8]) / 255.
    #     if item == 2:
    #         y[index] = np.array([229, 151, 22]) / 255.
    #     if item == 3:
    #         y[index] = np.array([250, 254, 40]) / 255.
    #     if item == 4:
    #         y[index] = np.array([78, 255, 38]) / 255.
    #     if item == 5:
    #         y[index] = np.array([98, 255, 255]) / 255.
    #     if item == 6:
    #         y[index] = np.array([50, 140, 139]) / 255.
    #     if item == 7:
    #         y[index] = np.array([61, 21, 253]) / 255.
    #     if item == 8:
    #         y[index] = np.array([255, 255, 255]) / 255.
    #     if item == 9:
    #         y[index] = np.array([144, 25, 238]) / 255.
    #
    #
    # return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

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

def get_cls_map(net, device, all_data_loader, y):

    y_pred, y_new = test(device, net, all_data_loader)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))

    # #Indina
    #     # # classification_map(y_re, y, 300,
    #     # #                    r'E:/zc_paper/five/code_main/classification_maps/Indian/SSFAN/' + 'IP_predictions.eps')
    #     # # classification_map(y_re, y, 300,
    #     # #                    'E:/zc_paper/five/code_main/classification_maps/Indian/SSFAN/' + 'IP_predictions.png')
    #     # # classification_map(gt_re, y, 300,
    #     # #                    'E:/zc_paper/five/code_main/classification_maps/Indian/SSFAN/' + 'IP_gt.png')
    #     # # print('------Get classification maps successful-------')
    #     #
    # # #PaviaU
    classification_map(y_re, y, 300,
                       r'E:/zc_paper/five/code_main/classification_maps/PaviaU/SSFAN/' + 'IP_predictions.eps')
    classification_map(y_re, y, 300,
                       'E:/zc_paper/five/code_main/classification_maps/PaviaU/SSFAN/' + 'IP_predictions.png')
    classification_map(gt_re, y, 300,
                       'E:/zc_paper/five/code_main/classification_maps/PaviaU/SSFAN/' + 'IP_gt.png')

    print('------Get classification maps successful-------')
    #     #
    #     # # #HanChuan
    #     # # classification_map(y_re, y, 300,
    #     # #                    r'E:/zc_paper/five/code_main/classification_maps/HanChuan/SSFAN/' + 'IP_predictions.eps')
    #     # # classification_map(y_re, y, 300,
    #     # #                    'E:/zc_paper/five/code_main/classification_maps/HanChuan/SSFAN/' + 'IP_predictions.png')
    #     # # classification_map(gt_re, y, 300,
    #     # #                    'E:/zc_paper/five/code_main/classification_maps/HanChuan/SSFAN/' + 'IP_gt.png')
    #     # #
    #     # # print('------Get classification maps successful-------')
    #     #
    #     # # LongKou
    #     # classification_map(y_re, y, 300,
    #     #                    r'E:/zc_paper/five/code_main/classification_maps/LongKou/SSFAN/' + 'IP_predictions.eps')
    #     # classification_map(y_re, y, 300,
    #     #                    'E:/zc_paper/five/code_main/classification_maps/LongKou/SSFAN/' + 'IP_predictions.png')
    #     # classification_map(gt_re, y, 300,
    #     #                    'E:/zc_paper/five/code_main/classification_maps/LongKou/SSFAN/' + 'IP_gt.png')
    #     #
    #     # print('------Get classification maps successful-------')

    #Indina
    # classification_map(y_re, y, 300,
    #                    r'E:/zc_paper/five/code_main/classification_maps/Indian/SSFTT/' + 'IP_predictions.eps')
    # classification_map(y_re, y, 300,
    #                    'E:/zc_paper/five/code_main/classification_maps/Indian/SSFTT/' + 'IP_predictions.png')
    # classification_map(gt_re, y, 300,
    #                    'E:/zc_paper/five/code_main/classification_maps/Indian/SSFTT/' + 'IP_gt.png')
    # print('------Get classification maps successful-------')

    # # # # #PaviaU
    # classification_map(y_re, y, 300,
    #                    r'E:/zc_paper/five/code_main/classification_maps/PaviaU/SSFAN/' + 'IP_predictions.eps')
    # classification_map(y_re, y, 300,
    #                    'E:/zc_paper/five/code_main/classification_maps/PaviaU/SSFAN/' + 'IP_predictions.png')
    # classification_map(gt_re, y, 300,
    #                    'E:/zc_paper/five/code_main/classification_maps/PaviaU/SSFAN/' + 'IP_gt.png')
    #
    # print('------Get classification maps successful-------')

    # #HanChuan
    # classification_map(y_re, y, 300,
    #                    r'E:/zc_paper/five/code_main/classification_maps/HanChuan/SSFTT/' + 'IP_predictions.eps')
    # classification_map(y_re, y, 300,
    #                    'E:/zc_paper/five/code_main/classification_maps/HanChuan/SSFTT/' + 'IP_predictions.png')
    # classification_map(gt_re, y, 300,
    #                    'E:/zc_paper/five/code_main/classification_maps/HanChuan/SSFTT/' + 'IP_gt.png')
    #
    # print('------Get classification maps successful-------')

    # # LongKou
    # classification_map(y_re, y, 300,
    #                    r'E:/zc_paper/five/code_main/classification_maps/LongKou/SSFTT/' + 'IP_predictions.eps')
    # classification_map(y_re, y, 300,
    #                    'E:/zc_paper/five/code_main/classification_maps/LongKou/SSFTT/' + 'IP_predictions.png')
    # classification_map(gt_re, y, 300,
    #                    'E:/zc_paper/five/code_main/classification_maps/LongKou/SSFTT/' + 'IP_gt.png')
    #
    # print('------Get classification maps successful-------')