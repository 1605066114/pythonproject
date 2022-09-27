import glob
import numpy as np
import torch
import os
import cv2
from model.dsenet_model  import Dsenet
from model.dsenet_model import  Dsenet
from skimage.metrics import structural_similarity
from sklearn import metrics
from scipy.ndimage.interpolation import zoom

def normalization(data):
    _range = np.max(data) - np.min(data)
    return ((data - np.min(data)) / _range)*255

def calculate_metric_percase(pred, gt):
    RMSE = metrics.mean_squared_error(pred, gt, squared=False)
    SSIM = structural_similarity(pred, gt)
    return RMSE, SSIM

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = Dsenet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('pth/epoch_950.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/test/image/*.npy')
    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = 'predictions/'+test_path.split('.')[0] + '_res.npy'
        # 读取图片
        img = np.load(test_path)
        wid, high = img.shape
        img = zoom(img, (384 / wid, 256 / high), order=3)

        # 转为灰度图
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # m=img_tensor.max()
        # img_tensor=img_tensor/m
        # # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        wid, high = pred.shape
        pred = zoom(pred, (512 / wid, 512 / high), order=3)
        save_res_path = 'predictions/result/' + test_path.split('.')[0].split('\\')[-1].split('_')[0] + '_' + str(int(test_path.split('.')[0].split('\\')[-1].split('_')[-1])) + '.npy'
        # pred=normalization(pred)
        # 处理结果
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        # 保存图片
        np.save(save_res_path, pred)