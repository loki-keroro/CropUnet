import matplotlib.pyplot as plt
from MyHandler import MyHandler # 根据你的实际文件和类名进行修改
from PIL import Image
import cv2
import base64
import numpy as np
import json

def Mat2base64(image):
    return str(base64.b64encode(cv2.imencode('.png',image)[1]))[2:-1]

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img

def test_handle():
    _service = MyHandler()

    # 创建一个假设的输入数据
    input_data = '/home/piesat/media/ljh/pycharm_project__ljh/unet-pytorch/cropunet/2_ny_0_0.png'
    img = cv_imread(input_data)
    img_base64data = Mat2base64(img)
    input_data = {
        "image": img_base64data
    }
    image = json.dumps(input_data)

    # 调用你的处理器
    data = _service.preprocess(image)
    data = _service.inference(data)
    data = _service.postprocess(data)
#
    contours, _ = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    loc_arr = []
    # 打印每个轮廓的坐标
    for contour in contours:
        for point in contour:
            x, y = point[0]
            loc_arr.append((x, y))
    print(loc_arr)
    # print(data)
#
test_handle()
