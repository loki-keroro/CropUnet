import os
import cv2
import base64
import numpy as np
import json
from PIL import Image

#opencv 转 base 64
def Mat2base64(image):
    return str(base64.b64encode(cv2.imencode('.png',image)[1]))[2:-1]

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img

def img2byte(image):
    img_encode = cv2.imeneode ('.png',image)[1]
    image_bytes = img_encode.tostring()
    return image_bytes

img_path = r'E:\AAA\2_ny_0_0.png'
img = cv_imread(img_path)
img_base64data = Mat2base64(img)

input_data = {
    "image": img_base64data
}
json_data = json.dumps(input_data)
# print(json_data)
# # 使用json.loads()将其解析为Python字典
# data = json.loads(json_data)
# image_base64 = data["image"]
# image_bytes = base64.b64decode(image_base64)
# image_np = np.frombuffer(image_bytes, dtype=np.uint8)
# image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

import requests

response = requests.post('http://192.168.8.27:8083/predictions/cropunet',
                         data=json_data)
print(response)
loc_arr = response.text
data_list = json.loads(loc_arr)

# 使用列表推导式将每个列表转换为ndarray
array_representation = [np.expand_dims(np.array(lst), axis=1) for lst in data_list]
print(len(array_representation))
# 将得到的列表转换为元组
tuple_of_arrays = tuple(array_representation)
print(tuple_of_arrays[2].shape)

# 3. 创建一个新的空白图像，并绘制所有轮廓
mask = np.zeros_like(img)
cv2.drawContours(mask, tuple_of_arrays, -1, (255), thickness=cv2.FILLED)
# cv2.imwrite('path_to_save_new_mask.png', mask)
cv2.imshow('Mask', mask)
cv2.waitKey(0)