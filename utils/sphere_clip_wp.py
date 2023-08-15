import os
import numpy as np
import cv2
import threading
import math

FACES = ['nz', 'pz', 'px', 'nx', 'ny', 'py']

def sphere_to_cube(file_path, resolution=2048, format="hdr", output="output"):
    # im = cv2.imread(file_path)
    im = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    hsize = resolution / 2
    pos_array = np.arange(0, resolution * resolution, 1)
    axA_array = np.floor_divide(pos_array, resolution)
    axB_array = np.fmod(pos_array, resolution)
    output_cubes = []
    tasks = []
    for i in range(0, 5):
        # output_cube = os.path.join(output, "%s%s.%s" % ("sp_", FACES[i], "png"))
        # output_cubes.append(output_cube)
        filename_with_extension = os.path.basename(file_path)
        filename_without_extension = os.path.splitext(filename_with_extension)[0]
        output_cube = os.path.join(output, filename_without_extension)
        task = threading.Thread(target=sphere_to_cube_process,
                                args=(im, i, axA_array, axB_array, resolution, hsize, format, output_cube),
                                name="sphere_to_cube_" + str(i))
        task.start()  # 启动进程
        tasks.append(task)
    for task in tasks:
        task.join()
    return output_cubes

def sphere_to_cube_process(im, face_id, axA_array, axB_array, size, hsize, format, output_cube):
    # nz
    if FACES[face_id] == 'nz':
        x_array = np.full(size * size, hsize)
        y_array = - axB_array + np.full(size * size, hsize)
        z_array = - axA_array + np.full(size * size, hsize)
    # pz
    elif FACES[face_id] == 'pz':
        x_array = np.full(size * size, -hsize)
        y_array = axB_array + np.full(size * size, -hsize)
        z_array = - axA_array + np.full(size * size, hsize)
    # px
    elif FACES[face_id] == 'px':
        x_array = axB_array + np.full(size * size, -hsize)
        y_array = np.full(size * size, hsize)
        z_array = - axA_array + np.full(size * size, hsize)
    # nx
    elif FACES[face_id] == 'nx':
        x_array = - axB_array + np.full(size * size, hsize)
        y_array = np.full(size * size, -hsize)
        z_array = - axA_array + np.full(size * size, hsize)
    # ny 下
    elif FACES[face_id] == 'ny':
        x_array = axB_array + np.full(size * size, -hsize)
        y_array = -axA_array + np.full(size * size, hsize)
        z_array = np.full(size * size, -hsize)

    r_array = np.sqrt(x_array * x_array + y_array * y_array + z_array * z_array)
    theta_array = np.arccos(z_array / r_array)
    phi_array = -np.arctan2(y_array, x_array)
    ix_array = np.floor_divide((im.shape[1] - 1) * phi_array, (2 * math.pi))
    iy_array = np.floor_divide((im.shape[0] - 1) * (theta_array), math.pi)
    ix_array = np.where(ix_array >= 0, ix_array, im.shape[1] + ix_array)
    iy_array = np.where(iy_array >= 0, iy_array, im.shape[0] + iy_array)
    index_array = iy_array * im.shape[1] + ix_array
    reshape_array = im.reshape((im.shape[0] * im.shape[1], 3))
    color_side = reshape_array[index_array.astype(int)]
    color_side = color_side.reshape((size, size, 3))
    clip_image(FACES[face_id], color_side, output_cube)

def clip_image(face, img, output_file):
    """
    下视角按2048*2048切割后再缩放成1024*1024；
    其他视角先从3072以下截图，再裁剪成1024*1024；
    """
    if face  == 'ny':
        img = np.rot90(img, 1)
        width, height, _ = img.shape
        size = 2048
        re_size = 1024
        # 计算覆盖输入文件所需的瓦片数
        num_tiles_x = width // size
        num_tiles_y = height // size
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                # 计算瓦片的坐标
                x_min = x * size
                y_min = y * size
                x_max = x_min + size
                y_max = y_min + size
                # 使用cv裁剪瓦片
                save_file = f'{output_file}_{face}_{x_min}_{y_min}.png'
                # save_file = os.path.join(output_file, "_%s_%s_%s%s" % (face, x_min, y_min, ".png"))
                color_side = cv2.resize(img[x_min:x_max, y_min:y_max],(re_size,re_size))
                if (veg_extract(color_side) and Is_approximately_pure_color_image(color_side)):
                    # cv2.imwrite(save_file, color_side)
                    cv2.imencode('.jpg', color_side)[1].tofile(save_file)

    #其他视角
    else:
        img = img[3072:, :]
        width, height, _ = img.shape
        size = 1024
        # 计算覆盖输入文件所需的瓦片数
        num_tiles_x = width // size
        num_tiles_y = height // size
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                # 计算瓦片的坐标
                x_min = x * size
                y_min = y * size
                x_max = x_min + size
                y_max = y_min + size
                # 使用cv裁剪瓦片
                save_file = f'{output_file}_{face}_{x_min}_{y_min}.png'
                # save_file = os.path.join(output_file, "_%s_%s_%s%s" % (face, x_min, y_min, ".png"))
                color_side =  cv2.resize(img[x_min:x_max, y_min:y_max],(size,size), interpolation=cv2.INTER_AREA)
                if (veg_extract(color_side) and Is_approximately_pure_color_image(color_side)):
                    # cv2.imwrite(save_file, color_side)
                    cv2.imencode('.jpg', color_side)[1].tofile(save_file)   #中文路径

def pitch_process_clip(input_folder_path, output_folder_path):
    file_list = os.listdir(input_folder_path)
    # 输出所有文件
    for file_name in file_list:
        file_extension = file_name.split('.')[-1]
        if file_extension.lower() == 'jpg':
            file_path = os.path.join(input_folder_path, file_name)
            os.makedirs(output_folder_path, exist_ok=True)
            sphere_to_cube(file_path, resolution=4096, format="JPG", output=output_folder_path)

def veg_extract(img):
    B, G, R = cv2.split(img)
    # 计算植被指数
    # cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    cive = 2.4 * G - B - R
    gray = cive.astype('uint8')
    # Apply thresholding to the image
    ret, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Calculate the percentage of white pixels in the image
    white_pixels = np.sum(thresh == 1)
    total_pixels = img.shape[0] * img.shape[1]
    percentage_white = (white_pixels / total_pixels) * 100
    if percentage_white > 30:
        return True
    else:
        return False

def Is_approximately_pure_color_image(image, threshold=20):
    # 将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_value = np.mean(gray_image)
    # 计算每个像素与平均像素值之间的差异
    pixel_diff = np.abs(gray_image - mean_value)
    # 计算差异的标准差或平均绝对差
    diff_std = np.std(pixel_diff)
    # 判断图像是否为近似纯颜色图像
    if diff_std >= threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    file_path = r'E:\project\高明区全景识别\data\1008高明查违\第四期\100MEDIA'
    out_dir = r'E:\project\高明区全景识别\data\img_1024_4'

    pitch_process_clip(file_path, out_dir)
