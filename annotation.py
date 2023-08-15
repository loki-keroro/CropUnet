import os
import random

#----------------------------------------------------------------------#
#   农田数据集较少，因此没有设验证集
#----------------------------------------------------------------------#
trainval_percent    = 1
train_percent       = 1
#-------------------------------------------------------#
#   指向农田数据集所在的文件夹
#   默认指向根目录下的Datasets
#-------------------------------------------------------#
devkit_path  = 'Datasets'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(devkit_path, 'Labels')
    saveBasePath    = os.path.join(devkit_path, 'ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i  in list:  
        name=total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")
