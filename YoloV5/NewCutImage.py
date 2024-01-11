import cv2, os
import numpy as np

imgSize = 240
redundanceSize = 40

"""
重命名
"""
def my_rename(img_dir):
    img_list = os.listdir(img_dir)
    for img_path in img_list:
        s_name = img_path.split('.')[0]
        n_name = img_dir + s_name.zfill(6) + '.jpg'
        os.rename(img_dir + s_name + '.jpg', n_name)


'''切图'''
def SLicerImg(path, savePath):
    lists = os.listdir(path)
    for name in lists:
        print(name)
        image = cv2.imdecode(np.fromfile(path + name, dtype=np.uint8), cv2.IMREAD_COLOR)
        # image = cv2.imread(path+name)
        width, height, k = image.shape
        widthMaxNum = int((width - imgSize) / (imgSize - redundanceSize)) + 2
        heigthMaxNum = int((height - imgSize) / (imgSize - redundanceSize)) + 2
        for i in range(widthMaxNum):
            sliceCoorX = i * (imgSize - redundanceSize) if i != widthMaxNum - 1 else width - imgSize
            for j in range(heigthMaxNum):
                sliceCoorY = j * (imgSize - redundanceSize) if j != heigthMaxNum - 1 else height - imgSize
                res = image[sliceCoorX:sliceCoorX + imgSize, sliceCoorY:sliceCoorY + imgSize]
                cv2.imwrite(savePath + '/' + name.split('.')[0] + '_' + str(i) + '_' + str(j) + '.jpg', res)


if __name__ == '__main__':
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\BigPic/'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\Testexp14\SmallPic/'
    SLicerImg(path, savePath)


