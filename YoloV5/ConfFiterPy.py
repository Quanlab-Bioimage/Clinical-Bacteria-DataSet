'''置信度筛选'''
import os
import shutil
from os.path import join
import numpy as np

'''置信度筛选txt'''
def ConfFiterTxt():
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet'
    confThre = 0.6
    edgeThre = 0.02
    imgType = '.jpg'
    imgPath = join(root, 'images')
    txtPath = join(root, 'PreTxt\labels')
    saveTxtPath = join(root, 'PreTxt\Fiter\labels')
    saveImgPath = join(root, 'PreTxt\Fiter\images')
    os.makedirs(saveImgPath, exist_ok=True)
    os.makedirs(saveTxtPath, exist_ok=True)
    ls = os.listdir(txtPath)
    fiterRectNumber = 0
    fiterPicNumber = 0
    for name in ls:
        data = np.loadtxt(join(txtPath, name), ndmin=2)
        newData = []
        for item in data:
            # if item[1] < edgeThre or item[2] < edgeThre or item[3] > 1 - edgeThre or item[4] > 1 - edgeThre:
            #     continue
            x0, x1 = item[1] - item[3], item[1] + item[3]
            y0, y1 = item[2] - item[4], item[2] + item[4]
            if edgeThre < x0 < x1 < 1 - edgeThre and edgeThre < y0 < y1 < 1 - edgeThre:
                if item[-1] < confThre:
                    fiterRectNumber += 1
                    newData.append(item)
        if len(newData) > 0:
            nameId = os.path.splitext(name)[0]
            newData = np.array(newData)
            np.savetxt(join(saveTxtPath, name), newData, fmt='%d %.8f %.8f %.8f %.8f %.8f')
            shutil.copyfile(join(imgPath, nameId + imgType), join(saveImgPath, nameId + imgType))
            fiterPicNumber += 1
    print('筛选出的框数量：', fiterRectNumber)
    print('筛选出的图片数量：', fiterPicNumber)

if __name__ == '__main__':
    ConfFiterTxt()

'''
筛选出的框数量： 136
筛选出的图片数量： 123
'''
