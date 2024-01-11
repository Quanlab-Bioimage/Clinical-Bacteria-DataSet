import os
from os.path import join

import numpy as np

from MyUtil import ReadXML

'''Xml转Txt'''
def XmlToTxt():
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\PreTxt\Fiter\Annotationsnew'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\PreTxt\Fiter\labelsNew'
    os.makedirs(savePath, exist_ok=True)
    ls = os.listdir(path)
    for name in ls:
        res, data = ReadXML(join(path, name))
        if res:
            newData = []
            for item in data:
                newData.append([int(item[4]), *item[:4]])
            if len(newData) > 0:
                np.savetxt(join(savePath, os.path.splitext(name)[0] + '.txt'), newData, fmt='%d %.8f %.8f %.8f %.8f')

if __name__ == '__main__':
    XmlToTxt()
