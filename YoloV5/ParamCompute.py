'''默认参数计算'''

import utils.autoanchor as autoAC
import os

def ComputeThr(path):
    ls = os.listdir(path)
    thr = 0
    for name in ls:
        with open(os.path.join(path + name), 'r') as f:
            while True:
                t = f.readline().strip()
                if len(t) == 0: break
                t = t.split(' ')
                t = float(t[3]) / float(t[4])
                if t > thr: thr = t
    return thr

if __name__ == '__main__':
    labelPath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet3\labels/'
    thr = ComputeThr(labelPath) + 0.1
    # 对数据集重新计算 anchors
    new_anchors = autoAC.kmean_anchors('./data/MyYamlI6Data.yaml', 12, 640, thr, 5000, True)
    print(new_anchors)
    print(thr)
    import numpy as np
    data = np.array(new_anchors).reshape([-1, 6])
    for item in data:
        str = ''
        for it in item: str += '%f,' % it
        print(str[:-1])
