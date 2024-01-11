import warnings
warnings.filterwarnings("ignore")
import os, yaml
import subprocess
import time

from train import main, parse_opt

if __name__ == '__main__':
    # 训练轮数
    epochs = 150
    # 预训练模型位置
    weightsPath = r'./yolov5l6.pt'
    # 网络配置cfg文件
    cfgPath = r'./models/MyYamlI6.yaml'
    # 数据配置文件路径, 若数据变化需要更改数据配置文件对应的目录
    dataPath = r'./data/MyYamlI6Data.yaml'
    # 单类别
    single_cls = False
    # 每批大小, -1为自动计算大小, 建议使用-1
    batchSize = -1
    # 1280或者640
    imgsz = 640
    # 使用cpu设备号 cuda device, i.e. 0 or 0,1,2,3 or cpu
    device = '0'
    # 是否开启多尺度训练
    multiScale = False
    # 模型保存路径
    project = r'./runs/train'
    # 模型保存路径的文件名
    name = 'exp'
    '''
    #根目录下打开cmd
    #输入：activate YoloV5
    #在输入：tensorboard --logdir runs\train --host=0.0.0.0
    tensorboard --logdir ./ --host=0.0.0.0
    '''
    # 可视化启动  程序运行后可通过浏览器输入网址http://0.0.0.0:6006/查看训练过程，局域网电脑通过http://192.168.1.113:6006/查看
    # sp = subprocess.Popen(r'tensorboard --logdir runs\train --host=0.0.0.0')
    # 缓存擦除
    with open(dataPath, 'r', encoding='utf-8') as f:
        dataInfo = yaml.load(f.read(), Loader=yaml.FullLoader)
        tmpPath = os.path.dirname(dataInfo['train'])
        tmpLs = os.listdir(tmpPath)
        for tmpName in tmpLs:
            if os.path.splitext(tmpName)[-1] == '.cache':
                os.remove(os.path.join(tmpPath, tmpName))
    # 参数更新
    opt = parse_opt()
    opt.epochs = epochs
    opt.weights = weightsPath
    opt.cfg = cfgPath
    opt.data = dataPath
    opt.batch_size = batchSize
    opt.imgsz = imgsz
    opt.device = device
    opt.multi_scale = multiScale
    opt.project = project
    opt.name = name
    opt.single_cls = single_cls
    # opt.evolve = 20
    # 训练开始
    main(opt)
    # sp.kill()
