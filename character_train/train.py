import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
	model = YOLO('yolo11s.yaml')   # 修改yaml
	model.load('yolo11s.pt')  #加载预训练权重
	model.train(data='.\\dataset\\GENSHIN.yaml',   #数据集yaml文件
	            imgsz=320,
	            epochs=100,
	            batch=16,
	            workers=20,  
	            device=0,   #没显卡则将0修改为'cpu'
	            optimizer='SGD',
                amp = False,
	            cache=False,   #服务器可设置为True，训练速度变快
	)