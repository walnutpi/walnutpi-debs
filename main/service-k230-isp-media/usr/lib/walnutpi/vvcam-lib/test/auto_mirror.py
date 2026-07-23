'''
每秒自动对摄像头图像水平镜像
'''
import cv2,time
import k230_display
import k230_sensor

from walnutpi import Display,IDE
# 初始化屏幕
Display.init()

# 打开摄像头
cap = k230_sensor.Sensor(640, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, img = cap.read()


count=0
pt=0
fps = 0
mirrof=False
while True:
    #计算帧率
    count+=1    
    if time.time()-pt >=1 : #超过1秒
        fps=1/((time.time()-pt)/count)#计算帧率
        fps = round(fps, 1)
        count=0
        pt=time.time()
        print("FPS:",fps)
    mirrof = not mirrof
    cap.set_hmirror(mirrof) # 水平镜像
        

    # 摄像头读取一帧图像    
    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.putText(img, 'FPS: '+str(fps), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2) #图像绘制帧率
    Display.show(img)
    # IDE.show(img)
    
cap .release() # 关闭摄像头
