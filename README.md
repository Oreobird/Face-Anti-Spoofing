# Face-Anti-Spoofing
A simple face anti-spoofing based on CNN trained with HSV + YCrCb color feature.

![Network Architecture](https://github.com/Oreobird/Face-Anti-Spoofing/raw/master/model.png) 

### Dependencies:<br>
		dlib >= 19.17.0
		tensorflow >= 1.12.0
		keras >= 2.2.4
		numpy >= 1.11.1
		scipy >= 0.14
		opencv-python >= 3.4.3

### Usage:<br>
* Data prepare:<br>
		1) Download train data and model file from:<br>
		[https://pan.baidu.com/s/1izOKKs8-CRy6Ykfa13r0Ew](https://pan.baidu.com/s/1izOKKs8-CRy6Ykfa13r0Ew)  code: u99k<br>
		2) untar data and model to your project dir
* Train model:<br>
		python data/NUAA/gen_label.py
		python main.py --train=True
* Test online via camera:<br>
		python main.py --online=True
		
Detailed design doc, please reference to:

![image](https://github.com/Oreobird/effect3d/blob/master/wechat.jpg)

