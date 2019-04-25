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
		download train data and model file from:<br>
		[](https://pan.baidu.com/s/1izOKKs8-CRy6Ykfa13r0Ew)<br>
		code: u99k<br>
* Train model:<br>
		python data/NUAA/gen_label.py
		python main.py --train=True

* Test online via camera:<br>
		python main.py --online=True
