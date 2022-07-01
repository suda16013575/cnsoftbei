目前改程序可以在本地直接运行,需要安装transformers和pytorch
# title部分
## 模型
title目录下是title的模型，模型需要部署在服务器
## 运行
title_predict.py是从模型加载到输入一句话，再到输出结果的过程，我们希望这个函数也能放在服务器
就是说我们传给服务器几个参数，比如我们传给服务器text="",length_control = False,min_length=5,max_length=30，
服务器就可以返回这个函数的结果，**并且最好load模型的部分已经在服务器做好**，这样调用函数时可以省去
load模型的时间，只有处理数据和模型推测时间

# abstract部分
## 模型
abstract目录下是abstract的模型
## 运行
abstract.py与title_predict.py追求的结果是一致的，
但是需要加载title的模型，也就是说abstract.py要和title目录在统一目录级别下，
Prefix_Tuning.py也在同级，因为abstract.py需要调用Prefix_Tuning.py

# label部分
## 模型
label目录下是label的模型
## 运行
同上，希望只做到传输text，调用函数就可以返回label

load模型部分已经在代码中表注