# 预训练模型的下载

百度网盘链接: https://pan.baidu.com/s/190g13zLyqCHXrv8DEhHPYQ 提取码: **cdaq** 

共有abstract、title、label三个文件夹，对应实现不同功能的预训练模型。

# 使用方法

首先在百度云盘中下载模型到对应的文件夹

## 标题预测

运行title_predict.py，调用predict_title_demo函数

参数设置如下：

- text——预测的文本
- length_control——是否对预测标题文本输出进行控制
- min_length——输出文本长度的最小值
- max_length——输出文本长度的最大值

## 摘要预测

运行abstract_predict.py，调用predict_abstract_demo函数

函数参数设置与标题摘要一致

## 类别预测

运行label_predict.py，调用predict_label_demo函数

输入的参数为预测的文本

------

联系邮箱：jlwu5@stu.suda.edu.cn
