# coding= utf-8
import Prefix_Tuning
from transformers.models.t5 import T5ForConditionalGeneration
from transformers.models.bert import BertTokenizer
from transformers import BertTokenizer, T5Config, T5ForConditionalGeneration
import torch


def predict_abstract_demo(text,length_control = False,min_length=5,max_length=30):
    """

    :param text: 输入的文本
    :param length_control: 是否要求对输出的文本进行长度控制
    :param min_length: 最短长度
    :param max_length: 最长长度
    :return: 预测的摘要
    """
    # load model
    model = "title"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = Prefix_Tuning.T5GenerationArgs()
    config = T5Config.from_pretrained(model)
    tokenizer = BertTokenizer.from_pretrained(model)
    plm = T5ForConditionalGeneration.from_pretrained(model)
    model = Prefix_Tuning.PrefixtuningTemplate(args, config, plm)
    model.modify_plm()
    model.load_state_dict(torch.load("abstract/prefix_mlp.pt"), strict=False)
    # load model end!
    input_text = tokenizer.encode_plus(text, return_tensors="pt")
    model.to(device)
    input_text.to(device)
    input_ids = input_text["input_ids"]
    input_ids.to(device)
    attention_mask = input_text["attention_mask"]
    attention_mask.to(device)
    # use model to generate abstract
    if length_control:
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, min_length=min_length,max_length=max_length).cpu().numpy()[0]
    # return title
    else:
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask).cpu().numpy()[0]
    return ''.join(tokenizer.decode(output[1:], skip_special_tokens=True)).replace(' ', '')


if __name__ == '__main__':
    text = "2015年06月11日08:42有0人参与0条评论>10条跨市班线将新增配客点,需要乘坐这些班线的乘客,以后可以在新增的配客点上车了。日前,记者从东莞市道路运输管理局了解到,为满足旅客需要,根据东莞道路客运业客流的实际情况,部分客运班线将申办配客站。涉及调整的班线大部分是省内班线,包括到汕头、深圳、阳江等地的班线,还有一条到湖南新田的班线。起讫站点拟申办配客点汕头市汽车客运站—虎门汽车客运站东莞市南城汽车客运站石龙汽车客运站—深圳北汽车客运站东莞市石排汽车客运站、东莞市大朗汽车客运站横沥汽车客运站—深圳北汽车客运站东莞市常平汽车客运站、东莞市樟木头振通汽车客运站深圳北汽车客运站—常平汽车客运站东莞市樟木头振通汽车客运站深圳北汽车客运站—东莞市石龙汽车客运站东莞市石排汽车客运站、东莞市大朗汽车客运站东莞市大朗汽车客运站—广州番禺市桥汽车站东莞市南城汽车客运站湖南新田县汽车站—东莞市汽车客运东站东莞市汽车客运站、东莞市南城汽车客运站阳江汽车客运站—东莞市汽车客运东站东莞市汽车客运站、东莞市虎门汽车客运站河源汽车客运站—东莞市南城汽车客运站东莞市中堂汽车客运站东莞市石龙汽车客运站—深圳机场汽车客运站东莞市东城城市候机楼客运配客点、东莞市南城城市候机楼客运配客点版权声明:凡注明来源为“东莞阳光网”的所有文字、图片、音视频、美术设计和程序等作品,版权均属东莞阳光网或相关权利人专属所有或持有所有。未经本网书面授权,不得进行一切形式的下载、转载或建立镜像。否则以侵权论,依法追究相关法律责任。"
    output = predict_abstract_demo(text)
    print(output)
