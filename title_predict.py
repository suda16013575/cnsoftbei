# coding= utf-8
from transformers.models.t5 import T5ForConditionalGeneration
from transformers.models.bert import BertTokenizer
import torch


def predict_title_demo(text,length_control = False,min_length=5,max_length=30):
    """
    :param text: 输入的文本
    :param length_control: 是否要求对输出的文本进行长度控制
    :param min_length: 最短长度
    :param max_length: 最长长度
    :return: 预测的标题
    """

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './title'
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load model end
    # encoder
    input_text = tokenizer.encode_plus(text, return_tensors="pt")
    input_text.to(device)
    input_ids = input_text["input_ids"]
    input_ids.to(device)
    attention_mask = input_text["attention_mask"]
    attention_mask.to(device)
    # use model to generate title
    if length_control:
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                            eos_token_id=tokenizer.sep_token_id, min_length=min_length,max_length=max_length).cpu().numpy()[0]
    else:
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                            eos_token_id=tokenizer.sep_token_id).cpu().numpy()[0]
    # return title
    return ''.join(tokenizer.decode(output[1:], skip_special_tokens=True)).replace(' ', '')


if __name__ == '__main__':
    text = "美国作家托马斯·弗里德曼在畅销书《世界是平的》里描述了全球化进程及其给人们生活带来的巨大改变。然而，在即将过去的2019年，世界遭到逆全球化暗流的冲击，面临着治理、信任、和平和发展四大“赤字”的挑战。对此，弗里德曼2019年10月在接受采访时说，全球化没有好与坏，关键在于你如何掌控它。他过去三十年间多次来过中国，他说中国给他留下了深刻的印象。 作为全球第二大经济体，中国2019年在维护经济全球化、削减全球四大“赤字”方面的努力，同样令世界印象深刻。 2019年，气候变化、网络安全、难民危机等非传统安全威胁持续蔓延。个别国家为实现自身利益最大化，推行单边主义和贸易保护主义，冲击全球治理体系和多边机制。国际货币基金组织一年四次下调全球经济增长率至3%，这是2008年全球金融危机以来的最低水平；美欧等经济体国债收益率曲线出现倒挂，投资者的避险情绪正在蔓延。 在这样的背景下，中国倡导的共商共建共享的全球治理观，可谓破解“治理赤字”的一剂良方。2019年，中国通过二十国集团、金砖峰会等多个国际合作平台，重申维护以联合国为核心的国际体系，努力构建更加公正合理的全球治理体系。中国共产党十九届四中全会明确提出，中国将积极参与全球治理体系的改革和建设，传递出坚定不移维护世界和平、促进共同发展的明确信号。 治理赤字加剧的重要原因在于信任赤字的扩大。2019年，国际竞争摩擦日趋增多，国际社会信任和合作的基础受到侵蚀。信任是国际关系最好的黏合剂，破解信任赤字，需要加强不同文明间的交流对话。这一年，中国举办亚洲文明对话大会，为世界各国超越文明冲突与文明隔阂提供了有益借鉴，达成了普遍共识。 2019年，地区冲突和局部战争持续不断，恐怖主义活动猖獗，和平赤字愈发突显。一些西方国家奉行“新干涉主义”，藉“人权”之名使用武力干涉他国内政，造成严重人道主义危机。近来，欧美社交网站掀起一股“十年挑战”风潮，许多网友通过照片对比晒出10年来的变化。然而，对于叙利亚、利比亚、伊拉克等战乱国家的网友来说，曾经的繁华因战火而凋零，他们晒出的是家破人亡的心酸和无奈。 作为联合国安理会常任理事国，2019年中国参与了几乎所有国际和地区热点问题的解决进程，在朝鲜半岛核问题、阿富汗、叙利亚等问题上发挥了建设性作用。在这一年里，中国不仅参加国际反恐军事演习，更有超过2500名中国维和官兵坚守在全球多个战乱地区。目前，中国是安理会常任理事国中第一大出兵国，也是联合国维和行动的主要出资国。对此，联合国副秘书长阿图尔•哈雷评价说，中国在联合国维和行动中发挥着独特和关键作用，没有中国这样的会员国支持，联合国维和行动不可能取得今天这样的成就。 需要指出的是，无论是破解治理赤字和信任赤字，还是破解和平赤字，归根到底，都离不开发展问题的解决。当前，国际社会对如何推进全球发展存在分歧。西方国家日益陷入发展模式困境，从“政治素人”的异军突起，到英国“脱欧”等“黑天鹅”事件层出不穷，上世纪80年代以来以“去监管化”为重要特征的新自由主义经济学加剧了南北发展的不平衡，也使得西方国家中产阶级规模持续缩小，造成民粹主义和民族主义回潮，弊端凸显，备受诟病。 作为新兴经济体的重要代表，中国主张坚持公平包容，打造平衡普惠的发展模式，让世界各国人民共享经济全球化发展成果。这显然切合实际，符合国际社会共同利益。在这一理念指导下，中国在2019年继续推进“一带一路”倡议，为国际社会提供更多优质公共产品；举办第二届中国国际进口博览会，为各国生产商打开“机遇之门”。与此同时，中国在电商、人工智能、大数据、区块链等领域新技术的蓬勃发展，为破解发展赤字进行了新的探索和实践。 在推选一个最能代表2019年国际形势的汉字时，中国人选了“难”这个字。虽然全球范围内的双边关系、多边关系和地区形势难题不断，但中国人仍将迎难而上，在攻坚克难中赢得新发展，为这个充满变数的世界注入更多确定性。"
    output = predict_title_demo(text)
    print(output)