import re
from xml.dom.minidom import parse
import xml.dom.minidom

from utils.json_io import write_json


def convert_xml2json(input_path: str, output_path: str):
    dom_tree = parse(input_path)
    root_ele = dom_tree.documentElement
    # print(root_ele.nodeName)
    select_section = root_ele.getElementsByTagName('section')[0]
    questions = select_section.getElementsByTagName('questions')
    doc_dic_list = []
    # questions 一个主题 题目
    for main_question in questions:
        background_ele = main_question.getElementsByTagName('text')[0]
        background = background_ele.childNodes[0].data
        # 去掉图片标记
        background = re.sub(r'[*]{2}.*?[*]{2}', '', background)
        sub_questions = main_question.getElementsByTagName('question')
        # 主题 题目下的单个题目
        for sub_question in sub_questions:
            doc_dic = {}
            doc_dic['id'] = sub_question.getAttribute('id')
            doc_dic['background'] = background
            question = sub_question.getElementsByTagName('text')[0]
            doc_dic['question'] = question.childNodes[0].data.replace('_', '')
            # options
            for option in sub_question.getElementsByTagName('option'):
                doc_dic[option.getAttribute('value')] = option.childNodes[0].data
            print(doc_dic)
            doc_dic_list.append(doc_dic)
        write_json(file_path=output_path, data=doc_dic_list)


if __name__ == '__main__':
    input_path = '../data/test_data/test_G.xml'
    output_path = '../data/test_data/test_G.json'
    convert_xml2json(input_path, output_path)
