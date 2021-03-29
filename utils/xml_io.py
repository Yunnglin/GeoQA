import re
from xml.dom.minidom import parse
import xml.dom.minidom

from utils.json_io import write_json


def get_answer(answers):
    if answers:
        answers = list(answers)
        for answer in answers:
            yield answer
    else:
        yield 'G'


def trim_space(sentence):
    return re.sub(r'[\s]', '', sentence)


def convert_xml2json(input_path: str, output_path: str, answers):
    dom_tree = parse(input_path)
    root_ele = dom_tree.documentElement
    # print(root_ele.nodeName)
    select_section = root_ele.getElementsByTagName('section')[0]
    questions = select_section.getElementsByTagName('questions')
    doc_dic_list = []
    answers = get_answer(answers)
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
            doc_dic['background'] = trim_space(background)
            question = sub_question.getElementsByTagName('text')[0]
            doc_dic['question'] = trim_space(question.childNodes[0].data.replace('_', ''))
            doc_dic['answer'] = next(answers)
            # options
            for option in sub_question.getElementsByTagName('option'):
                doc_dic[option.getAttribute('value')] = trim_space(option.childNodes[0].data)
            print(doc_dic)
            doc_dic_list.append(doc_dic)
    write_json(file_path=output_path, data=doc_dic_list)
    return doc_dic_list


if __name__ == '__main__':
    file_label = 'D'
    if file_label == 'D':
        answers = 'DBABDDAACCCBACC'
    else:
        answers = None

    input_path = f'../data/test_data/test_{file_label}.xml'
    output_path = f'../data/test_data/test_{file_label}.json'
    res = convert_xml2json(input_path, output_path, answers)
    print(res)
