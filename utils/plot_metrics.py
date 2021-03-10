import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def process_info(path):
    # process ach_info_8model.log

    res = []
    single = {'model_type': '', 'data': []}
    with open(path, 'r', encoding='utf8') as info:
        for line in info:
            if 'Num Epochs' in line:
                num_epochs = line.split('=')[-1].strip()
                # print(num_epochs)
            if 'Model type' in line:
                model_type = line.split(' ')[-1].strip()
                # print(model_type)
                if single['data']:
                    res.append(single.copy())
                    single = {'model_type': '', 'data': []}
                single['model_type'] = model_type
            if 'eval_performance' in line:
                # [precision, recall, f1score, eval_loss]
                per = [x.split(':')[-1].strip() for x in line.split('\t')[-1].split('|')]
                # print(per)
                single['data'].append(per)
        res.append(single.copy())
        return res


class Data:
    def __init__(self, label):
        self.label = label
        self.data = {
            "precision": [],
            "recall": [],
            "f1": [],
            "eval_loss": []
        }


# plot precision
def plot_data(label_data, metric):
    for d in label_data:
        x = np.arange(1, len(d.data[metric]) + 1, 1)
        y = np.array(d.data[metric], dtype=float)
        plt.plot(x, y, label=d.label)
    plt.legend(fontsize="x-small")
    plt.grid()

    # y轴范围
    if metric != 'eval_loss':
        plt.ylim(0, 1)
        my_y_ticks = np.arange(0, 1, 0.1)
        plt.yticks(my_y_ticks)
    # x轴范围
    plt.xlim(1, 20)
    my_x_ticks = np.arange(1, 20, 1)
    plt.xticks(my_x_ticks)

    # 设置坐标轴名称和标题
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.title(metric + " figure")

    plt.savefig(f"../result/figures/{metric}.png")
    plt.show()


if __name__ == '__main__':
    label_data = []
    path = "../logs/ach_info_8model.log"
    for r in process_info(path):
        data = Data(r['model_type'])
        for d in r['data']:
            data.data["precision"].append(d[0])
            data.data["recall"].append(d[1])
            data.data["f1"].append(d[2])
            data.data["eval_loss"].append(d[3])
        label_data.append(data)
    # plot_data(label_data, 'precision')
    # plot_data(label_data, "recall")
    # plot_data(label_data, "f1")
    plot_data(label_data, "eval_loss")
