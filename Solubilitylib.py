import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from transformers import BertTokenizer, BertModel
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import re
import matplotlib_inline
import iFeatureOmegaCLI
import logging
from Bio import SeqIO

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#获取GPU
def try_all_gpus():
    "Defined in :numref:`sec_use_gpu`"
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        #print('model is in processing...')
        # display.display(self.fig)
        # display.clear_output(wait=True)
        # plt.savefig("/home/bli/bin_ProtSol/trian.png")
        
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点
    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
def use_svg_display():
    #display.set_matplotlib_formats('svg')
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class SolubilityDatasetBio(Dataset):
    def __init__(self, biopath, tokenizer, max_length=512):
        self.seqs, self.labels, self.bioinfo = self.load_dataset(biopath)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self, biopath):
        sequences = []
        labels = []
        bioinfo_list = []

        # 使用Biopython来解析fasta文件
        for record in SeqIO.parse(biopath, "fasta"):
            # 获取fasta标题，例如">97522|1|training"
            title = record.description
            # 从标题中提取标签，这里假设标签位于标题的第二部分，使用"|"分隔
            label = int(title.split("|")[1])
            # 获取氨基酸序列
            sequence = str(record.seq)

            sequences.append(sequence)
            labels.append(label)
        protein = iFeatureOmegaCLI.iProtein(biopath)
        protein.get_descriptor("PAAC")
        bioinfo1 = protein.encodings
        protein.get_descriptor("CKSAAGP type 1")
        bioinfo2 = protein.encodings
        bioinfo_list = np.concatenate((bioinfo1,bioinfo2),axis=1) 

        assert len(sequences) == len(labels)
        return sequences, labels, bioinfo_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        sample['bioinfo'] = torch.tensor(self.bioinfo[idx], dtype=torch.float32)

        return sample
    
class SolubilityDatasetBioPrediction(Dataset):
    def __init__(self, biopath, tokenizer, max_length=512):
        self.seqs, self.bioinfo = self.load_dataset(biopath)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self, biopath):
        sequences = []
        bioinfo_list = []

        # 使用Biopython来解析fasta文件
        for record in SeqIO.parse(biopath, "fasta"):
            # 获取fasta标题，例如">97522|1|training"
            title = record.description
            # 从标题中提取标签，这里假设标签位于标题的第二部分，使用"|"分隔
            # 获取氨基酸序列
            sequence = str(record.seq)

            sequences.append(sequence)
        protein = iFeatureOmegaCLI.iProtein(biopath)
        protein.get_descriptor("PAAC")
        bioinfo1 = protein.encodings
        protein.get_descriptor("CKSAAGP type 1")
        bioinfo2 = protein.encodings
        bioinfo_list = np.concatenate((bioinfo1,bioinfo2),axis=1) 

        return sequences, bioinfo_list

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['bioinfo'] = torch.tensor(self.bioinfo[idx], dtype=torch.float32)

        return sample

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for data in data_iter:
            x_input_ids, x_token_type_ids, x_attention_mask = data['input_ids'], data['token_type_ids'], data['attention_mask']
            x_bioinfo = data['bioinfo']
            X = {}
            X['input_ids'] = x_input_ids.cuda()
            X['token_type_ids'] = x_token_type_ids.cuda()
            X['attention_mask'] = x_attention_mask.cuda()
            X['bioinfo'] = x_bioinfo.cuda()
            y = data['labels'].cuda()
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    net.eval()  # 设置为评估模式
    if isinstance(net, nn.Module):
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for data in data_iter:
            x_input_ids, x_token_type_ids, x_attention_mask = data['input_ids'], data['token_type_ids'], data['attention_mask']
            x_bioinfo = data['bioinfo']
            X = {}
            X['input_ids'] = x_input_ids.cuda()
            X['token_type_ids'] = x_token_type_ids.cuda()
            X['attention_mask'] = x_attention_mask.cuda()
            X['bioinfo'] = x_bioinfo.cuda()
            y = data['labels'].cuda()
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def log_init(log_path ="./record/acc.log"): 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(filename = log_path)
    formatter = logging.Formatter("%(asctime)s|%(message)s",)
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

def print_box(message):
    box_width = 40
    message = f" {message} "
    padding = (box_width - len(message)) // 2
    border = '*' * box_width
    padding_str = '*' + ' ' * padding

    print(border)
    print(padding_str + message + ' ' * (box_width - len(padding_str) - len(message)) + '*')
    print(border)
