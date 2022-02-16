# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Config(object):

    """配置参数"""
    def __init__(self, n_classes=10):
        self.model_name = 'TextCNN'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.num_classes = n_classes                                           # 类别数

        self.vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'vocab.txt')                                   # 词表
        self.n_vocab = len(open(self.vocab_path,'r',encoding='utf-8').readlines()) # 词表大小，在运行时赋值

        self.learning_rate = 1e-3                                       # 学习率
        self.embedding_size = 256                                       # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)

        self.pretrained_emb = 'bert'
        if self.pretrained_emb == 'bert':
            self.embedding_weight = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'pretrained_embeddings/bert_word_embedding.pt'))
            self.embedding_size = self.embedding_weight.shape[1]

'''Convolutional Neural Networks for Sentence Classification'''


class TextCNN(nn.Module):
    def __init__(self,config:Config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embedding_size, padding_idx=0)

        if config.pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(config.embedding_weight,requires_grad=False)
        
#         for para in self.embedding.parameters():
#             para.requires_grad = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels = 1, out_channels = config.num_filters, \
            kernel_size = (k, config.embedding_size), stride=1) for k in config.filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(config.num_filters * len(config.filter_sizes), 768)
    
        self.fc_2 = nn.Linear(768, config.num_classes)
        
        
        self.fc_half = nn.Linear(768*2, 768)
        print("config.num_filters * len(config.filter_sizes)", config.num_filters * len(config.filter_sizes))

    def conv_and_pool(self, x, conv):
        # x -> [batch_size, 1, seq_len, embedding_size]
        x = F.relu(conv(x)).squeeze(3) # conv(x) -> [batch_size, num_filters, seq_len, 1] ->[batch_size, num_filters, seq_len]
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # max_pool1d -> [batch_size, num_filters, 1] -> [batch_size, num_filters]
        return x

    def forward(self, x, g_emb=None):
        out = self.embedding(x) # [batch_size, seq_len, embedding_size]
        out = out.unsqueeze(1) # [batch_size, 1, seq_len, embedding_size]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # [batch_size, len(config.filter_sizes)*num_filters]
        
        if g_emb is None:
            g_emb = torch.zeros_like(out)
        
        out = torch.cat([out, g_emb], dim=1)

        out = self.dropout(out)
        out = self.fc_half(out)
        
        out = self.dropout(out)
        out = self.fc_1(out)
        out = self.dropout(out)
        out = self.fc_2(out)
        return out
    
    
    def text_cnn_embedding_infer(self, x, g_emb=None):
        out = self.embedding(x) # [batch_size, seq_len, embedding_size]
        out = out.unsqueeze(1) # [batch_size, 1, seq_len, embedding_size]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # [batch_size, len(config.filter_sizes)*num_filters]
        
#         if g_emb is None:
#             g_emb = torch.zeros_like(out)
        if g_emb is None:
            g_emb = torch.zeros_like(out) #out
        
        out = torch.cat([out, g_emb], dim=1)

# #         out = self.dropout(out)
        out = self.fc_half(out)
        
#         out = self.dropout(out)
        out = self.fc_1(out)
        return out


