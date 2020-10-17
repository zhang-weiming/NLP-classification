# LSTM 文本分类

## pip主要依赖

    torch=1.2.0
    torchtext=0.6.0
    spacy=2.3.2


## 模型结构

    (embedding): Embedding(6244, 300),
    (lstm): LSTM(300, 128, batch_first=True, bidirectional=True),
    (drop): Dropout(p=0.5, inplace=False),
    (fc): Linear(in_features=256, out_features=1, bias=True)


## 数据集

kaggle真假新闻数据集

下载地址：https://www.kaggle.com/nopdev/real-and-fake-news-dataset


## 参考

tutorial: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0