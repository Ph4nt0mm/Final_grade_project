import copy

import torch
import warnings
import random
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from torchtext import data
from torchtext import datasets

from model import CNN

'''
Подсчет доли одинаковых ответов 
'''


def binary_accuracy(preds, y):
    rounded_preds = torch.round(F.sigmoid(preds))

    return (rounded_preds == y).sum() / len(y)


'''
Переводит модель в режим валидации/обучения
Проганяет все итераторы, суммирует точность и лосс
Возыращает их среднее
'''


def train_func(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text.cuda()).squeeze(1)

        loss = criterion(predictions.float(), batch.label.float().cuda())
        acc = binary_accuracy(predictions.float(), batch.label.float().cuda())

        loss.backward()
        optimizer.step()

        epoch_loss += loss
        epoch_acc += acc

    # getBack(loss.grad_fn)
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_func(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text.cuda()).squeeze(1)

            loss = criterion(predictions.float(), batch.label.float().cuda())
            acc = binary_accuracy(predictions.float(), batch.label.float().cuda())

            epoch_loss += loss
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


'''
Запуск модели:
Принимает модель, обучающий, валидационный и тестовый датасеты
разбивются на батчи, создается оптимизатор и функцию потерь
каждую эпоху выводятся результаты обучения и валидации
Далее расчитываутся максимум квантизаторов 
Сравнивается модель с и без квантизации
выводится результат на тестах
'''


def process_model(name, func, model, data, crit, optim=None):
    if optim is not None:
        train_loss, train_acc = func(model, data, optim, crit)
    else:
        train_loss, train_acc = func(model, data, crit)

    print(f'{name}Loss: {train_loss:.3f}, '
          f'Acc: {train_acc * 100:.2f}%, ')

    return train_loss, train_acc


def save_reses(loss_arr, acc_arr, pair_la):
    loss_arr.append(float(pair_la[0].data))
    acc_arr.append(float(pair_la[1].data))


def run_model(model, train, valid, test):
    BATCH_SIZE = 8
    QUANT_BIT = 3
    N_EPOCHS = 8

    val_float_los = []
    val_float_acc = []

    val_dyn_los = []
    val_dyn_acc = []

    val_sta_los = []
    val_sta_acc = []

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        repeat=False)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    print("Model created")

    model.set_quantize_layers()

    # Обучение

    for epoch in range(N_EPOCHS//2):
        print(f'Epoch: {epoch + 1:02}')

        process_model("Train \t", train_func, model, train_iterator, criterion, optimizer)
        save_reses(val_float_los, val_float_acc, process_model("Val \t", evaluate_func, model, valid_iterator, criterion))

        model.set_quantize_layers(True, QUANT_BIT, "dynamic")
        save_reses(val_dyn_los, val_dyn_acc, process_model("ValD \t", evaluate_func, model, valid_iterator, criterion))

        model.set_quantize_layers()

    # Подсчет скаляров

    model.eval()

    with torch.no_grad():
        for _, iter_v in zip(range(3), train_iterator):
            _ = model.train_quant(iter_v.text.cuda())

    # # Динамическая квантизация
    #
    # model.set_quantize_layers(True, QUANT_BIT, "dynamic")
    # process_model("QuantD TD\t", evaluate_func, model, test_iterator, criterion)
    # process_model("QuantD VD\t", evaluate_func, model, valid_iterator, criterion)

    # Статическая необученная квантизация

    model.set_quantize_layers(True, QUANT_BIT, "static")
    save_reses(val_sta_los, val_sta_acc, process_model("ValS VS\t", evaluate_func, model, valid_iterator, criterion))

    # Дообучение статики

    model.set_quantize_layers()
    #
    for epoch in range(N_EPOCHS//2):
        print(f'Epoch: {epoch + 1 + N_EPOCHS/2:02}')

        process_model("Train \t", train_func, model, train_iterator, criterion, optimizer)
        save_reses(val_float_los, val_float_acc, process_model("Val \t", evaluate_func, model, valid_iterator, criterion))

        model.set_quantize_layers(True, QUANT_BIT, "dynamic")
        save_reses(val_dyn_los, val_dyn_acc, process_model("ValD \t", evaluate_func, model, valid_iterator, criterion))
        model.set_quantize_layers(True, QUANT_BIT, "static")
        save_reses(val_sta_los, val_sta_acc, process_model("ValS \t", evaluate_func, model, valid_iterator, criterion))

        model.set_quantize_layers()

    # Статическая обученная квантизация

    # process_model("QuantS TS\t", evaluate_func, model, train_iterator, criterion)
    # process_model("QuantS VS\t", evaluate_func, model, valid_iterator, criterion)
    #
    # model.set_quantize_layers()

    return (np.array(val_float_los), np.array(val_dyn_los), np.array(val_sta_los)), \
           (np.array(val_float_acc), np.array(val_dyn_acc), np.array(val_sta_acc))


def run_model_train_quant(model, train, valid, test):
    BATCH_SIZE = 8
    QUANT_BIT = 3
    N_EPOCHS = 8

    val_float_los = []
    val_float_acc = []

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        repeat=False)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    print("Model created")

    model.set_quantize_layers()

    # Обучение

    for epoch in range(N_EPOCHS//2):
        print(f'Epoch: {epoch + 1:02}')

        process_model("Train \t", train_func, model, train_iterator, criterion, optimizer)
        save_reses(val_float_los, val_float_acc, process_model("Val \t", evaluate_func, model, valid_iterator, criterion))

    # Подсчет скаляров

    model.eval()

    with torch.no_grad():
        for _, iter_v in zip(range(3), train_iterator):
            _ = model.train_quant(iter_v.text.cuda())

    # # Динамическая квантизация
    #
    # model.set_quantize_layers(True, QUANT_BIT, "dynamic")
    # process_model("QuantD TD\t", evaluate_func, model, test_iterator, criterion)
    # process_model("QuantD VD\t", evaluate_func, model, valid_iterator, criterion)


    # Дообучение статики

    model.set_quantize_layers(True, QUANT_BIT, "static", True)

    for epoch in range(N_EPOCHS//2):
        print(f'Epoch: {epoch + 1 + N_EPOCHS/2:02}')

        process_model("Train \t", train_func, model, train_iterator, criterion, optimizer)
        save_reses(val_float_los, val_float_acc, process_model("Val \t", evaluate_func, model, valid_iterator, criterion))

    # Статическая обученная квантизация

    # process_model("QuantS TS\t", evaluate_func, model, train_iterator, criterion)
    # process_model("QuantS VS\t", evaluate_func, model, valid_iterator, criterion)
    #
    # model.set_quantize_layers()

    return np.array(val_float_los), np.array(val_float_acc)


''' 
Функция работает по пайплайну торчтекста.
Создает поля, загружает и разбивает данные дефолтного датасета
бинарной классификации комментариев IMDB.
Предобработка данных не производится, создается модель и запускается run_model
'''


import matplotlib.pyplot as plt
import numpy as np

def run_compare():
    warnings.filterwarnings("ignore")

    print("CNN run")

    SEED = 0

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField()

    train_src, test = datasets.IMDB.splits(TEXT, LABEL)
    train, valid = train_src.split(random_state=random.seed(SEED))

    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5

    TEXT.build_vocab(train, max_size=250, vectors="glove.6B." + str(EMBEDDING_DIM) + "d")
    LABEL.build_vocab(train)

    print("ready to use")
    print("data loaded")

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    sf = []

    sf.append(run_model(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    sf.append(run_model(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    sf.append(run_model(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    sf.append(run_model(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    sf.append(run_model(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    ss = []

    ss.append(run_model_train_quant(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    ss.append(run_model_train_quant(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    ss.append(run_model_train_quant(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    ss.append(run_model_train_quant(model, train, valid, test))

    model = CNN(len(TEXT.vocab), EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model = model.cuda()

    ss.append(run_model_train_quant(model, train, valid, test))

    N_FROM = 3
    N_TO = 8
    # print (sf, ss)

    plt.figure()
    for i in sf:
        plt.subplot(211)
        plt.plot(i[0][0], color="lightpink")
        plt.plot(i[0][1], color="lightgreen")
        plt.plot(range(N_FROM, N_TO), i[0][2], color="lightblue")

        plt.subplot(212)
        plt.plot(i[1][0], color="lightpink")
        plt.plot(i[1][1], color="lightgreen")
        plt.plot(range(N_FROM, N_TO), i[1][2], color="lightblue")

    plt.subplot(211)
    plt.ylabel('loss')

    plt.plot(np.mean(np.array(sf)[:,0], axis=0)[0], label="mean float", color="deeppink")
    plt.plot(np.mean(np.array(sf)[:,0], axis=0)[1], label="mean dynam", color="darkgreen")
    plt.plot(range(N_FROM, N_TO), np.mean(np.array(sf)[:,0], axis=0)[2], label="mean stat", color="darkblue")

    plt.subplot(212)
    plt.ylabel('acc')
    plt.xlabel('n iter')
    plt.plot(np.mean(np.array(sf)[:,1], axis=0)[0], label="mean float", color="deeppink")
    plt.plot(np.mean(np.array(sf)[:,1], axis=0)[1], label="mean dynam", color="darkgreen")
    plt.plot(range(N_FROM, N_TO), np.mean(np.array(sf)[:,1], axis=0)[2], label="mean stat", color="darkblue")
    plt.legend()
    plt.show()

    plt.figure()
    for i in ss:
        plt.subplot(211)
        plt.ylabel('loss')
        plt.plot(i[0])

        plt.subplot(212)
        plt.ylabel('acc')
        plt.plot(i[1])
    plt.show()

def my_plotter(data, align):
    r'''

    :param data: np array with sizes N_MODELS x 2 x N_ITERS
    :param align:
    :return:
    '''