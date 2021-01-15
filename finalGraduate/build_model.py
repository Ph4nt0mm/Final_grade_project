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


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                # print(n[0])
                # print('Tensor with grad found:', tensor)
                # print(' - gradient:', tensor.grad)
                # print()
            except AttributeError as e:
                getBack(n[0])


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


def process_model(name, func, model, data, crit, optim = None):
    if optim is not None:
        train_loss, train_acc = func(model, data, optim, crit)
    else:
        train_loss, train_acc = func(model, data, crit)

    print(f'{name}Loss: {train_loss:.3f}, '
          f'Acc: {train_acc * 100:.2f}%, ')

    return train_loss, train_acc


def run_model(model, train, valid, test):
    BATCH_SIZE = 8
    QUANT_BIT = 4
    N_EPOCHS = 4

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        repeat=False)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    print("Model created")

    # for name, param in model.named_parameters():
    #     print(name)

    # Обучение

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch + 1:02}')

        process_model("Train \t", train_func, model, train_iterator, criterion, optimizer)
        process_model("Val \t", evaluate_func, model, valid_iterator, criterion)

    process_model("Test \t", evaluate_func, model, test_iterator, criterion)

    # Подсчет скаляров

    model.eval()

    with torch.no_grad():
        for _, iter_v in zip(range(3), train_iterator):
            _ = model.train_quant(iter_v.text.cuda())


    #
    model_nt = copy.deepcopy(model)

    # Динамическая квантизация

    model.set_quantize_layers(True, QUANT_BIT)
    process_model("QuantD TD\t", evaluate_func, model, test_iterator, criterion)
    process_model("QuantD VD\t", evaluate_func, model, valid_iterator, criterion)

    model_nt.set_quantize_layers(True, QUANT_BIT)
    process_model("QuantD TD\t", evaluate_func, model_nt, test_iterator, criterion)
    process_model("QuantD VD\t", evaluate_func, model_nt, valid_iterator, criterion)

    # Статическая необученная квантизация

    model.set_quantize_layers(True, QUANT_BIT, "static")
    process_model("QuantS TS\t", evaluate_func, model, train_iterator, criterion)
    process_model("QuantS VS\t", evaluate_func, model, valid_iterator, criterion)

    model_nt.set_quantize_layers(True, QUANT_BIT, "static")
    process_model("QuantS TS\t", evaluate_func, model_nt, train_iterator, criterion)
    process_model("QuantS VS\t", evaluate_func, model_nt, valid_iterator, criterion)

    model.get_scalers()

    # Дообучение статики

    model.set_quantize_layers(True, QUANT_BIT, "static", True)

    # for name, param in model.named_parameters():
    #     print(name)

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch + 1:02}')

        process_model("Train \t", train_func, model, train_iterator, criterion, optimizer)
        process_model("Val \t", evaluate_func, model, valid_iterator, criterion)

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch + 1:02}')

        process_model("Train \t", train_func, model_nt, train_iterator, criterion, optimizer)
        process_model("Val \t", evaluate_func, model_nt, valid_iterator, criterion)

    model.get_scalers()


    # Статическая обученная квантизация

    process_model("QuantS TS\t", evaluate_func, model, train_iterator, criterion)
    process_model("QuantS VS\t", evaluate_func, model, valid_iterator, criterion)

    process_model("QuantS TS\t", evaluate_func, model_nt, train_iterator, criterion)
    process_model("QuantS VS\t", evaluate_func, model_nt, valid_iterator, criterion)

    model.set_quantize_layers()


''' 
Функция работает по пайплайну торчтекста.
Создает поля, загружает и разбивает данные дефолтного датасета
бинарной классификации комментариев IMDB.
Предобработка данных не производится, создается модель и запускается run_model
'''


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

    run_model(model, train, valid, test)
