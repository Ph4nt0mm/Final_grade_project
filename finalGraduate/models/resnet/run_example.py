import torch
import torchvision.transforms as transforms
import torchvision
import os

from models.CNN_test.layers import Conv2d
from models.resnet.resnet_cifar_models import resnet20

def get_cifar_transofrms():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return transform_train, transform_test


def read_cifar_10(path, train_batch_size, test_batch_size):

    # train - 50000 samples, test - 10000

    transform_train, transform_test = get_cifar_transofrms()

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, 'cifar10'),
                                            train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=os.path.join(path, 'cifar10'),
                                           train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


def copy_conv_layer(old_layer):
    new_lin = Conv2d(
            old_layer.in_channels, old_layer.out_channels, kernel_size=old_layer.kernel_size,
            stride=old_layer.stride, padding=old_layer.padding, bias=old_layer.bias)

    new_lin.weight = old_layer.weight
    new_lin.bias = old_layer.bias

    return new_lin


def copy_conv_layers(model):
    model.conv1 = copy_conv_layer(model.conv1)
    for i in range(3):
        model.layer1[i].conv1 = copy_conv_layer(model.layer1[i].conv1)
        model.layer1[i].conv2 = copy_conv_layer(model.layer1[i].conv2)
    for i in range(3):
        model.layer2[i].conv1 = copy_conv_layer(model.layer2[i].conv1)
        model.layer2[i].conv2 = copy_conv_layer(model.layer2[i].conv2)
    for i in range(3):
        model.layer3[i].conv1 = copy_conv_layer(model.layer3[i].conv1)
        model.layer3[i].conv2 = copy_conv_layer(model.layer3[i].conv2)

    return model


def get_statistic_inp(model):
    print(model.conv1.quant.q_w_sum)
    print(model.conv1.quant.max_weigh_v)
    for i in range(3):
        print(model.layer1[i].conv1.quant.max_weigh_v)
        print(model.layer1[i].conv2.quant.max_weigh_v)
    for i in range(3):
        print(model.layer2[i].conv1.quant.max_weigh_v)
        print(model.layer2[i].conv2.quant.max_weigh_v)
    for i in range(3):
        print(model.layer3[i].conv1.quant.max_weigh_v)
        print(model.layer3[i].conv2.quant.max_weigh_v)


def get_qsin_loss(model):
    res = 0
    res = res + model.conv1.quant.q_w_sum + model.conv1.quant.q_i_sum
    for i in range(3):
        res = res + model.layer1[i].conv1.quant.q_w_sum + model.layer1[i].conv1.quant.q_i_sum
        res = res + model.layer1[i].conv2.quant.q_w_sum + model.layer1[i].conv2.quant.q_i_sum
    for i in range(3):
        res = res + model.layer2[i].conv1.quant.q_w_sum + model.layer2[i].conv1.quant.q_i_sum
        res = res + model.layer2[i].conv2.quant.q_w_sum + model.layer2[i].conv2.quant.q_i_sum
    for i in range(3):
        res = res + model.layer3[i].conv1.quant.q_w_sum + model.layer3[i].conv1.quant.q_i_sum
        res = res + model.layer3[i].conv2.quant.q_w_sum + model.layer3[i].conv2.quant.q_i_sum

    return res

def get_i_qsin_loss(model):
    res = 0
    res = res + model.conv1.quant.q_i_sum
    for i in range(3):
        res = res + model.layer1[i].conv1.quant.q_i_sum
        res = res + model.layer1[i].conv2.quant.q_i_sum
    for i in range(3):
        res = res + model.layer2[i].conv1.quant.q_i_sum
        res = res + model.layer2[i].conv2.quant.q_i_sum
    for i in range(3):
        res = res + model.layer3[i].conv1.quant.q_i_sum
        res = res + model.layer3[i].conv2.quant.q_i_sum

    return res


def get_w_qsin_loss(model):
    res = 0
    res = res + model.conv1.quant.q_w_sum
    for i in range(3):
        res = res + model.layer1[i].conv1.quant.q_w_sum
        res = res + model.layer1[i].conv2.quant.q_w_sum
    for i in range(3):
        res = res + model.layer2[i].conv1.quant.q_w_sum
        res = res + model.layer2[i].conv2.quant.q_w_sum
    for i in range(3):
        res = res + model.layer3[i].conv1.quant.q_w_sum
        res = res + model.layer3[i].conv2.quant.q_w_sum

    return res


def set_quantize(model, quantize: bool = False, bitness: int = 4,
                 quantize_type: str = None,
                 trainable: bool = False, use_qloss: bool = False):
    model.conv1.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)
    for i in range(3):
        model.layer1[i].conv1.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)
        model.layer1[i].conv2.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)
    for i in range(3):
        model.layer2[i].conv1.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)
        model.layer2[i].conv2.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)
    for i in range(3):
        model.layer3[i].conv1.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)
        model.layer3[i].conv2.set_quantize(quantize, bitness,
                                  quantize_type, trainable, use_qloss)


def run_resnet():
    # if __name__ == '__main__':
    path_to_data = './models/resnet/cifar10'
    checkpoint_path = './models/resnet/resnet_20_cifar10_91.73.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet20(10)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state'])
    model.to(device)

    train_loader, validation_loader = read_cifar_10(path_to_data, 1000, 1000)

    model.eval()
    model = copy_conv_layers(model)

    correct = 0
    total = 0

    # get_statistic_inp(model)
    set_quantize(model, True, 8, "static_train")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in zip(range(1), validation_loader):
            print(batch_idx)
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    # get_statistic_inp(model)

    def run_model(num_bits = None, quant_type = None, use_qsin = False):
        model.eval()

        correct = 0
        total = 0

        if num_bits is None:
            set_quantize(model)
        else:
            set_quantize(model, True, num_bits, quant_type, use_qsin, use_qsin)
        print(f"{num_bits} {quant_type}")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(validation_loader):
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.to(device)).sum().item()

        quantized_accuracy = 100 * correct / total

        print('Test accuracy: {}%'.format(quantized_accuracy))

    run_model()
    run_model(8, "dynamic")
    run_model(4, "dynamic")
    run_model(8, "static")
    run_model(4, "static")

    model.train()

    # get_statistic_inp(model)

    criterion = torch.nn.CrossEntropyLoss()

    set_quantize(model, True, 4, "static", True, True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.to(device)

    for batch_idx, (inputs, labels) in zip(range(2), train_loader):
        correct = 0
        total = 0

        # print()
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.to(device)).sum().item()

        # print(outputs)
        loss = criterion(outputs, labels.to(device))
        qsin_w_loss = get_w_qsin_loss(model) * 0.001
        qsin_i_loss = get_i_qsin_loss(model) * 0.0000001

        qsinloss = qsin_w_loss + qsin_i_loss

        # print(get_w_qsin_loss(model))
        # print(get_i_qsin_loss(model))
        # print(loss)

        loss = loss + qsinloss
        # loss_qsin = get_w_qsin_loss(model)
        # loss_qsin.backward()

        loss.backward()
        optimizer.step()

        quantized_accuracy = 100 * correct / total

        print(f'Test {batch_idx} accuracy: {quantized_accuracy}%')

    run_model(4, "static", True)

    # get_statistic_inp(model)