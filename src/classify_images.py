from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import List
from load_dataset import ImageDataset, Transform, TransformToRGB
import torch
from net import *


def train(classes: List[str], net_path: str, batch_size: int, network):
    image_dataset = ImageDataset("data/train", classes, transform=Transform())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=7)

    # get some random training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

    net = network()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # flatten labels
            labels = torch.flatten(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")

    torch.save(net.state_dict(), net_path)


def train_adam(classes: List[str], net_path: str, batch_size: int, network):
    image_dataset = ImageDataset("data/train", classes, transform=Transform())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=7)

    # get some random training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

    net = network()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # flatten labels
            labels = torch.flatten(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")

    torch.save(net.state_dict(), net_path)


def train_v2(classes: List[str], net_path: str, batch_size: int, network):
    image_dataset = ImageDataset("data/train", classes, transform=Transform())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=7)

    # get some random training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

    net = network()

    criterion = nn.KLDivLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = F.log_softmax(net(inputs), dim=1)  # Apply log_softmax to model outputs
            labels_onehot = F.one_hot(labels, num_classes=len(classes)).float()  # Convert labels to one-hot encoding
            loss = criterion(outputs, labels_onehot)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")
    torch.save(net.state_dict(), net_path)


def train_v3(classes: List[str], net_path: str, batch_size: int, network):
    image_dataset = ImageDataset("data/train", classes, transform=TransformToRGB())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=7)

    net = network(num_classes=len(classes))

    # Use Focal Loss
    criterion = FocalLoss(gamma=2)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")
    torch.save(net.state_dict(), net_path)


def train_v4(classes: List[str], net_path: str, batch_size: int, network):
    image_dataset = ImageDataset("data/train", classes, transform=TransformToRGB())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    net = network(num_classes=len(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")
    torch.save(net.state_dict(), net_path)


def test(classes: List[str], net_path: str, batch_size: int, network):
    image_dataset = ImageDataset("data/test", classes, transform=Transform())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=7)

    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # print images
    # imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

    net = network()
    net.load_state_dict(torch.load(net_path))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the {total} test images: {100 * correct // total} %")

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


def test_v3(classes: List[str], net_path: str, batch_size: int, network: BaseNet):
    image_dataset = ImageDataset("data/test", classes, transform=TransformToRGB())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # print images
    # imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

    net = network(num_classes=len(classes))
    net.load_state_dict(torch.load(net_path))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the {total} test images: {100 * correct // total} %")

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


if __name__ == "__main__":
    batch_size = 20
    classes = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

    # net_path = "data/face_image_net.pth"
    # net_path = "data/face_image_net_with_dropout.pth"
    # net_path = "data/face_image_net_with_dropout_removed_blank_images.pth"
    # net_path = "data/face_image_net_with_dropout_removed_blank_images_kl_div_loss.pth"
    # net_path = "data/face_image_net_with_dropout_removed_blank_images_pretrained_resnet_focal_loss.pth"
    # net_path = "data/face_image_net_with_dropout_removed_blank_images_4_conv_layer.pth"
    # net_path = "data/face_image_net_with_dropout_removed_blank_images_4_conv_layer_adam.pth"

    net_path = "data/face_image_net_with_dropout_removed_blank_images_resnet152_.pth"

    # train_adam(classes, net_path, batch_size, NetL4WithDropout)
    # train_v2(classes, net_path, batch_size, NetWithDropout)
    # train_v3(classes, net_path, batch_size, NetResnet18)
    train_v4(classes, net_path, batch_size, NetResnetresnet152)
    # test(classes, net_path, batch_size, NetL4WithDropout)
    # test_v3(classes, net_path, batch_size, NetResnet18)
    test_v3(classes, net_path, batch_size, NetResnetresnet152)
