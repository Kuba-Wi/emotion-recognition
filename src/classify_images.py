from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import List
from load_dataset import ImageDataset, Transform, TransformToRGB
import torch
from net import *
import matplotlib.pyplot as plt
from datetime import datetime


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


def train_with_additional_data(classes, net_path: str, batch_size: int, model: BaseNet, num_epochs: int, start_from = None):
    image_dataset = ImageDataset("data/train", classes, transform=Transform())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=7)

    total_step = len(data_loader)
    class_correct = list(0. for _ in range(7))
    class_total = list(0. for _ in range(7))

    net = model()
    if start_from is not None:
        net.load_state_dict(torch.load(net_path + f"model_epoch_{start_from}.pth"))

    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    with open(net_path + "log.log", "w") as log_file:
        log_file.write(f"Start at {datetime.now().strftime('%H:%M:%S')}" + '\n')
        log_file.flush()

        for epoch in range(start_from + 1, num_epochs):
            running_loss = 0.0
            running_corrects = 0

            with open(net_path + f"model_epoch_{epoch}.log", "w") as model_file:
                for i, (images, labels) in enumerate(data_loader):
                    # print(f"Batch {i}: Images size: {images.size()}, Labels size: {labels.size()}")  # Debug
                    # Forward pass
                    outputs = net(images)
                    # print(f"Batch {i}: Output size: {outputs.size()}, Labels size: {labels.size()}")  # Debug

                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    running_corrects += (predicted == labels).sum().item()

                    c = (predicted == labels).squeeze()
                    for j in range(len(labels)):
                        label = labels[j]
                        class_correct[label] += c[j].item()
                        class_total[label] += 1

                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}")
                    model_file.write(f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}\n")

                epoch_loss = running_loss / total_step
                epoch_acc = running_corrects / len(data_loader.dataset)

                s = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}'
                print(s)
                log_file.write(s + '\n')

                torch.save(net.state_dict(), net_path + f"model_epoch_{epoch}.pth")

                # Calculate class-wise accuracy
                for i in range(7):
                    if class_total[i]:
                        s = f'Accuracy of class {classes[i]} : {100 * class_correct[i] / class_total[i]:.2f}%'
                        print(s)
                        log_file.write(s + '\n')
                log_file.flush()

        print("Training finished")
        log_file.write(f"End at {datetime.now().strftime('%H:%M:%S')}" + '\n')

    return class_correct, class_total


def test_with_additional_data(classes, net_path: str, batch_size: int, model: BaseNet, num_epochs: List[int]):
    image_dataset = ImageDataset("data/test", classes, transform=Transform())
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=7)

    with open(net_path + f"test_log_{min(num_epochs)}_{max(num_epochs)}.log", "w") as log_file:
        log_file.write(f"Start at {datetime.now().strftime('%H:%M:%S')}" + '\n')
        log_file.flush()

        for epoch in num_epochs:
            class_correct = list(0. for _ in range(len(classes)))
            class_total = list(0. for _ in range(len(classes)))

            net = model()
            net.load_state_dict(torch.load(net_path + f"model_epoch_{epoch + 1}.pth"))

            for i, (images, labels) in enumerate(data_loader):
                outputs = net(images)

                _, predicted = torch.max(outputs, 1)

                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

            epoch_acc = sum(class_correct) / sum(class_total)
            s = f'Epoch [{epoch + 1}], Accuracy: {epoch_acc:.4f}'
            print(s)
            log_file.write(s + '\n')

            # Calculate class-wise accuracy
            for i in range(len(classes)):
                if class_total[i]:
                    s = f'Accuracy of class {classes[i]} : {100 * class_correct[i] / class_total[i]:.2f}%'
                    print(s)
                    log_file.write(s + '\n')
            log_file.flush()

        print("Testing finished")
        log_file.write(f"End at {datetime.now().strftime('%H:%M:%S')}" + '\n')


def run_with_plot():
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    net_path = "data/face_image_net_last/"
    batch_size = 64
    net_model = EmotionRecognitionModel
    num_epochs = 100

    test_with_additional_data(classes, net_path, batch_size, net_model, list(range(36)))

    # class_correct, class_total = train_with_additional_data(classes, net_path, batch_size, net_model, num_epochs)
    #
    # # Calculate class-wise and overall accuracy
    # class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(7)]
    # overall_accuracy = 100 * sum(class_correct) / sum(class_total)
    #
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.bar(len(classes), class_accuracies, color='blue')
    # plt.axhline(y=overall_accuracy, color='r', linestyle='-')
    # plt.xlabel('Class')
    # plt.ylabel('Accuracy (%)')
    # plt.title('Class-wise and Overall Accuracy')
    # plt.show()


if __name__ == "__main__":
    run_with_plot()
    # batch_size = 64
    # # classes = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
    # classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    #
    # # net_path = "data/face_image_net.pth"
    # # net_path = "data/face_image_net_with_dropout.pth"
    # # net_path = "data/face_image_net_with_dropout_removed_blank_images.pth"
    # # net_path = "data/face_image_net_with_dropout_removed_blank_images_kl_div_loss.pth"
    # # net_path = "data/face_image_net_with_dropout_removed_blank_images_pretrained_resnet_focal_loss.pth"
    # # net_path = "data/face_image_net_with_dropout_removed_blank_images_4_conv_layer.pth"
    # # net_path = "data/face_image_net_with_dropout_removed_blank_images_4_conv_layer_adam.pth"
    #
    # net_path = "data/face_image_net_with_dropout_removed_blank_images_resnet152_.pth"
    #
    # # train_adam(classes, net_path, batch_size, NetL4WithDropout)
    # # train_v2(classes, net_path, batch_size, NetWithDropout)
    # # train_v3(classes, net_path, batch_size, NetResnet18)
    # train_v4(classes, net_path, batch_size, NetResnetresnet152)
    # # test(classes, net_path, batch_size, NetL4WithDropout)
    # # test_v3(classes, net_path, batch_size, NetResnet18)
    # test_v3(classes, net_path, batch_size, NetResnetresnet152)



