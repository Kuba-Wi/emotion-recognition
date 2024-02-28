from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from typing import Iterable
from load_dataset import ImageDataset, Transform, TransformToRGB
from net import *
import matplotlib.pyplot as plt
from datetime import datetime
from config import Config
from pathlib import Path
from matplotlib.ticker import FuncFormatter


def train_from_config(config: Config, start_from: int = 0):
    image_dataset = ImageDataset("data/train", config.classes, transform=config.transform())
    data_loader = DataLoader(image_dataset, batch_size=config.batch_size, shuffle=True, num_workers=7)

    total_step = len(data_loader)
    Path(config.net_dir).mkdir(parents=True, exist_ok=True)

    if start_from:
        config.net_model.load_state_dict(torch.load(config.net_dir + f"model_epoch_{start_from}.pth"))

    config.net_model.train()
    criterion = config.criterion
    optimizer = config.optimizer(config.net_model.parameters())

    with open(config.net_dir + "training.log", "a") as log_file:
        log_file.write(f"Start at {datetime.now().strftime('%H:%M:%S')}" + '\n')
        log_file.flush()

        for epoch in range(start_from + 1, config.num_epochs + 1):
            class_correct = list(0. for _ in range(len(config.classes)))
            class_total = list(0. for _ in range(len(config.classes)))
            running_loss = 0.0
            running_corrects = 0

            with open(config.net_dir + f"model_epoch_{epoch}.log", "w") as model_file:
                for i, (images, labels) in enumerate(data_loader):
                    # Forward pass
                    if config.custom_criterion_call is None:
                        outputs = config.net_model(images)
                        loss = criterion(outputs, labels)
                    else:
                        mod_outputs, mod_labels = config.custom_criterion_call(config.net_model(images), labels,
                                                                               len(config.classes))
                        outputs = mod_outputs
                        loss = criterion(outputs, mod_labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Runtime stats and logs
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    running_corrects += (predicted == labels).sum().item()

                    c = (predicted == labels).squeeze()
                    for j in range(len(labels)):
                        label = labels[j]
                        class_correct[label] += c[j].item()
                        class_total[label] += 1

                    s = f"[{epoch}, {i + 1:5d}] loss: {loss.item():.3f}"
                    print(s)
                    model_file.write(s + '\n')

                # Save model
                torch.save(config.net_model.state_dict(), config.net_dir + f"model_epoch_{epoch}.pth")

                # End of epoch stats and logs
                epoch_loss = running_loss / total_step
                epoch_acc = running_corrects / len(data_loader.dataset)
                s = f'Epoch [{epoch}/{config.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}'
                print(s)
                log_file.write(s + '\n')

                # Calculate class-wise accuracy
                for i in range(len(config.classes)):
                    if class_total[i]:
                        s = f'Accuracy of class {config.classes[i]} : {100 * class_correct[i] / class_total[i]:.2f}%'
                        print(s)
                        log_file.write(s + '\n')
                log_file.flush()

        print("Training finished")
        log_file.write(f"End at {datetime.now().strftime('%H:%M:%S')}" + '\n')


def test_from_config(config: Config, num_epochs: Iterable[int]):
    image_dataset = ImageDataset("data/test", config.classes, transform=config.transform())
    data_loader = DataLoader(image_dataset, batch_size=config.batch_size, shuffle=True, num_workers=7)

    with open(config.net_dir + f"testing.log", "a") as log_file:
        log_file.write(f"Start at {datetime.now().strftime('%H:%M:%S')}" + '\n')
        log_file.flush()

        for epoch in num_epochs:
            class_correct = list(0. for _ in range(len(config.classes)))
            class_total = list(0. for _ in range(len(config.classes)))

            config.net_model.load_state_dict(torch.load(config.net_dir + f"model_epoch_{epoch}.pth"))

            for i, (images, labels) in enumerate(data_loader):
                outputs = config.net_model(images)

                _, predicted = torch.max(outputs, 1)
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

            epoch_acc = sum(class_correct) / sum(class_total)
            s = f'Epoch [{epoch}], Accuracy: {epoch_acc:.4f}'
            print(s)
            log_file.write(s + '\n')

            # Calculate class-wise accuracy
            for i in range(len(config.classes)):
                if class_total[i]:
                    s = f'Accuracy of class {config.classes[i]} : {100 * class_correct[i] / class_total[i]:.2f}%'
                    print(s)
                    log_file.write(s + '\n')
            log_file.flush()

        print("Testing finished")
        log_file.write(f"End at {datetime.now().strftime('%H:%M:%S')}" + '\n')


def plot_accuracy_log(config: Config, additional_text: str):
    epochs = []
    accuracy_train = []
    accuracy_test = []

    try:
        with open(config.net_dir + f"training.log", 'r') as file:
            for line in file:
                if 'Epoch' in line:
                    parts = line.strip().split(',')
                    epoch_part = parts[0].split('[')[1].split('/')[0].strip()
                    accuracy_part = parts[2].split(':')[1].strip()
                    epochs.append(int(epoch_part))
                    accuracy_train.append(float(accuracy_part) * 100)

        with open(config.net_dir + f"testing.log", 'r') as file:
            for line in file:
                if 'Epoch' in line:
                    parts = line.strip().split(',')
                    accuracy_part = parts[1].split(':')[1].strip()
                    accuracy_test.append(float(accuracy_part) * 100)
    except Exception as e:
        print(f"Wystąpił błąd podczas wczytywania pliku: {e}")

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, accuracy_train, label='Zbiór treningowy', color='blue')
    plt.plot(epochs, accuracy_test, label='Zbiór testowy', color='red')

    plt.xlabel('Epoki', fontsize=14)
    plt.xlim(min(epochs), max(epochs) + 1)
    plt.xticks(range(min(epochs), max(epochs) + 1))

    plt.ylabel('Dokładność', fontsize=14)
    plt.ylim(0, 105)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.yticks(range(0, 101, 10))

    plt.grid(True)

    max_point_train = (epochs[accuracy_train.index(max(accuracy_train))], round(max(accuracy_train), 2))
    max_point_test = (epochs[accuracy_test.index(max(accuracy_test))], round(max(accuracy_test), 2))

    plt.scatter(max_point_train[0], max_point_train[1], color='blue',
                label=f'Max ({max_point_train[0]}, {max_point_train[1]}%)')
    plt.scatter(max_point_test[0], max_point_test[1], color='red',
                label=f'Max ({max_point_test[0]}, {max_point_test[1]}%)')

    plt.title('Dokładność modelu w kolejnych epokach dla ' + additional_text, fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    config1 = Config(classes=["angry", "fear", "happy", "neutral", "sad", "surprise"],
                     net_model=BasicNet(6),
                     net_dir='data/basic_net_6_classes_batch_4/',
                     batch_size=4,
                     num_epochs=20,
                     optimizer=lambda model_params: optim.SGD(model_params, lr=0.001, momentum=0.9),
                     criterion=nn.CrossEntropyLoss(),
                     transform=Transform)

    config2 = Config(classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                     net_model=BasicNet(7),
                     net_dir='data/basic_net_7_classes_batch_64/',
                     batch_size=64,
                     num_epochs=20,
                     optimizer=lambda model_params: optim.SGD(model_params, lr=0.001, momentum=0.9),
                     criterion=nn.CrossEntropyLoss(),
                     transform=Transform)

    config3 = Config(classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                     net_model=BasicNet(7),
                     net_dir='data/basic_net_7_classes_batch_4/',
                     batch_size=4,
                     num_epochs=20,
                     optimizer=lambda model_params: optim.SGD(model_params, lr=0.001, momentum=0.9),
                     criterion=nn.CrossEntropyLoss(),
                     transform=Transform)

    config4 = Config(classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                     net_model=BasicNet(7),
                     net_dir='data/basic_net_7_classes_batch_4_adam/',
                     batch_size=4,
                     num_epochs=20,
                     optimizer=lambda model_params: optim.Adam(model_params, lr=0.0001),
                     criterion=nn.CrossEntropyLoss(),
                     transform=Transform)

    config5 = Config(classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                     net_model=BasicNet(7),
                     net_dir='data/basic_net_7_classes_batch_4_kldivloss/',
                     batch_size=4,
                     num_epochs=20,
                     optimizer=lambda model_params: optim.SGD(model_params, lr=0.001, momentum=0.9),
                     criterion=nn.KLDivLoss(),
                     transform=Transform,
                     custom_criterion_call=lambda net_outputs, labels, classes_len:
                     (F.log_softmax(net_outputs, dim=1), F.one_hot(labels, num_classes=classes_len).float()))

    config6 = Config(classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                     net_model=BasicNetWithDropout(7),
                     net_dir='data/basic_net_with_dropout/',
                     batch_size=4,
                     num_epochs=20,
                     optimizer=lambda model_params: optim.SGD(model_params, lr=0.001, momentum=0.9),
                     criterion=nn.CrossEntropyLoss(),
                     transform=Transform)

    config7 = Config(classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                     net_model=NetResnet152(7),
                     net_dir='data/resnet_152/',
                     batch_size=64,
                     num_epochs=20,
                     optimizer=lambda model_params: optim.Adam(model_params, lr=0.0001),
                     criterion=nn.CrossEntropyLoss(),
                     transform=TransformToRGB)

    config8 = Config(classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                     net_model=EmotionRecognitionModel(),
                     net_dir='data/advanced_net/',
                     batch_size=64,
                     num_epochs=20,
                     optimizer=lambda model_params: optim.Adam(model_params, lr=0.0001),
                     criterion=nn.CrossEntropyLoss(),
                     transform=Transform)

    # plot_accuracy_log(config4, 'podstawowej sieci neuronowej z trzema warstwami konwolucyjnymi')

    # # train_from_config(config5, start_from=0)
    # # test_from_config(config5, range(1, 21))
    # plot_accuracy_log(config5, 'podstawowej sieci neuronowej z trzema warstwami konwolucyjnymi')
    #
    # # train_from_config(config6, start_from=0)
    # # test_from_config(config6, range(1, 21))
    # plot_accuracy_log(config6, 'podstawowej sieci neuronowej z trzema warstwami konwolucyjnymi\ni dodatkową warstwą odrzucającą')
    #
    # # train_from_config(config7, start_from=0)
    # # test_from_config(config7, range(1, 21))
    # plot_accuracy_log(config7, 'sieci ResNet-152')

    plot_accuracy_log(config8, 'ulepszonej sieci neuronowej z pięcioma warstwami konwolucyjnymi')
