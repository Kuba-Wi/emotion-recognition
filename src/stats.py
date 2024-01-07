import os
import matplotlib.pyplot as plt
from typing import Dict, List


def count_images_in_folder(folder_path: str) -> int:
    """Liczy ilość plików w podanym folderze."""
    return len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])


def count_emotions_images(base_path: str, emotions: list) -> Dict[str, int]:
    """Zlicza ilość zdjęć dla każdej emocji w podanym katalogu."""
    counts = {}
    for emotion in emotions:
        emotion_path = os.path.join(base_path, emotion)
        counts[emotion] = count_images_in_folder(emotion_path)
    return counts


def plot_emotion_counts(train_counts: Dict[str, int], test_counts: Dict[str, int], class_names: List[str]) -> None:
    """Tworzy wykresy słupkowe liczby zdjęć dla każdej emocji."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].bar(class_names, train_counts.values())
    ax[0].set_title('Zbiór treningowy', fontsize=14)
    ax[0].set_xlabel('Emocje', fontsize=12)
    ax[0].set_ylabel('Liczba zdjęć', fontsize=12)
    ax[0].tick_params(axis='x', rotation=60)  # Pionowe etykiety osi X

    ax[1].bar(class_names, test_counts.values())
    ax[1].set_title('Zbiór testowy', fontsize=14)
    ax[1].set_xlabel('Emocje', fontsize=12)
    ax[1].set_ylabel('Liczba zdjęć', fontsize=12)
    ax[1].tick_params(axis='x', rotation=60)  # Pionowe etykiety osi X

    plt.suptitle('Rozkład klas emocji w zbiorze FER-2013', fontsize=16)

    plt.show()


def plot_dataset_stats():
    # Ścieżki do folderów głównych
    train_path = 'data/train'
    test_path = 'data/test'

    # Lista emocji
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotions_translated = ['Złość', 'Obrzydzenie', 'Strach', 'Szczęście', 'Smutek', 'Zaskoczenie', 'Neutralność']

    # Zliczanie zdjęć
    train_counts = count_emotions_images(train_path, emotions)
    test_counts = count_emotions_images(test_path, emotions)

    # Wyświetlanie wykresów
    plot_emotion_counts(train_counts, test_counts, emotions_translated)

    for emotion, emotion_translated in zip(emotions, emotions_translated):
        total = train_counts[emotion] + test_counts[emotion]
        print(f'Proporcje danych dla klasy {emotion_translated.lower()}: '
              f'{round(float(100 * train_counts[emotion] / total), 2)}%/'
              f'{round(float(100 * test_counts[emotion] / total), 2)}%')

    total = sum(train_counts.values()) + sum(test_counts.values())
    print(f'Treningowy: {sum(train_counts.values())}, Testowy: {sum(test_counts.values())}, Razem: {total}')
    print(f'Razem {round(float(100 * sum(train_counts.values()) / total), 2)}%/'
          f'{round(float(100 * sum(test_counts.values()) / total), 2)}%')


if __name__ == "__main__":
    plot_dataset_stats()
