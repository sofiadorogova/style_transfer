import os
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm 

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def gray_world_correction(img: np.ndarray) -> np.ndarray:
    """
    Реализация самого простого варианта Gray World:
    Вычисляем среднее значение по каждому каналу и приводим каждый канал 
    к общему среднему.
    
    :param img: np.array формата (H, W, 3), dtype = uint8 (0..255)
    :return: скорректированный np.array того же размера/типа
    """

    mean_per_channel = img.mean(axis=(0, 1))  # [meanR, meanG, meanB]
    # Среднее среди каналов (целевое "серое")
    gray_mean = mean_per_channel.mean()

    # Избегаем деления на 0
    scale = np.where(mean_per_channel == 0, 1, mean_per_channel)
    gain = gray_mean / scale  # Множители для каждого канала

    # Применяем коррекцию
    # shape (H, W, 3), умножаем каждый канал на соответствующий gain
    corrected = img.astype(np.float32)
    for c in range(3):
        corrected[..., c] *= gain[c]

    # Обрезаем до [0..255] и переводим обратно в uint8
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected

class StainDataset(Dataset):
    """
    Класс, возвращающий пары изображений: (HE_img, Ki67_img).
    1) Фильтрует "слишком белые" изображения (mean >= white_threshold).
    2) Сохраняет отфильтрованные оригиналы в отдельные папки.
    3) Баланс белого (GrayWorld) делает "на лету" при __getitem__.
    """
    def __init__(self,
                 he_dir: str,
                 ki_dir: str,
                 he_filtered_dir: str,
                 ki_filtered_dir: str,
                 transform=None,
                 save_filtered=True,
                 train=True,
                 white_threshold=230):
        """
        :param white_threshold: если средняя яркость (0..255) выше этого порога —
                                картинка считается белой и исключается.
        """
        super().__init__()

        self.he_dir = Path(he_dir)
        self.ki_dir = Path(ki_dir)
        self.he_filtered_dir = Path(he_filtered_dir)
        self.ki_filtered_dir = Path(ki_filtered_dir)
        self.he_filtered_dir.mkdir(parents=True, exist_ok=True)
        self.ki_filtered_dir.mkdir(parents=True, exist_ok=True)

        self.white_threshold = white_threshold
        self.save_filtered = save_filtered
        self.train = train

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        if self.save_filtered:
            self.he_filtered_dir.mkdir(parents=True, exist_ok=True)
            self.ki_filtered_dir.mkdir(parents=True, exist_ok=True)

            # Собираем файлы из исходных папок
            all_he_files = sorted([
                f for f in os.listdir(self.he_dir)
                if f.lower().endswith('.png')
            ])
            all_ki_files = sorted([
                f for f in os.listdir(self.ki_dir)
                if f.lower().endswith('.png')
            ])
            length = min(len(all_he_files), len(all_ki_files))

            valid_he_files = []
            valid_ki_files = []

            for i in tqdm(range(length), desc="Filtering images"):
                he_name = all_he_files[i]
                ki_name = all_ki_files[i]
                he_path = self.he_dir / he_name
                ki_path = self.ki_dir / ki_name

                he_img = Image.open(he_path).convert('RGB')
                ki_img = Image.open(ki_path).convert('RGB')

                he_np = np.array(he_img)
                ki_np = np.array(ki_img)

                he_mean = he_np.mean()
                ki_mean = ki_np.mean()

                # Фильтруем
                if he_mean < self.white_threshold and ki_mean < self.white_threshold:
                    valid_he_files.append(he_name)
                    valid_ki_files.append(ki_name)
                    # Сохраняем файлы
                    he_img.save(self.he_filtered_dir / he_name)
                    ki_img.save(self.ki_filtered_dir / ki_name)

            self.he_files = valid_he_files
            self.ki_files = valid_ki_files
            self.length = len(self.he_files)
            print(f"After filtering (and saving) white images: {self.length} pairs remain.")

        else:
            # Режим, когда мы УЖЕ имеем сохранённые файлы.
            # Просто читаем их из he_filtered_dir/ki_filtered_dir
            all_he_files = sorted([
                f for f in os.listdir(self.he_filtered_dir)
                if f.lower().endswith('.png')
            ])
            all_ki_files = sorted([
                f for f in os.listdir(self.ki_filtered_dir)
                if f.lower().endswith('.png')
            ])
            length = min(len(all_he_files), len(all_ki_files))

            self.he_files = all_he_files[:length]
            self.ki_files = all_ki_files[:length]
            self.length = len(self.he_files)
            print(f"Found {self.length} pairs in filtered directories.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        he_path = os.path.join(self.he_filtered_dir, self.he_files[idx])
        ki67_path = os.path.join(self.ki_filtered_dir, self.ki_files[idx])

        he_img_pil = Image.open(he_path).convert('RGB')
        ki_img_pil = Image.open(ki67_path).convert('RGB')

        #Используем баланс белого
        if self.train:
            he_np = np.array(he_img_pil)
            ki_np = np.array(ki_img_pil)

            he_corrected = gray_world_correction(he_np)
            ki_corrected = gray_world_correction(ki_np)

            he_img_pil = Image.fromarray(he_corrected)
            ki_img_pil = Image.fromarray(ki_corrected)

        if self.transform is not None:
            he_img = self.transform(he_img_pil)
            ki67_img = self.transform(ki_img_pil)

        return he_img, ki67_img



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    he_dir = Path("data/dataset_HE/tiles")
    ki_dir = Path("data/dataset_Ki67/tiles")
    
    he_filtered_dir = Path("data/HE_filtered")
    ki_filtered_dir = Path("data/Ki67_filtered")

    dataset = StainDataset(
        he_dir=he_dir,
        ki_dir=ki_dir,
        he_filtered_dir=he_filtered_dir,
        ki_filtered_dir=ki_filtered_dir,
        transform=transform,
        save_filtered=True,
        white_threshold=230
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"Dataset length: {len(dataset)}")

    # Проверим первые несколько элементов
    for i, (he_batch, ki_batch) in enumerate(dataloader):
        print("HE batch shape:", he_batch.shape)
        print("Ki67 batch shape:", ki_batch.shape)
        if i > 5:
            break