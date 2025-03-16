import os
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm 

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class StainDataset(Dataset):
    """
    Класс, возвращающий пары изображений: (HE_img, Ki67_img).
    Фильтруем (убираем) те пары, где HE или Ki67 картинка "слишком белая".
    """
    def __init__(self,
                 he_dir: str,
                 ki67_dir: str,
                 transform=None,
                 white_threshold=240):
        """
        :param white_threshold: если средняя яркость (0..255) выше этого порога —
                                картинка считается белой и исключается.
        """
        super().__init__()
        self.he_dir = he_dir
        self.ki67_dir = ki67_dir
        self.white_threshold = white_threshold

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        # Собираем списки файлов (HE, Ki67)
        all_he_files = sorted([
            f for f in os.listdir(self.he_dir)
            if f.lower().endswith('.png')
        ])
        all_ki_files = sorted([
            f for f in os.listdir(self.ki67_dir)
            if f.lower().endswith('.png')
        ])

        # Берём минимальную длину (чтобы не выйти за границы)
        length = min(len(all_he_files), len(all_ki_files))

        valid_he_files = []
        valid_ki_files = []

        for i in tqdm(range(length), desc="Filtering images"):
            he_name = all_he_files[i]
            ki_name = all_ki_files[i]
            he_path = os.path.join(self.he_dir, he_name)
            ki_path = os.path.join(self.ki67_dir, ki_name)

            # Загружаем PIL-изображения, проверяем среднюю яркость
            he_img = Image.open(he_path).convert('RGB')
            ki_img = Image.open(ki_path).convert('RGB')

            he_np = np.array(he_img)  # [H,W,3], 0..255
            ki_np = np.array(ki_img)

            # Считаем среднее значение пикселей (по всем каналам)
            he_mean = he_np.mean()
            ki_mean = ki_np.mean()

            # Если обе картинки не "слишком белые"
            if he_mean < self.white_threshold and ki_mean < self.white_threshold:
                valid_he_files.append(he_name)
                valid_ki_files.append(ki_name)
            # Иначе пропускаем эту пару (не добавляем)

        self.he_files = valid_he_files
        self.ki67_files = valid_ki_files
        self.length = len(self.he_files)  # после фильтрации

        print(f"After filtering white images: {self.length} pairs remain.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        he_path = os.path.join(self.he_dir, self.he_files[idx])
        ki67_path = os.path.join(self.ki67_dir, self.ki67_files[idx])

        he_img = Image.open(he_path).convert('RGB')
        ki67_img = Image.open(ki67_path).convert('RGB')

        if self.transform is not None:
            he_img = self.transform(he_img)
            ki67_img = self.transform(ki67_img)

        return he_img, ki67_img

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor()
    ])

    he_dir = Path("data/dataset_stain_transformation/dataset_HE/tiles")
    ki_dir = Path("data/dataset_stain_transformation/dataset_Ki67/tiles")

    dataset = StainDataset(
        he_dir=he_dir,
        ki67_dir=ki_dir,
        transform=transform,
        white_threshold=240  # верхний порог яркости для фильтра
    )
    

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"Dataset length: {len(dataset)}")

    for i, (he_batch, ki67_batch) in enumerate(dataloader):
        print("HE batch shape:", he_batch.shape)
        print("Ki67 batch shape:", ki67_batch.shape)
        # Можно проверить первые несколько
        if i > 5:
            break