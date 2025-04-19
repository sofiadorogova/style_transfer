import os
import random
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
    1) Фильтрует "слишком белые" изображения (mean >= white_threshold).
    2) Сохраняет отфильтрованные оригиналы в отдельные папки.
    """
    def __init__(self,
                 he_dir: str,
                 ki_dir: str,
                 he_filtered_dir: str,
                 ki_filtered_dir: str,
                 transform=None,
                 save_filtered=True,
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

                    he_img.save(self.he_filtered_dir / he_name)
                    ki_img.save(self.ki_filtered_dir / ki_name)

            self.he_files = valid_he_files
            self.ki_files = valid_ki_files
            self.length = len(self.he_files)
            print(f"After filtering (and saving) white images: {self.length} pairs remain.")

        else:
            # Режим, когда мы УЖЕ имеем сохранённые файлы.
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

        if self.transform is not None:
            he_img = self.transform(he_img_pil)
            ki67_img = self.transform(ki_img_pil)

        return he_img, ki67_img