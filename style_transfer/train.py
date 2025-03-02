from pathlib import Path
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from networks import VGGDiscriminator, UNet
from dataset import StainDataset
from gen_loss import GenLoss

EPOCHS = 100
GEN_ITERS = 7

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = UNet().to(device)
    discriminator = VGGDiscriminator().to(device)

    # Загружаем данные
    he_dir = Path("data/dataset_stain_transformation/dataset_HE/tiles")
    ki_dir = Path("data/dataset_stain_transformation/dataset_Ki67/tiles")
    data = StainDataset(he_dir, ki_dir)

    # Генератор нужен, чтобы данные всегда делились одинаково
    number_generator = torch.Generator().manual_seed(42)
    train_data, val_data, _ = random_split(data, [0.6, 0.2, 0.2], number_generator)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=4)

    # Создаем оптимизаторы, которые будут обновлять параметры модели
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-5)

    # Нужно сделать лоссы для генератора и дискриминатора
    # Кажется, что для дискириминатора можно использовать просто MSE
    loss_gen = GenLoss()
    loss_dis = MSELoss()

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()

        total_gen_loss = 0.0
        total_dis_loss = 0.0

        for data in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
            he_data, ki_data = data[0].to(device), data[1].to(device)

            # Перед каждой эпохой нужно обнулять градиенты в оптимизаторах
            optimizer_dis.zero_grad()

            # Сначала прогоним дискриминатор на реальных картинках
            dis_pred_real = discriminator(ki_data)
            dis_loss_real = loss_dis(dis_pred_real, torch.ones_like(dis_pred_real))
            dis_loss_real.backward()  # Эта строчка прокидывает градиенты от лосса на все параметры

            # Теперь будем генерировать картинки для дискриминатора
            ki_fake = generator(he_data)
            dis_pred_fake = discriminator(ki_fake.detach())

            dis_loss_fake = loss_dis(dis_pred_fake, torch.zeros_like(dis_pred_fake))
            dis_loss_fake.backward()
            optimizer_dis.step()  # Выполняем обновление параметров оптимизаторов

            # Теперь прогон генератора с новым дискриминатором
            for _ in range(GEN_ITERS):
                optimizer_gen.zero_grad()
                dis_pred_fake = discriminator(ki_fake)
                gen_loss = loss_gen(ki_fake, ki_data, dis_pred_fake)
                gen_loss.backward()
                optimizer_gen.step()

                ki_fake = generator(he_data)  # Генерируем новую картинку с новым генератором

                total_gen_loss += gen_loss.item()

            # print(f"Fake: {dis_loss_fake}, real: {dis_loss_real}")

            # Считаем лоссы
            total_gen_loss = total_gen_loss / GEN_ITERS
            step_dis_loss = 0.5 * (dis_loss_real.item() + dis_loss_fake.item())
            total_dis_loss += step_dis_loss

        # Считаем средние лоссы за трейн
        avg_gen_loss = total_gen_loss / len(train_loader)
        avg_dis_loss = total_dis_loss / len(train_loader)

        # Переведем модели в режим валидации
        generator.eval()
        discriminator.eval()

        val_gen_loss = 0.0
        val_dis_loss = 0.0

        with torch.no_grad():  # Выключает подсчет градиентов, ускоряя валидацию
            for data in tqdm(val_loader):  # tqdm просто делает полоску загрузки
                he_data, ki_data = data[0].to(device), data[1].to(device)

                # Сначала прогоним дискриминатор на реальных картинках
                dis_pred_real = discriminator(ki_data)
                dis_loss_real = loss_dis(dis_pred_real, torch.ones_like(dis_pred_real))

                # Теперь будем генерировать картинки
                ki_fake = generator(he_data)
                dis_pred_fake = discriminator(ki_fake)

                gen_loss = loss_gen(ki_fake, ki_data, dis_pred_fake)
                dis_loss_fake = loss_dis(dis_pred_fake, torch.zeros_like(dis_pred_fake))

                val_gen_loss += gen_loss.item()
                val_dis_loss += 0.5 * (dis_loss_real.item() + dis_loss_fake.item())

        # Средний лосс по VAL
        val_gen_loss /= len(val_loader)
        val_dis_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Gen Loss: {avg_gen_loss:.4f}, Train Dis Loss: {avg_dis_loss:.4f} | "
              f"Val Gen Loss: {val_gen_loss:.4f}, Val Dis Loss: {val_dis_loss:.4f}")

    # Сохраним модельки
    torch.save(generator.state_dict(), "models/generator.pt")
    torch.save(discriminator.state_dict(), "models/discriminator.pt")
