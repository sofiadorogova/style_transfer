import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks import VGGDiscriminator, UNet
from dataset import StainDataset
from gen_loss import GenLoss

EPOCHS = 100

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = UNet().to(device)
    discriminator = VGGDiscriminator().to(device)

    # Загружаем данные
    he_dir = "D:\Downloads\dataset_stain_transformation\dataset_HE\tiles"
    ki_dir = "D:\Downloads\dataset_stain_transformation\dataset_Ki67\tiles"
    data = StainDataset("path/to/data/he", "path/to/data/ki")

    # Генератор нужен, чтобы данные всегда делились одинаково
    generator = torch.Generator().manual_seed(42)
    train_data, val_data, _ = random_split(data, [0.6, 0.2, 0.2], generator)

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
        num_steps = 0

        for data in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
            he_data, ki_data = data

            # Перед каждой эпохой нужно обнулять градиенты в оптимизаторах
            optimizer_gen.zero_grad()
            optimizer_dis.zero_grad()

            # Сначала прогоним дискриминатор на реальных картинках
            dis_pred_real = discriminator(ki_data)
            dis_loss_real = loss_dis(dis_pred_real, torch.ones_like(dis_pred_real))
            dis_loss_real.backward()  # Эта строчка прокидывает градиенты от лосса на все параметры

            # Теперь будем генерировать картинки
            ki_fake = generator(he_data)
            dis_pred_fake = discriminator(ki_fake)

            gen_loss = loss_gen(ki_fake, ki_data, dis_pred_fake)
            gen_loss.backward()

            dis_loss_fake = loss_dis(dis_pred_fake, torch.zeros_like(dis_pred_fake))
            dis_loss_fake.backward()

            # Выполняем обновление параметров оптимизаторов
            optimizer_gen.step()
            optimizer_dis.step()

            # Считаем лоссы
            total_gen_loss += gen_loss_value.item()
            step_dis_loss = 0.5*(dis_loss_real.item() + dis_loss_fake.item())
            total_dis_loss += step_dis_loss

            num_steps += 1

        # Считаем средние лоссы за трейн
        avg_gen_loss = total_gen_loss / num_steps
        avg_dis_loss = total_dis_loss / num_steps

        # Переведем модели в режим валидации
        generator.eval()
        discriminator.eval()

        val_gen_loss = 0.0
        val_dis_loss = 0.0
        val_steps = 0

        with torch.no_grad():  # Выключает подсчет градиентов, ускоряя валидацию
            for data in tqdm(val_loader):  # tqdm просто делает полоску загрузки
                he_data, ki_data = data

                # Сначала прогоним дискриминатор на реальных картинках
                dis_pred_real = discriminator(ki_data)
                dis_loss_real = loss_dis(dis_pred_real, torch.ones_like(dis_pred_real))

                # Теперь будем генерировать картинки
                ki_fake = generator(he_data)
                gen_loss = loss_gen(ki_fake, ki_data)

                dis_pred_fake = discriminator(ki_fake)
                dis_loss_fake = loss_dis(dis_pred_fake, torch.zeros_like(dis_pred_fake))

                val_gen_loss += gen_loss_value.item()
                val_dis_loss += 0.5*(dis_loss_real.item() + dis_loss_fake.item())
                val_steps += 1
    
        # Средний лосс по VAL
        val_gen_loss /= val_steps
        val_dis_loss /= val_steps

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Gen Loss: {avg_gen_loss:.4f}, Train Dis Loss: {avg_dis_loss:.4f} | "
              f"Val Gen Loss: {val_gen_loss:.4f}, Val Dis Loss: {val_dis_loss:.4f}")

    # Сохраним модельки
    torch.save(generator.state_dict(), "path/to/save/generator")
    torch.save(discriminator.state_dict(), "path/to/save/discriminator")
