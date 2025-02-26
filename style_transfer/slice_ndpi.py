import os
import math
import argparse
import openslide


def slice_ndpi(
    ndpi_path: str,
    output_dir: str,
    tile_size: int = 512,
    level: int = 0,
    overlap: int = 0
):
    """
    Нарезка WSI-файла (NDPI) на тайлы.

    :param ndpi_path: Путь к NDPI-файлу.
    :param output_dir: Папка для сохранения нарезанных тайлов.
    :param tile_size: Размер одной плитки (в пикселях).
    :param level: Уровень разрешения (0 - максимальный).
    :param overlap: Перекрытие между тайлами (в пикселях).
    """
    # Открываем NDPI через OpenSlide
    slide = openslide.OpenSlide(ndpi_path)

    # Получаем размеры на нужном уровне
    width, height = slide.level_dimensions[level]

    # Создаём выходную директорию (если не существует)
    os.makedirs(output_dir, exist_ok=True)

    # Рассчитываем, сколько тайлов поместится по оси X и по оси Y
    step = tile_size - overlap  # шаг, учитывая перекрытие
    num_tiles_x = math.ceil(width / step)
    num_tiles_y = math.ceil(height / step)

    # Итерация по сетке
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Верхний левый угол текущего тайла
            x = i * step
            y = j * step

            # Читаем регион из слайда
            region = slide.read_region(
                (x, y),
                level,
                (tile_size, tile_size)
            ).convert("RGB")

            # Имя файла для сохранения
            tile_filename = f"tile_{i}_{j}.jpg"
            tile_path = os.path.join(output_dir, tile_filename)

            # Сохраняем в формате JPEG
            region.save(tile_path, quality=90)

    print(f"Готово! Нарезано {num_tiles_x * num_tiles_y} тайлов, сохранены в {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Скрипт для нарезки NDPI-файла на тайлы с помощью OpenSlide."
    )
    parser.add_argument("ndpi_path", help="Путь к файлу NDPI (Hamamatsu).")
    parser.add_argument("output_dir", help="Папка, куда будут сохранены тайлы.")
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Размер плитки (в пикселях), по умолчанию 512."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Уровень пирамиды (0 - максимальное разрешение)."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Перекрытие между тайлами (в пикселях), по умолчанию 0."
    )
    args = parser.parse_args()

    slice_ndpi(
        ndpi_path=args.ndpi_path,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        level=args.level,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()

#poetry run python style_transfer\slice_ndpi.py "D:\Downloads\HT16-5845 A4 Ki67.ndpi" "style_transfer/dataset"