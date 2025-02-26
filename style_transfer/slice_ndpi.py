import json
import os
import sys
import openslide
from PIL import Image

def polygon_to_bbox(coords):
    """
    Преобразует массив координат (список точек)
    в bounding box: (x_min, y_min, width, height).

    coords: например,
      [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min]
      ]
    Возвращает (x_min, y_min, w, h).
    """
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def main(ndpi_path, geojson_path, output_dir):
    """
    :param ndpi_path: Путь к NDPI-файлу
    :param geojson_path: Путь к JSON-файлу (GeoJSON), который содержит полигоны
    :param output_dir: Папка, куда складывать результат
    """

    # 1. Загружаем JSON-данные
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # это список Feature

    # 2. Открываем NDPI через OpenSlide
    slide = openslide.OpenSlide(ndpi_path)

    # Создадим выходную папку
    os.makedirs(output_dir, exist_ok=True)

    # 3. Найдём Feature с objectType="annotation" — считаем, это "общая" область
    annotation_feature = None
    for feature in data:
        if feature.get('properties', {}).get('objectType') == 'annotation':
            annotation_feature = feature
            break

    if annotation_feature is None:
        print("[WARNING] Не найдена аннотация с objectType='annotation'. Пропускаем preview.")
    else:
        coords = annotation_feature['geometry']['coordinates'][0]
        x, y, w, h = polygon_to_bbox(coords)

        # -- Делаем превью области, чтобы не упасть по памяти. --
        # Способ A: Уменьшаем вручную через 'level' и пропорции
        # Берём некоторый downsample, например 8 (или 4, 16 — в зависимости от размера).
        downsample = 8
        
        # Узнаем, какой уровень в пирамиде ближе всего к downsample=8
        best_level = slide.get_best_level_for_downsample(downsample)
        # Сколько пикселей это будет (учитывая downsample)
        w_reduced = int(w / downsample)
        h_reduced = int(h / downsample)

        # Читаем регион на best_level
        preview_region = slide.read_region((x, y), best_level, (w_reduced, h_reduced)).convert("RGB")

        # Сохраняем превью
        preview_path = os.path.join(output_dir, "preview.jpg")
        preview_region.save(preview_path, "JPEG", quality=90)
        print(f"[INFO] Сохранено превью области: {preview_path}")


    # 4. Список всех фич, у которых "objectType" = "tile"
    tile_features = [
        f for f in data
        if f.get('properties', {}).get('objectType') == 'tile'
    ]
    print(f"[INFO] Найдено тайлов: {len(tile_features)}")

    # Берём первые 200
    tile_features = tile_features[:200]

    # Папка для тайлов
    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    for i, feature in enumerate(tile_features, start=1):
        coords = feature['geometry']['coordinates'][0]
        x, y, w, h = polygon_to_bbox(coords)

        # Считываем регион в максимальном уровне (0)
        region = slide.read_region((x, y), 0, (w, h)).convert("RGB")

        # Имя
        tile_name = feature['properties'].get('name', f"Tile_{i}")
        tile_filename = f"{i:03d}_{tile_name}.jpg"
        tile_path = os.path.join(tiles_dir, tile_filename)

        region.save(tile_path, "JPEG", quality=90)
        print(f"[INFO] Сохранён тайл: {tile_path}")

    print("[INFO] Готово!")


if __name__ == "__main__":
    # Пример использования:
    # python slice_from_json.py file.ndpi tiles.json output_folder
    if len(sys.argv) < 4:
        print("Использование: python slice_from_json.py <ndpi> <json> <output_dir>")
        sys.exit(1)

    ndpi_file = sys.argv[1]
    json_file = sys.argv[2]
    out_dir = sys.argv[3]

    main(ndpi_file, json_file, out_dir)
#poetry run python style_transfer\slice_ndpi.py "D:\Downloads\HT16-5845 A4 Ki67.ndpi" "D:\Downloads\HT16-5845 A4 Ki67.geojson" "style_transfer/dataset"
