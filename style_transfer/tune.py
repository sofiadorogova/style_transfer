"""
tune_cycleGAN.py
Ищет гиперпараметры Extra-Gradient CycleGAN и сохраняет лучший конфиг.
"""

from pathlib import Path
import yaml
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from networks import UNet, VGGDiscriminator
from dataset  import StainDataset
from train_cycleGAN+EG import CycleGANTrainer 

# 1.  функция, которую будет запускать Ray Tune

def train_tune(config):
    """Одно испытание (trial) Ray Tune."""
    # — модели —
    G_XtoY, F_YtoX = UNet(), UNet()
    D_X, D_Y       = VGGDiscriminator(), VGGDiscriminator()

    # — датасет —
    ds = StainDataset(
        he_dir="data/dataset_HE/tiles",
        ki_dir="data/dataset_Ki67/tiles",
        he_filtered_dir="data/HE_filtered",
        ki_filtered_dir="data/Ki67_filtered",
        save_filtered=False,
    )

    trainer = CycleGANTrainer(
        G_XtoY, F_YtoX, D_X, D_Y, ds,
        batch_size       = config["batch_size"],
        lr_g             = config["lr_g"],
        lr_d             = config["lr_d"],
        weight_decay     = config["weight_decay"],
        lambda_cycle     = 10.0,
        grad_clip_value  = config["grad_clip"],
        epochs           = config["epochs"],   # обычно < 20
        save_every       = 1000,               # не пишем чекпойнты
    )

    best_val_g = float("inf")
    for epoch in range(1, trainer.epochs + 1):
        trainer.train_epoch()
        val_g, *_ = trainer.validate()
        tune.report(val_g=val_g)               #метрика для Tune

        # досрочный выход — если нет улучшения 3 эпохи
        if val_g < best_val_g:
            best_val_g = val_g
            bad = 0
        else:
            bad += 1
            if bad >= 3:
                break

# 2.  запускаем поиск

def run_hpo(save_path="configs/eg_best.yaml",
            num_samples=25,
            max_epochs=20):
    """Запуск Ray Tune и сохранение лучшего конфига в YAML."""
    # поисковое пространство
    space = {
        "lr_g":        tune.loguniform(1e-5, 5e-4),
        "lr_d":        tune.loguniform(1e-5, 5e-4),
        "weight_decay":tune.choice([0.0, 1e-5, 1e-4]),
        "grad_clip":   tune.choice([None, 5.0, 10.0]),
        "batch_size":  tune.choice([4, 8]),
        "epochs":      max_epochs,
    }

    scheduler = ASHAScheduler(
        metric="val_g",
        mode="min",
        max_t=max_epochs,
        grace_period=3,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(train_tune, {"cpu": 8, "gpu": 1}),
        param_space=space,
        tune_config=tune.TuneConfig(
            metric="val_g",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=ray.air.RunConfig(name="CycleGAN‑EG‑search",
                                     local_dir="ray_results"),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="val_g", mode="min")
    best_cfg = best.config
    print("Лучший конфиг:", best_cfg)

    # сохраняем в YAML
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.safe_dump({
            "optimizer": {
                "name":          "EG",
                "lr_g":          float(best_cfg["lr_g"]),
                "lr_d":          float(best_cfg["lr_d"]),
                "weight_decay":  float(best_cfg["weight_decay"]),
                "grad_clip":     best_cfg["grad_clip"],
            },
            "trainer": {
                "batch_size":    int(best_cfg["batch_size"]),
                "lambda_cycle":  10.0,
                "epochs":        100,        # полноценное финальное обучение
                "save_every":    10,
                "gpu_id":        0,
            },
            "dataset": {
                "he_dir":        "data/dataset_HE/tiles",
                "ki_dir":        "data/dataset_Ki67/tiles",
                "he_filtered_dir":"data/HE_filtered",
                "ki_filtered_dir":"data/Ki67_filtered",
                "save_filtered": False,
            }
        }, f)
    print(f"Best config сохранён в {save_path}")


if __name__ == "__main__":
    ray.init()
    run_hpo()

#Запуск: python tune_cycleGAN.py            # сохранит configs/eg_best.yaml