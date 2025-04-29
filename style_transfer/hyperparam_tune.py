import os
import optuna
import torch
import gc
from argparse import ArgumentParser, Namespace

from train import CycleGANTrainer
from metric import fid
import json

# Использовать доступные ресурсы GPU
os.environ.pop("CUDA_VVISIBLE_DEVICES", None)
torch.backends.cudnn.benchmark = True

# Константы
BATCH_SIZE = 1
TUNE_EPOCHS = 5
TUNE_SAVE_EVERY = TUNE_EPOCHS + 1

# Общие аргументы для тренировки
args_base = {
    'he_dir': "data/dataset_HE/tiles",
    'ki_dir': "data/dataset_Ki67/tiles",
    'he_filt': "data/HE_filtered",
    'ki_filt': "data/Ki67_filtered",
}

def objective(trial, opt_name):
    # Пробуем гиперпараметры
    lr_g = trial.suggest_loguniform("lr_g", 1e-6, 1e-3)
    lr_d = trial.suggest_loguniform("lr_d", 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-8, 1e-2)
    lambda_cycle = trial.suggest_uniform("lambda_cycle", 1.0, 20.0)
    lambda_id = trial.suggest_uniform("lambda_id", 0.0, 1.0)
    disc_steps = trial.suggest_int("disc_steps", 1, 5)
    label_smoothing = trial.suggest_uniform("label_smoothing", 0.0, 0.2)
    
    # Объединяем общие аргументы и настраиваемые параметры
    args = Namespace(**args_base, **{
        'optimizer_g': opt_name,
        'optimizer_d': opt_name,
        'batch_size': BATCH_SIZE,
        'lr_g': lr_g,
        'lr_d': lr_d,
        'weight_decay': weight_decay,
        'lambda_cycle': lambda_cycle,
        'lambda_id': lambda_id,
        'grad_clip': None,
        'disc_steps': disc_steps,
        'label_smoothing': label_smoothing,
        'epochs': TUNE_EPOCHS,
        'save_every': TUNE_SAVE_EVERY,
        'logdir': f"runs/tune/{opt_name}/{trial.number}",
        'resume': None
    })

    # Инит тренера и запуск короткого обучения
    trainer = CycleGANTrainer(args)
    trainer.run()

    # Валидация потерь
    val_g, val_dx, val_dy = trainer.validate()

    # Расчёт FID
    fake, real = [], []
    with torch.no_grad():
        for x, y in trainer.val_loader:
            x, y = x.to(trainer.device), y.to(trainer.device)
            fy = trainer.G(x)
            fake.append(fy.cpu().squeeze(0))
            real.append(y.cpu().squeeze(0))
    fid_score = fid(fake, real)

    # Целевое значение
    target = val_g + fid_score
    trial.set_user_attr("val_g_loss", val_g)
    trial.set_user_attr("fid", fid_score)

    # Освобождение памяти
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return target

def main():
    parser = ArgumentParser()
    parser.add_argument("--method",
                        choices=["adam", "ogda", "opt_adam", "eg"],
                        required=True,
                        help="Какой оптимизатор тюнить?")
    args_cli = parser.parse_args()
    method = args_cli.method

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, method), n_trials=30)

    best = study.best_params
    print("=== Лучшие параметры для:", method, "===")
    print(best)

    # Сохранение лучших параметров
    with open(f"best_params_{method}.json", "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()