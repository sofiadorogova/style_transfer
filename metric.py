import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from PIL import Image
import os
import torch.nn.functional as func

inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  # Удаляем последний слой
inception_model.eval()

# Преобразование изображений именно под модель inception_v3
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get(images, model, flag = 'fid' ,batch_size=64):
    ansewrs = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack(images[i:i+batch_size])
        with torch.no_grad():
            pred = model(batch)
            if (flag == 'is'):
                pred = func.softmax(pred, dim=1)
        ansewrs.append(pred.cpu().numpy())
    return np.concatenate(ansewrs, axis=0)

def fid(generated_images, real_images):

    # real images - изображения - ground truth
    # generated_images - изображения, сгенерированные ганом: эти вектора -
    # это именно тензоры, которые ты получаешь на выходе из модели

    real = get(real_images, inception_model)
    generated = get(generated_images, inception_model)

    # Вычисляем статистики
    mu_real = np.mean(real, axis=0)
    sigma_real = np.cov(real, rowvar=False)
    mu_generated = np.mean(generated, axis=0)
    sigma_generated = np.cov(generated, rowvar=False)

    # Вычисление метрики
    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real.dot(sigma_generated))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2*covmean)
    return fid

def inception_score(generated_images, splits = 10):
    generated = get(generated_images, inception_model, 'is')
    split_scores = []
    for i in range(splits):
        part = generated[i * (len(generated) // splits): (i + 1) * (len(generated) // splits)]
        pyx = part
        py = np.mean(pyx, axis=0)
        kl = pyx * (np.log(pyx) - np.log(py))
        kl = np.sum(kl, axis=1)
        split_scores.append(np.exp(np.mean(kl)))
    
    return np.mean(split_scores), np.std(split_scores)
    

def load_images_from_folder(folder_name):
    images = []
    folder = os.listdir(folder_name)
    
    for filename in folder:
        img_path = os.path.join(folder_name, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
        except:
            print(f"Не удалось загрузить изображение: {img_path}")
    return images

real_images = "/Users/anastasiia/6семестр/m1p/images" #папка с реальными изображениями
generated_images = "/Users/anastasiia/6семестр/m1p/images" #папка с полученными изображениями из сетки
    
real_images = load_images_from_folder(real_images)
generated_images = load_images_from_folder(generated_images)
    
#вычисление метрик

fid_score = fid(generated_images, real_images) #сравнивает 2 набора данных - чем меньше значение, тем лучше
in_score = inception_score(generated_images, 5) #оценивает качество и разнообразие полученных изображений
