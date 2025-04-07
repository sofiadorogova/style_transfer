# Автоматическая генерация редких гистологических окрасок с помощью CycleGAN

Данный проект посвящён задаче автоматической генерации редких гистологических окрасок (таких как Masson’s Trichrome, PAS и др.) на основе уже имеющихся снимков, окрашенных гематоксилином и эозином (H&E). Мы используем современные методы машинного обучения, в частности CycleGAN, что позволяет избегать дополнительной химической обработки образцов, сокращая время и затраты лаборатории, а также повышая эффективность диагностических процедур.


## 1. Краткое описание проекта

- **Входные данные**: изображения окраски H&E. 
  
<p align="center">
  <img src="pictures\photo_2025-04-01_11-06-55.jpg" alt="Диаграмма" width="500">
<p align="center"> Пример входных данных (H&E)</figcaption>
</figure>


- **Выход**: изображения специальной (редкой) окраски, например Masson’s Trichrome, PAS и т. д.

<p align="center">
  <img src="pictures\photo_2025-03-01_21-23-31.jpg" width="500" alt="Пример выходных данных">
<p align="center"> <strong></strong>Пример выходных данных </figcaption>
</figure>

- **Цель**: научить модель преобразовывать изображения H&E в изображения редкой окраски, сохраняя морфологическую информацию и корректно передавая цветовые особенности.  

В результате предлагается система, способная значительно упростить процесс получения редких окрасок и повысить точность диагностирования за счёт дополнительной информации, доступной патологам.

## 2. Концептуальное описание CycleGAN

**CycleGAN** – это подход к переносу стиля между двумя доменами ($X$ и $Y$) без использования парных данных. Основная идея:
1. Используются две пары генераторов и дискриминаторов:  
   - Генераторы $G: X \to Y$ и $F: Y \to X$ .  
   - Дискриминаторы $D_Y$ (оценивает, действительно ли образец принадлежит домену $Y$) и $D_X$  (оценивает принадлежность домену $X$).
<p align="center">
  <img src="pictures\arcitecture.png" width="500" alt="Работа CycleGAN">
<p align="center"><strong></strong> Работа CycleGAN </figcaption>
</figure>

1. **Adversarial Loss** (противоположные цели генератора и дискриминатора) заставляет выход генератора быть «максимально похожим» на образцы целевого домена, чтобы дискриминатор не смог отличить сгенерированные изображения от реальных.

<p align="center">
  <img src="pictures\loss-GAN.png" width="300" alt="Loss-функция GAN">
<p align="center"><strong></strong> Loss-функция GAN</figcaption>
</figure>

1. **Cycle-Consistency Loss** вводится, чтобы при переводе из домена $X$ в $Y$, а потом обратно – в $X$ (через генератор $F$ ), результат оставался как можно ближе к исходному изображению.

<p align="center">
  <img src="pictures\cycle-loss.png" width="300" alt="CycleLoss">
<p align="center"><strong></strong> CycleLoss</figcaption>
</figure>



> **Важно!** Мы не используем Identity Loss в данной постановке (часто она добавляется, чтобы стабилизировать обучение и сохранить цветовую гамму, но здесь от неё решено отказаться).

Таким образом, модель одновременно оптимизирует два вида лоссов (adversarial и cycle-consistency), чтобы добиться реалистичных преобразований между доменами.


## 3. Проблемы в обучении CycleGAN

- **Нестабильность в обучении**: GAN-ы, в том числе и CycleGAN, зачастую страдают от нестабильной динамики обучения, когда дисбаланс в обучении генератора и дискриминатора приводит к «замиранию» или «колебанию» лоссов.  
- **Mode collapse**: генератор может начать «зацикливаться» на одном или нескольких шаблонах, избегая разнообразия.  
- **Сложность гиперпараметров**: необходимо аккуратно подбирать $\lambda$, learning rate, а также архитектуру и размер мини-батча, чтобы получить хороший результат.  
- **Проблема галлюцинаций**: иногда модель «выдумывает» детали, которых не существует, и это особенно критично в медицинских задачах, где неправильная информация может сбить врача.

## 4. Результаты обучения

Ниже представлены некоторые результаты.

### 4.1 Численные результаты (графики лоссов)

- **График лосса дискриминатора ( $D_x$ )**: демонстрирует, насколько дискриминатор (применительно к домену X) учится отличать реальные снимки H&E от сгенерированных.  
- **График лосса генератора ( $G_x$ )**: показывает качество генератора, который пытается «обмануть» дискриминатор, генерируя достоверные изображения редкой окраски.

<p align="center">
  <img src="pictures\loss_train.jpg" width="400" alt="Кривая обучения генератора">
<p align="center"><strong></strong> Кривая обучения генератора </figcaption>
</figure>


### 4.2 Визуальные результаты

1. **Пример «галлюцинаций» в передаче цвета**: модель может некорректно изменять цветовые оттенки, делая некоторые области слишком яркими или, наоборот, тусклыми.  

<p align="center">
  <img src="pictures\color_mistakes.png" width="500" alt="Галлюцинации в цвете">
<p align="center"><strong></strong> Пример галлюцинаций в цвете </figcaption>
</figure>


1. **Передача информации о Ki-67 положительных клетках**: важно проверить, что модель правильно учитывает детали в визуализации клеток, особенно маркеров деления, без подмены сути изображения.

<p align="center">
  <img src="pictures\pair_1_collage.png" width="500" alt="Галлюцинации">
<p align="center"><strong></strong> Пример ошибок в распознавании Ki-67 положительных клеток </figcaption>
</figure>



## 5. Реализация (архитектура модели)

- **Генераторы**: используем **U-Net**. Такая структура хорошо подходит для задач преобразования изображений, поскольку «пропускает» информацию через скипы, что помогает сохранять пространственную информацию на разных уровнях.  
- **Дискриминаторы**: используем **VGGnet** (сокращённые варианты на базе VGG). Выбор в пользу VGG-моделей обусловлен тем, что они глубже, чем простые PatchGAN-дискриминаторы, и лучше различают глобальные и локальные признаки на изображении.

## 6. Возможные направления развития

- **Оптимизация архитектуры как седловой задачи**: рассматривать задачу обучения GAN как решение седловой задачи оптимизации (min-max), применяя современные методы (например, консенсусные оптимизаторы, адаптивные алгоритмы с регулировкой обучения генератора и дискриминатора). Это может улучшить стабильность и сходимость.  
- **Дополнительные регуляризаторы**: возможно, использование методов вроде R1-регуляризации (Penalized Gradient) в дискриминаторе для снижения переобучения и mode collapse.  
- **Совмещение с вниманием (attention)**: для более корректного переноса структурных особенностей, особенно важных для медицинских изображений (сохранение клеточных границ, выявление тонких морфологических деталей).  
- **Подбор гиперпараметров**: анализ влияния ($\lambda$) и других коэффициентов на итоговую визуализацию, чтобы найти оптимальный баланс между достоверностью и сохранением структуры.


