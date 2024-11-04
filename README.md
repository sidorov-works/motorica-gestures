## Спринт 1

### Разные мысли

> Будущая модель, какой бы она не была, не должна выдавать слишком частую смену жеста: невозможно в течение ограниченного количества таймстемпов несколько раз сменить жест.

> Рассмотреть вариант добавления класса (или нескольких классов), соответствующего переходам между жестами.

> 30 измерений = 1 секунда реального времени

> Необходимо прояснить у заказчика: 
>
> - **Насколько частую смену жестов должна уметь отрабатывать модель?** Например, корректно распознать жест, по которому у нас будет окно хотя бы в $N=30$ измерений более реально, чем если смена жестов будет происходить чаще.
>
> - Какова общая допустимая задержка **системы принятия решения** на выполнение жеста? И какую часть от этой задержки в системе принятия решения занимают **получение данных** и **формирование команд**?
>
> ![Система принятия решений](img/decision-making-system.png)

> Возможно потребуется очистка данных (может быть, даже вручную): 
> - пилот мог отвлечься и опоздать с выполнением жеста
> - пилот мог перепутать жест и начать выполнять не то, что требовалось по команде

> Уровни датчиков на разных монтажах (отличаются пилоты, отличаются даты проведения измерений) могут принципиально различаться. Если предположить (пока что чисто эвристически), что датчики с высоким уровнем сигнала предпочтительнее для обучения модели, то ситуация складывается такая: в некоторых монтажах в ключевые (важные) признаки могут войти одни датчики, в то время как в других монтажах ключевым окажется уже другой набор датчиков.
>
> Идея: иметь в запасе несколько обученных моделей, отличающихся наборами признаков (датчиков, показания которых стали факторами модели). В момент калиброви, когда система принятия решений уже получает сигналы с датчиков и стало понятно, какие датчики сейчас дают высокий сигнал, выбрать наиболее подходящую модель на лету.