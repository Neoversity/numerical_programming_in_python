
# Спектральна кластеризація аудіосигналів

Ця домашня робота демонструє використання перетворення Фур'є та спектральної кластеризації для класифікації аудіосигналів із набору даних ESC-50. Сигнали включають звуки з двох категорій: `dog` та `chirping_birds`. Спектрограми аудіосигналів створюються за допомогою перетворення Фур'є, далі виконується pooling для зменшення розмірності, а потім вони перетворюються у вектори для кластеризації.

## Робочий процес проєкту

1. **Підготовка даних:**
   - Аудіофайли фільтруються з набору даних ESC-50 для включення лише звуків з мітками `dog` та `chirping_birds`.
   
2. **Виділення ознак:**
   - Для кожного аудіосигналу застосовується перетворення Фур'є (STFT) для отримання спектрограми.

3. **Pooling:**
   - Зменшення розмірності спектрограм за допомогою методу max-pooling.

4. **Кластеризація:**
   - Використання методу спектральної кластеризації для поділу звуків на дві категорії.

5. **Аналіз результатів:**
   - Аналіз отриманих кластерів показав, що звуки різного походження (собаки та птахів) потрапили в різні кластери.
   
## Висновки:
Перетворення Фур'є дозволяє ефективно вилучати частотні ознаки зі звукових сигналів, що дає можливість успішно класифікувати та кластеризувати звуки різного походження.
