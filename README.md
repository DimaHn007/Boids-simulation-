# Boids-simulation-
Це програма, яка моделює поведінку птахів у зграйному форматі за допомогою моделі Boids. Ця модель використовує три основні правила для моделювання поведінки птахів у зграї:

- Відокремлення (Separation): Кожен птах уникатиме зіткнень з іншими птахами, тримаючи мінімальну відстань від них.
- Узгодження (Alignment): Птахи будуть спрямовані у середньому в тому ж напрямку, що й їхні сусіди.
- Когезія (Cohesion): Птахи будуть рухатися в напрямку центру мас зграї, намагаючись залишатись разом.

У програмі використовуються бібліотеки numpy, matplotlib та scipy.spatial.distance для обчислення відстаней між птахами та обробки графіки. Клас Boids відповідає за саму модель та її параметри, такі як кількість птахів, їхні позиції та швидкості, а також правила моделювання їхньої поведінки.

![image](https://github.com/user-attachments/assets/af67d724-150a-419c-825f-65a33b42d5b1)

