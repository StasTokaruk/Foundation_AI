import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import os

GLOBAL_MODEL = None
SIMULATION_IMAGES = {}
VALID_DIGITS = np.arange(2, 10)

def prepare_and_train_cnn():
    # Завантаження даних
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Створення масок для фільтрації
    train_mask = np.isin(y_train, VALID_DIGITS)
    test_mask = np.isin(y_test, VALID_DIGITS)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Перетворення та нормалізація
    x_train = np.expand_dims(x_train.astype('float32') / 255.0, -1)
    x_test = np.expand_dims(x_test.astype('float32') / 255.0, -1)

    # Зсув:(Всього 8 класів)
    y_train_shifted = y_train - 2
    y_test_shifted = y_test - 2

    y_train_cat = to_categorical(y_train_shifted, num_classes=8)
    y_test_cat = to_categorical(y_test_shifted, num_classes=8)

    # Архітектура моделі
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(8, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Тренування моделі
    model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_data=(x_test, y_test_cat), verbose=1)

    # Завантаження
    model.save("speed_sign_cnn.h5")
    print(f"Модель успішно збережена як speed_sign_cnn.h5")

    return model


# Завантажує та зберігає зображення знаків (цифр 2-9) для симуляції
def load_sign_images():
    global SIMULATION_IMAGES
    (x_train, y_train), _ = mnist.load_data()
    x_train = np.expand_dims(x_train.astype('float32') / 255.0, -1)

    for digit in VALID_DIGITS:
        SIMULATION_IMAGES[digit] = x_train[y_train == digit]

# Повертає випадкове зображення цифри для симуляції камери
def get_sign_image(digit_value):
    if digit_value not in SIMULATION_IMAGES or not SIMULATION_IMAGES:
        load_sign_images()

    if digit_value not in SIMULATION_IMAGES:
        raise ValueError(f"Цифра {digit_value} не підтримується як знак.")

    # Вибираємо випадкове зображення цієї цифри
    images = SIMULATION_IMAGES[digit_value]
    return random.choice(images)

#Завантажує або повертає збережену модель CNN
def get_cnn_model():
    global GLOBAL_MODEL

    if GLOBAL_MODEL is None:
        if not os.path.exists("speed_sign_cnn.h5"):
            print("Модель CNN не знайдена. Запуск тренування")
            GLOBAL_MODEL = prepare_and_train_cnn()
        else:
            print(f"Завантаження збереженої моделі CNN з speed_sign_cnn.h5")
            GLOBAL_MODEL = load_model("speed_sign_cnn.h5")

    return GLOBAL_MODEL

# Приймає зображення знака, повертає визначену швидкість (20-90 км/год)
def recognize_sign(image_data):
    model = get_cnn_model()
    # Форматування вхідних даних
    input_data = np.expand_dims(image_data, 0)
    # Прогноз
    predictions = model.predict(input_data, verbose=0)
    # Отримання (0..7)
    predicted_class = np.argmax(predictions[0])
    # Зворотний зсув
    predicted_digit = predicted_class + 2

    return predicted_digit * 10