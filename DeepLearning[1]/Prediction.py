import cv2
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

# QUESTIONS = ['Меня ебали вчера хачи?', 'ЕБАЛ СЕХРАНА?', 'Дрочил на маму Глеба?', 'Отсосу бесплатно на',
#              'Занимался анальным дайвингом', 'Последователь Куколтизма?', 'Лизал пизду у Гахпе?',
#              'Жидко срал в штаны', 'Сосал за алмазы в майнкрафте', 'Засовывал хуй во включенный пылесос',
#              'Лизал киску Татьяны Юрьевны', 'Откусывал хуи у обрезанных', 'Блевал в рот потом глотал',
#              'Называл алевтину ебаной шлюхой', 'Нюхал трусы мамы Глеба', 'Пил говно у Башкиров',
#              'Дрочил пока ебуться свои родители на ', 'Щекотал яички лицом?', 'Скидывал дикпик бабушке Никиты',
#              'Поднимал вар', 'Думал что шевелить извилинами = хихикать'
#              ]

CATEGORIES = ['Dog', 'Cat']

image = r'C:\Users\sadi\PycharmProjects\DeepLearning_Cats_Dogs\PetImages\Test\doggo.jpg'

def prepare(filepath):
    img_size = 100
    img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    new_array = new_array / 255.0
    return new_array.reshape(-1, img_size, img_size, 1)


model = tf.keras.models.load_model(r"C:\Users\sadi\PycharmProjects\DeepLearning_Cats_Dogs\PetImages\FINALML.model")
prediction = model.predict([prepare(image)])
print(prediction[0][0])
print(CATEGORIES[int((prediction[0][0]))])
img = mpimg.imread(image)
imgplot = plt.imshow(img)
plt.title(CATEGORIES[int(round((prediction[0][0])))])
plt.show()

