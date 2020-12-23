import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CATEGORIES = ['Dog', 'Cat']

image = r'PATH'

def prepare(filepath):
    img_size = 100
    img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    new_array = new_array / 255.0
    return new_array.reshape(-1, img_size, img_size, 1)


model = tf.keras.models.load_model(r"PATH\FINALML.model")
prediction = model.predict([prepare(image)])
print(prediction[0][0])
print(CATEGORIES[int((prediction[0][0]))])
img = mpimg.imread(image)
imgplot = plt.imshow(img)
plt.title(CATEGORIES[int(round((prediction[0][0])))])
plt.show()

