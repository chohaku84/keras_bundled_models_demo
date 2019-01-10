# keras_bundled_models_demo
## Keras applications
https://keras.io/applications/#models-for-image-classification-with-weights-trained-on-imagenet

Xception
VGG16
VGG19
ResNet50
InceptionV3
InceptionResNetV2
MobileNet
DenseNet
NASNet
MobileNetV2

## Model .h5 file
.h5 weight files will be downloaded into "C:\Users\{user-name}\.keras\models"

## Notes
Note that some of these models need unique "preprocess_input" functions and input shape (target_size)

For example, for Xception model
```
from keras.applications.xception import preprocess_input

img_path = '../dataset/animal/cat.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
```
