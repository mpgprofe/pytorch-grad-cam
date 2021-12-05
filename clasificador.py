import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.__version__



import keras
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing import image
import urllib
import numpy as np
from keras.applications import VGG19
from keras import backend as K




#from keras.applications.vgg16 import VGG16
#from keras.applications import resnet50 


K.clear_session()

#model = tf.keras.applications.resnet.ResNet50(weights='imagenet') # se cambia en 4 sitios
#model = tf.keras.applications.vgg19.VGG19(weights='imagenet')
model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
#model.summary()

def show_images(*args, tags=None):
    n = len(args[0])
    plt.figure(figsize=(20, 4*len(args)))
    for i in range(n):
        for j in range(len(args)):
            data = args[j][i]
            img = data['image']
            ax = plt.subplot(len(args), n, i + 1 + j*n)
            cmap = None if 'cmap' not in data else data['cmap']
            
            if len(img.shape) == 3 and img.shape[2] == 1:
                ax.imshow(img.reshape(img.shape[:2]), cmap=cmap)
            else:
                ax.imshow(img, cmap=cmap)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            if 'tag' in data:
                tags = data['tag']
                n_lines = len(tags.split('\n'))
                ax.text(0.0,-0.1 * n_lines, tags, size=12, 
                        ha="left", transform=ax.transAxes)

    plt.show()

urls = ['./examples/f1.jpg', 
        './examples/f2.jpg',
        './examples/f3.jpg',
        './examples/g1.jpg',
        './examples/g2.jpg',
        './examples/g3.jpg',
        './examples/g4.jpg',
        './examples/g5.jpg',
        './examples/g6.jpg',
        './examples/g7.jpg',
        './examples/g8.jpg',
        './examples/g9.jpg',
        './examples/g10.jpg',
        './examples/g11.jpg',
        './examples/g12.jpg']

batch = []
raw_images = []
for i, url in enumerate(urls):
    file_name = url #f'img{i}.jpg'
    # Descargamos la imagen
    #urllib.request.urlretrieve(url, file_name)
    
    # La leemos con un tamaño 224 x 224 (según la entrada de la red)
    img = image.load_img(file_name, target_size=(299, 299)) #224, 224 menos para xception
    
    # Convertimos en un array de numpy
    x = image.img_to_array(img)
    
    raw_images.append(np.copy(x)/255)

    # y finalmente preprocesamos
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    
    # la agregamos a nuestro batch
    batch.append(x)

batch = np.array(batch)
preds = model.predict(batch)

clases = np.argmax(preds, axis=1)
#return the second highest probability:
def get_n_highest_prob(predicicion,n):
    return np.argsort(predicicion)[-n]

print(clases)
print(get_n_highest_prob(preds[0], 1))
'''
print("Clases: ", clases)
#para poder usar los mapas de características necesito el número de la clase
#visualizar el fichero map_clsloc.txt para entenderlo mejor
import csv
with open('map_clsloc.txt', 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    map_clsloc = {rows[0]:rows[1] for rows in reader}
'''    




texts = ['\n'.join([f"{tupla[1]}: {int(tupla[2]*100) }% nºClase:{get_n_highest_prob(preds[i], j+1)} " for j, tupla in  enumerate(dpred)])
                    for i, dpred in enumerate(tf.keras.applications.inception_v3.decode_predictions(preds, top=5))]


dpred = tf.keras.applications.inception_v3.decode_predictions(preds, top=3)
for i , class_name in enumerate(enumerate(dpred)):
    print(i, class_name)

#0:3
#3:9
#9:15
images = [{'image': im, 'tag': tag} for im, tag in zip(raw_images[0:3], texts[0:3])]
show_images(images)   



