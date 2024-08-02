from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

# Definindo o caminho das pastas e categorias
categories = ['Final Fantasy VII Remake', 'Resident Evil 4 Remake', 'Counter Strike 2', 'Hollow Knight', 'Hogwarts Legacy']

# Parâmetros
img_height, img_width = 128, 128

# Carregando o modelo salvo
model = load_model('image_classification_model.h5')

# Função para carregar e pré-processar uma nova imagem
def preprocess_image(img_path, img_height, img_width):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão para o batch
    img_array = img_array / 255.0  # Normaliza a imagem
    return img_array

# Caminho da nova imagem
new_image_path = 'caminho/para/nova/imagem.png'

# Pré-processando a nova imagem
new_image = preprocess_image(new_image_path, img_height, img_width)

# Fazendo a previsão
prediction = model.predict(new_image)
predicted_label = np.argmax(prediction, axis=1)

# Mapeando o índice da previsão para a categoria correspondente
predicted_category = categories[predicted_label[0]]
print(f'A imagem foi classificada como: {predicted_category}')