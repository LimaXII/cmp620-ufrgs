import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Definindo o caminho das pastas
dataset_path = './dataset'

categories = [
    'Final Fantasy VII Remake', 'Resident Evil 4 Remake', 'Counter Strike 2',
    'Hollow Knight', 'Hogwarts Legacy', 'Baldurs Gate 3', 'Minecraft',
    'No Man\'s Sky', 'Persona 5 Royal', 'The Elder Scrolls V Skyrim'
]

# Parâmetros
img_height, img_width = 350, 350
batch_size = 32
epochs = 10
num_augmented_images_per_input = 4  # Número de imagens aumentadas por imagem de entrada

# Carregando e pré-processando as imagens
print("Pré-processando as imagens e fazendo reshape...")
def load_images(dataset_path, categories, img_height, img_width):
    images = []
    labels = []
    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images(dataset_path, categories, img_height, img_width)

# Normalizando as imagens
images = images / 255.0

print("Pré-processamento concluído. \n")

# Criando o gerador de dados com Data Augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.4,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Gerando imagens aumentadas
augmented_images = []
augmented_labels = []

for img, label in zip(images, labels):
    img = np.expand_dims(img, 0)  # Expande a dimensão para (1, img_height, img_width, 3)
    i = 0
    for batch in datagen.flow(img, batch_size=1):
        augmented_images.append(batch[0])
        augmented_labels.append(label)
        i += 1
        if i >= num_augmented_images_per_input:
            break

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Combinando as imagens originais e aumentadas
X = np.concatenate((images, augmented_images), axis=0)
y = np.concatenate((labels, augmented_labels), axis=0)

# Dividindo em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compilando o modelo
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Iniciando treino...\n")
# Treinando o modelo
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
print("Modelo treinado. \n")

# Salvando o modelo
model.save('./models/BIG-model-v3.keras')
print("Modelo salvo em ./models/BIG-model-v3.keras")

# Plotando a acurácia e a perda
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.legend()
plt.title('Acurácia')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda de Treino')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.legend()
plt.title('Perda')

plt.show()