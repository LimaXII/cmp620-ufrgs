import time, requests, os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# -------------------------------------------
# Mudar essas duas variáveis 
# URL da página de screenshots
url = 'https://steamcommunity.com/app/1462040/screenshots/'
folder_name = 'scrapping_images'
# -------------------------------------------

# Configurar o driver do Selenium
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(url)
time.sleep(5)  # Esperar a página carregar

# Rolagem para carregar mais imagens
previous_height = driver.execute_script("return document.body.scrollHeight")
images_loaded = 0

while images_loaded < 1000:
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    time.sleep(5)  # Esperar carregar as imagens

    # Verificar se novas imagens foram carregadas
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == previous_height:
        break  # Não há mais imagens para carregar
    previous_height = new_height

    # Contar o número de imagens carregadas
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    images_loaded = len(soup.find_all('img', class_='apphub_CardContentPreviewImage'))
    print(f'Imagens carregadas até agora: {images_loaded}')

# Analisar o conteúdo HTML carregado
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

# Encontrar todas as imagens de screenshots
screenshots = soup.find_all('img', class_='apphub_CardContentPreviewImage')
save_path = f"D:\\Dataset Jung\\{folder_name}"

# Corrigir o caminho para evitar problemas com barras invertidas
save_path = os.path.normpath(save_path)

# Criar a pasta se ela não existir
os.makedirs(save_path, exist_ok=True)

# Baixar e salvar cada imagem (até 1.000 ou menos se não houver tantas)
for i, screenshot in enumerate(screenshots[:1000]):
    img_url = screenshot['src']
    img_data = requests.get(img_url).content
    with open(os.path.join(save_path, f'screenshot_{i+1}.jpg'), 'wb') as handler:
        handler.write(img_data)

    print(f'Screenshot {i+1} baixada.')

print('Todas as screenshots foram baixadas.')
