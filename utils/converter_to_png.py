import os
from PIL import Image

def convert_images_to_png(folder_path, output_prefix):
    if not os.path.isdir(folder_path):
        print("O caminho especificado não é uma pasta.")
        return

    # Lista de extensões de arquivos de imagem suportados
    supported_extensions = ['.jpg', '.jpeg', '.jfif', '.bmp', '.gif', '.tiff', '.webp']

    files = [f for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in supported_extensions)]
    if not files:
        print("Nenhum arquivo de imagem suportado encontrado na pasta.")
        return

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        try:
            with Image.open(file_path) as img:
                output_file = os.path.join(folder_path, f"{output_prefix}({i}).png")
                img.save(output_file, 'PNG')
                print(f"{file} convertido para {output_file}")
            
            os.remove(file_path)
            print(f"{file} original excluído.")
        except Exception as e:
            print(f"Erro ao converter {file}: {e}")

# Especifique o caminho da pasta e o prefixo de saída aqui
folder_path = "D:\\Dataset Jung\\scrapping_images"
# Nome do arquivo de imagem de saída.
output_prefix = "ff7_remake"

convert_images_to_png(folder_path, output_prefix)