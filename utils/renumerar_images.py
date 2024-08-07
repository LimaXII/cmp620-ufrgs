import os
import time

caminho_pasta = 'C:\\Users\\thiag\\Downloads\\Skyrim\\'

i = 0
for nome_arquivo in os.listdir(caminho_pasta):
    arquivo_antigo = caminho_pasta + nome_arquivo
    arquivo_novo = caminho_pasta + f'{i}.png'
    os.rename(arquivo_antigo, arquivo_novo)
    print(arquivo_novo)
    i += 1

i = 0
for nome_arquivo in os.listdir(caminho_pasta):
    arquivo_antigo = caminho_pasta + nome_arquivo
    arquivo_novo = caminho_pasta + f'skyrim({i}).png'
    os.rename(arquivo_antigo, arquivo_novo)
    print(arquivo_novo)
    i += 1