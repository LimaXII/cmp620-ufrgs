import pyscreenshot as ImageGrab
import time

# Nome do arquivo de imagem de saída.
output_folder = "C:\\Users\\thiag\\Downloads\\CounterStrike2\\"
output_prefix = "counter_strike_2"
output_first_index = 0

for i in range(100):
    time.sleep(15)
    imagem = ImageGrab.grab()
    if output_first_index != 0:
        imagem.save(output_folder+output_prefix+"("+str(output_first_index)+").png", "png")
    else:
        imagem.save(output_folder+output_prefix+".png", "png")
    output_first_index += 1
    
