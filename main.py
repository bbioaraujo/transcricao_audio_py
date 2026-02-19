from recursos import gerenciador_transcricao
import datetime
import sys

# "D:\\TEMP\\Juridico_reuniao_inicial.wav"

#arquivos = [sys.argv[1]]


arquivos = ["D:\\TEMP\\Erica_alinhameto_Seguros_2026.wav" ]

print(f'Iniciando transcrição do arquivo: {datetime.datetime.now()}')
for arquivo in arquivos:
    print(f'------------Iniciando transcrição do arquivo:{arquivo}-{datetime.datetime.now()}')

    gerenciador_transcricao.transcrever_com_falantes(arquivo, "large-v3")

    print(f'------------Fim da transcrição do arquivo:{arquivo}-{datetime.datetime.now()}')



print(f'Fim da transcrição do arquivo: -{datetime.datetime.now()}')



