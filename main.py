from glob import glob
import matplotlib
import scipy.io.wavfile as wavfile
import numpy as np
import pandas as pd
from scipy.fftpack import fft


def identificar_nota(indice: int, notas):
    """Identifica a nota correspondente ao áudio especificado pelo índice."""
    limite_minimo = 0.1 * np.max(densidades_espectrais[indice])
    indices_validos = densidades_espectrais[indice] > limite_minimo

    frequencias_filtradas = frequencias[indice][indices_validos]

    # Verificar qual nota está mais próxima da frequência dominante
    frequencia_dominante = frequencias_filtradas[np.argmax(densidades_espectrais[indice][indices_validos])]
    diferencas = notas[['1ª Harmônica (Hz)', '2ª Harmônica (Hz)']].apply(
        lambda x: np.min(np.abs(x - frequencia_dominante)), axis=1
    )
    nota_mais_proxima = notas.iloc[np.argmin(diferencas)]

    return nota_mais_proxima['Nota'], frequencia_dominante


if __name__ == '__main__':
    matplotlib.use('TkAgg')  # Carregar metadados dos áudios
    notas = pd.read_csv('notas.csv')
    print(type(notas))
    arquivos_som = glob(r'./Violão/*.wav')

    if not arquivos_som:
        raise FileNotFoundError("Nenhum arquivo .wav encontrado no diretório './Violão/'.")

    frequencia_amostragem = 16_000
    periodo = 1 / frequencia_amostragem

    # Ler e processar dados de áudio
    dados_audio = [wavfile.read(arquivo)[1] for arquivo in arquivos_som]
    frequencias = [np.fft.fftfreq(len(dados), periodo) for dados in dados_audio]

    # Realizar FFT e calcular densidades espectrais
    transformadas_fft = [fft(dados) for dados in dados_audio]
    densidades_espectrais = [np.abs(fft_dados) for fft_dados in transformadas_fft]

    # Identificar e exibir nota correspondente a cada áudio
    for i, arquivo in enumerate(arquivos_som):
        nota, freq_dominante = identificar_nota(indice=i, notas=notas)
        print(f"Arquivo: {arquivo}, Nota: {nota}, Frequência Dominante: {freq_dominante:.2f} Hz")


