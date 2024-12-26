from glob import glob
import scipy.io.wavfile as wavfile
import numpy as np
import time
import pandas as pd
from scipy.fft import rfft, rfftfreq
from typing import List, Tuple


def carregar_dados_audio(arquivos_audio: List[str]) -> List[np.ndarray]:
    """Carrega os dados de áudio a partir dos arquivos fornecidos."""
    return [wavfile.read(arquivo)[1] for arquivo in arquivos_audio]


def calcular_fft(
        dados_audio: List[np.ndarray], frequencia_amostragem: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    """Calcula a Transformada de Fourier e retorna as densidades espectrais e frequências correspondentes."""
    periodo = 1 / frequencia_amostragem
    transformadas = [rfft(dados) for dados in dados_audio]
    densidades_espectrais = [np.abs(fft_dados) for fft_dados in transformadas]
    frequencias = [rfftfreq(len(dados), periodo) for dados in dados_audio]

    return densidades_espectrais, frequencias


def filtrar_frequencias(densidade: np.ndarray, frequencias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filtra as frequências com base em uma densidade mínima de 10% do pico."""
    limite_minimo = 0.1 * np.max(densidade)
    indices_validos = densidade > limite_minimo
    return frequencias[indices_validos], densidade[indices_validos]


def identificar_nota(
    densidade_espectral: np.ndarray, frequencias: np.ndarray, notas: pd.DataFrame
) -> Tuple[str, float]:
    """Identifica a nota correspondente ao espectro de áudio fornecido."""
    frequencias_filtradas, espectro_filtrado = filtrar_frequencias(densidade_espectral, frequencias)

    # Frequência dominante
    frequencia_dominante = frequencias_filtradas[np.argmax(espectro_filtrado)]

    # Calcular diferenças em relação às harmônicas das notas
    harm1_diff = np.abs(notas['1ª Harmônica (Hz)'].values - frequencia_dominante)
    harm2_diff = np.abs(notas['2ª Harmônica (Hz)'].values - frequencia_dominante)
    diferencas = np.minimum(harm1_diff, harm2_diff)

    nota_mais_proxima = notas.iloc[np.argmin(diferencas)]

    return nota_mais_proxima['Nota'], float(frequencia_dominante)


def processar_audio(arquivos_audio: List[str], notas: pd.DataFrame, frequencia_amostragem: float) -> pd.DataFrame:
    """Processa todos os arquivos de áudio e retorna as notas identificadas e suas frequências dominantes."""
    # Carregar dados de áudio
    dados_audio = carregar_dados_audio(arquivos_audio)

    # Calcular FFTs e densidades espectrais
    densidades_espectrais, frequencias = calcular_fft(dados_audio, frequencia_amostragem)

    # Identificar notas
    resultado = {'Arquivo': [], 'Nota': [], 'Frequência Dominante': []}
    for i, arquivo in enumerate(arquivos_audio):
        nota, freq_dominante = identificar_nota(densidades_espectrais[i], frequencias[i], notas)
        resultado['Arquivo'].append(arquivo)
        resultado['Nota'].append(nota)
        resultado['Frequência Dominante'].append(round(freq_dominante, 2))

    return pd.DataFrame(resultado)


def main():
    """Função principal para execução do programa."""
    t0 = time.time()

    # Carregar as notas e arquivos de áudio
    notas = pd.read_csv('notas.csv')
    arquivos_audio = glob('./Violão/*.wav')

    # Frequência de amostragem do áudio
    frequencia_amostragem = 16_000

    # Processar os áudios
    resultado = processar_audio(arquivos_audio, notas, frequencia_amostragem)

    # Exibir resultados e tempo total
    print(resultado)
    print(f"Tempo total: {time.time() - t0:.2f} segundos")


if __name__ == '__main__':
    main()
