from glob import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.fft import rfft, rfftfreq


def carregar_dados_audio(arquivos_audio: List[str]) -> List[np.ndarray]:
    """Carrega os dados de áudio a partir dos arquivos fornecidos."""
    return [wavfile.read(arquivo)[1] for arquivo in arquivos_audio]


def calcular_fft(
        dados_audio: List[np.ndarray], frequencia_amostragem: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Calcula a Transformada de Fourier e retorna as densidades espectrais e frequências correspondentes."""
    periodo = 1 / frequencia_amostragem
    transformadas = [rfft(dados) for dados in dados_audio]
    densidades_espectrais = [abs(fft_dados) for fft_dados in transformadas]
    frequencias = [rfftfreq(len(dados), periodo) for dados in dados_audio]

    return densidades_espectrais, frequencias


def filtrar_frequencias(densidade: np.ndarray, frequencias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filtra as frequências com base em uma densidade mínima de 10% do pico."""
    limite_minimo = 0.1 * np.max(densidade)
    indices_validos = densidade > limite_minimo
    return frequencias[indices_validos], densidade[indices_validos]


def verificar_desafinacao(
        frequencia_dominante: float, frequencias_harmonicas: List[float], tolerancia: float = 1.0
) -> str:
    """Verifica se a corda está desafinada com base na frequência dominante e nas harmônicas."""
    # Encontre a harmônica mais próxima da frequência dominante
    diferencas = [abs(frequencia_dominante - f) for f in frequencias_harmonicas]
    harmonica_mais_proxima = frequencias_harmonicas[np.argmin(diferencas)]
    diferenca = frequencia_dominante - harmonica_mais_proxima

    # Determinar condição de afinação
    if abs(diferenca) <= tolerancia:
        return "Afinada"
    elif diferenca > tolerancia:
        return "Afrouxe a corda"
    else:
        return "Aperte a corda"


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
    dados_audio = carregar_dados_audio(arquivos_audio)
    densidades_espectrais, frequencias = calcular_fft(dados_audio, frequencia_amostragem)

    resultado = {'Arquivo': [], 'Nota': [], 'Frequência Dominante': [], 'Status': []}
    for i, arquivo in enumerate(arquivos_audio):
        nota, freq_dominante = identificar_nota(densidades_espectrais[i], frequencias[i], notas)

        # Extraindo todas as harmônicas da nota identificada
        frequencias_harmonicas = notas.loc[notas['Nota'] == nota, ['1ª Harmônica (Hz)', '2ª Harmônica (Hz)']].values[0]

        # Verificar desafinação usando todas as harmônicas
        status = verificar_desafinacao(freq_dominante, frequencias_harmonicas)

        resultado['Arquivo'].append(arquivo)
        resultado['Nota'].append(nota)
        resultado['Frequência Dominante'].append(round(freq_dominante, 2))
        resultado['Status'].append(status)

    return pd.DataFrame(resultado)


def main():
    """Função principal para execução do programa."""

    # Carregar as notas e arquivos de áudio
    notas = pd.read_csv('notas.csv')
    arquivos_audio = glob('./Violão/*.wav')

    # Frequência de amostragem do áudio
    frequencia_de_amostragem = 16_000

    # Processar os áudios
    resultado = processar_audio(arquivos_audio, notas, frequencia_de_amostragem)

    # Exibir resultados e tempo total
    print(resultado)


if __name__ == '__main__':
    main()
