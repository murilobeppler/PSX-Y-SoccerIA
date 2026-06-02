"""
Módulo de utilidades para o Sistema de Análise de Futebol.
Contém funções de I/O de vídeo e cálculos geométricos sobre bounding boxes.
"""

import cv2
import numpy as np
import sys


def ler_video(caminho_video: str) -> list[np.ndarray]:
    """
    Lê todos os frames de um arquivo de vídeo e retorna como lista de arrays NumPy (BGR).
    Encerra o programa caso o arquivo não possa ser aberto.
    """
    captura = cv2.VideoCapture(caminho_video)

    if not captura.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {caminho_video}")
        sys.exit(1)

    frames = []
    while True:
        sucesso, frame = captura.read()
        if not sucesso:
            break
        frames.append(frame)

    captura.release()
    print(f"[INFO] Vídeo carregado: {len(frames)} frames lidos de '{caminho_video}'")
    return frames


def salvar_video(frames_saida: list[np.ndarray], caminho_saida: str, fps: int = 24) -> None:
    """
    Salva uma lista de frames em um arquivo de vídeo MP4 (codec XVID).
    """
    if not frames_saida:
        print("[AVISO] Nenhum frame para salvar.")
        return

    altura, largura = frames_saida[0].shape[:2]
    codec = cv2.VideoWriter_fourcc(*"XVID")
    escritor = cv2.VideoWriter(caminho_saida, codec, fps, (largura, altura))

    for frame in frames_saida:
        escritor.write(frame)

    escritor.release()
    print(f"[INFO] Vídeo salvo em: '{caminho_saida}' ({len(frames_saida)} frames, {fps} FPS)")


def obter_centro_bbox(bbox: list | tuple | np.ndarray) -> tuple[int, int]:
    """
    Calcula o ponto central de uma bounding box [x1, y1, x2, y2].
    Retorna (cx, cy) como inteiros.
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def obter_largura_bbox(bbox: list | tuple | np.ndarray) -> int:
    """
    Retorna a largura de uma bounding box [x1, y1, x2, y2].
    """
    return int(bbox[2] - bbox[0])


def obter_posicao_pe(bbox: list | tuple | np.ndarray) -> tuple[int, int]:
    """
    Retorna a posição estimada dos pés do jogador:
    ponto inferior-central da bounding box [x1, y1, x2, y2].
    Essa é a referência para cálculos de distância até a bola.
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    return cx, int(y2)


def calcular_distancia(p1: tuple, p2: tuple) -> float:
    """
    Calcula a distância euclidiana entre dois pontos (x, y) em pixels.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def obter_fps_video(caminho_video: str) -> float:
    """
    Retorna o FPS (frames por segundo) do arquivo de vídeo.
    Usado para converter frames em tempo real (cálculo de velocidade).
    """
    captura = cv2.VideoCapture(caminho_video)
    fps = captura.get(cv2.CAP_PROP_FPS)
    captura.release()
    return fps
