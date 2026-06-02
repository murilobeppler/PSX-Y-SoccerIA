"""
Módulo Estimador de Movimento de Câmera — Optical Flow (Lucas-Kanade).

Lógica:
  - Usa features das extremidades do frame (fundo estático: arquibancadas, placas)
    para estimar o deslocamento (pan) da câmera entre frames consecutivos.
  - Esse deslocamento é subtraído da posição dos jogadores no rastreador,
    isolando o movimento real dos jogadores em relação ao campo.
  - Região de interesse: faixa de 20px nas bordas laterais (esquerda/direita),
    onde o fundo é estático e não contém jogadores.
"""

import cv2
import numpy as np
import pickle
import os


class EstimadorMovimentoCamera:
    """Estima o deslocamento de câmera (pan horizontal/vertical) via Optical Flow."""

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames

        # Parâmetros do Lucas-Kanade (Optical Flow esparso)
        self.parametros_lk = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Parâmetros para detecção de features (Shi-Tomasi)
        self.parametros_features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=None,
        )

    # ─────────────────────────────────────────────
    # Máscara de bordas (fundo estático)
    # ─────────────────────────────────────────────

    def _criar_mascara_bordas(self, frame: np.ndarray) -> np.ndarray:
        """
        Cria máscara que seleciona apenas as faixas laterais do frame (20px).
        Essas regiões contêm elementos estáticos (arquibancadas, placas)
        ideais para medir o deslocamento da câmera.
        """
        altura, largura = frame.shape[:2]
        mascara = np.zeros((altura, largura), dtype=np.uint8)

        largura_faixa = 20
        mascara[:, :largura_faixa] = 255          # Faixa esquerda
        mascara[:, largura - largura_faixa:] = 255  # Faixa direita

        return mascara

    # ─────────────────────────────────────────────
    # Cálculo do Optical Flow
    # ─────────────────────────────────────────────

    def obter_movimento_camera(
        self,
        ler_do_cache: bool = False,
        caminho_cache: str = None,
    ) -> list[np.ndarray]:
        """
        Calcula o deslocamento da câmera para cada frame.
        Retorna lista de arrays [dx, dy] representando o pan em pixels.
        O primeiro frame sempre tem deslocamento [0, 0].
        """
        # Tentar carregar do cache
        if ler_do_cache and caminho_cache and os.path.exists(caminho_cache):
            with open(caminho_cache, "rb") as f:
                print(f"[INFO] Movimento de câmera carregado do cache: '{caminho_cache}'")
                return pickle.load(f)

        movimento_camera = [np.array([0.0, 0.0])]  # Frame 0 = referência

        frame_anterior_cinza = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        mascara = self._criar_mascara_bordas(self.frames[0])

        for i in range(1, len(self.frames)):
            frame_atual_cinza = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)

            # Detectar features no frame anterior (apenas nas bordas)
            self.parametros_features["mask"] = mascara
            features_anteriores = cv2.goodFeaturesToTrack(
                frame_anterior_cinza, **self.parametros_features
            )

            if features_anteriores is None or len(features_anteriores) == 0:
                movimento_camera.append(np.array([0.0, 0.0]))
                frame_anterior_cinza = frame_atual_cinza
                continue

            # Calcular Optical Flow (Lucas-Kanade)
            features_novas, status, _ = cv2.calcOpticalFlowPyrLK(
                frame_anterior_cinza,
                frame_atual_cinza,
                features_anteriores,
                None,
                **self.parametros_lk,
            )

            # Filtrar apenas matches válidos
            validos = status.flatten() == 1
            pontos_anteriores = features_anteriores[validos]
            pontos_novos = features_novas[validos]

            if len(pontos_anteriores) == 0:
                movimento_camera.append(np.array([0.0, 0.0]))
                frame_anterior_cinza = frame_atual_cinza
                continue

            # Deslocamento médio = mediana para robustez contra outliers
            deslocamento = np.median(pontos_novos - pontos_anteriores, axis=0).flatten()
            movimento_camera.append(deslocamento)

            frame_anterior_cinza = frame_atual_cinza

        # Salvar cache
        if caminho_cache:
            os.makedirs(os.path.dirname(caminho_cache), exist_ok=True)
            with open(caminho_cache, "wb") as f:
                pickle.dump(movimento_camera, f)
            print(f"[INFO] Movimento de câmera salvo no cache: '{caminho_cache}'")

        print(f"[INFO] Movimento de câmera estimado para {len(movimento_camera)} frames.")
        return movimento_camera
