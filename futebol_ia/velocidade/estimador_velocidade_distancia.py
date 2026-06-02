"""
Módulo Estimador de Velocidade e Distância.

Lógica:
  - Agrupa a movimentação a cada janela de 5 frames para suavizar ruídos.
  - Usa 'posicao_transformada' (metros) para calcular a distância percorrida.
  - Calcula o tempo da janela (frames / fps) e deriva a velocidade em km/h.
  - Injeta 'velocidade' e 'distancia' nos dados de rastreamento.
  - Desenha velocidade e distância no vídeo com overlay visual.
"""

import cv2
import numpy as np


class EstimadorVelocidadeDistancia:
    """Calcula e exibe velocidade (km/h) e distância (m) dos jogadores."""

    JANELA_FRAMES = 5  # Suavização: agrupar a cada N frames

    def __init__(self, fps: float):
        self.fps = fps

    # ─────────────────────────────────────────────
    # Cálculo de velocidade e distância
    # ─────────────────────────────────────────────

    def adicionar_velocidade_e_distancia(
        self, rastreamentos: list[dict]
    ) -> None:
        """
        Para cada objeto, calcula distância acumulada (metros) e velocidade
        instantânea (km/h) usando janelas de JANELA_FRAMES frames.
        Injeta 'velocidade' e 'distancia' nos dados de cada frame.
        """
        total_frames = len(rastreamentos)

        for indice_frame in range(0, total_frames, self.JANELA_FRAMES):
            ultimo_frame_janela = min(indice_frame + self.JANELA_FRAMES, total_frames - 1)

            for id_obj in rastreamentos[indice_frame].keys():
                # Verificar se o objeto existe no último frame da janela
                if id_obj not in rastreamentos[ultimo_frame_janela]:
                    continue

                pos_inicio = rastreamentos[indice_frame][id_obj].get("posicao_transformada")
                pos_fim = rastreamentos[ultimo_frame_janela][id_obj].get("posicao_transformada")

                if pos_inicio is None or pos_fim is None:
                    # Sem dados de posição transformada; zerar métricas
                    for j in range(indice_frame, ultimo_frame_janela + 1):
                        if id_obj in rastreamentos[j]:
                            rastreamentos[j][id_obj]["velocidade"] = 0.0
                            rastreamentos[j][id_obj]["distancia"] = 0.0
                    continue

                # Distância em metros
                distancia_metros = np.sqrt(
                    (pos_fim[0] - pos_inicio[0]) ** 2
                    + (pos_fim[1] - pos_inicio[1]) ** 2
                )

                # Tempo da janela em segundos
                num_frames_janela = ultimo_frame_janela - indice_frame
                if num_frames_janela == 0:
                    num_frames_janela = 1
                tempo_segundos = num_frames_janela / self.fps

                # Velocidade em km/h
                velocidade_ms = distancia_metros / tempo_segundos
                velocidade_kmh = velocidade_ms * 3.6

                # Propagar valores para todos os frames da janela
                for j in range(indice_frame, ultimo_frame_janela + 1):
                    if id_obj in rastreamentos[j]:
                        rastreamentos[j][id_obj]["velocidade"] = velocidade_kmh
                        rastreamentos[j][id_obj]["distancia"] = distancia_metros

    # ─────────────────────────────────────────────
    # Anotação visual no vídeo
    # ─────────────────────────────────────────────

    def desenhar_velocidade_distancia(
        self,
        frames: list[np.ndarray],
        rastreamentos: list[dict],
    ) -> list[np.ndarray]:
        """
        Desenha velocidade (km/h) e distância (m) abaixo de cada jogador/árbitro.
        """
        for indice_frame, frame in enumerate(frames):
            info_objetos = rastreamentos[indice_frame]

            for id_obj, dados in info_objetos.items():
                velocidade = dados.get("velocidade")
                distancia = dados.get("distancia")

                if velocidade is None or distancia is None:
                    continue

                # Posição: abaixo do retângulo de ID (base da bbox + offset)
                bbox = dados.get("bbox")
                if bbox is None:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                cx = (x1 + x2) // 2

                # Fundo semi-transparente para legibilidade
                x_fundo = cx - 45
                y_fundo = int(y2) + 30
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (x_fundo, y_fundo),
                    (x_fundo + 90, y_fundo + 32),
                    (20, 20, 20),
                    cv2.FILLED,
                )
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Texto: velocidade
                texto_vel = f"{velocidade:.1f} km/h"
                cv2.putText(
                    frame,
                    texto_vel,
                    (x_fundo + 5, y_fundo + 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1,
                )

                # Texto: distância
                texto_dist = f"{distancia:.1f} m"
                cv2.putText(
                    frame,
                    texto_dist,
                    (x_fundo + 5, y_fundo + 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 200, 200),
                    1,
                )

        return frames
