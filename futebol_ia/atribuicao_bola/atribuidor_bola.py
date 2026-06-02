"""
Módulo Atribuidor de Posse de Bola.

Lógica:
  - Para cada frame, calcula a distância (pixels) da bola até os pés
    de cada jogador.
  - Se a menor distância for menor que o LIMIAR_POSSE, o jogador
    mais próximo é considerado em posse da bola.
  - Mantém histórico de posse por time para exibir porcentagem no vídeo.
"""

import cv2
import numpy as np

from futebol_ia.utils import obter_posicao_pe, calcular_distancia


class AtribuidorBola:
    """Atribui posse de bola ao jogador/time mais próximo em cada frame."""

    LIMIAR_POSSE = 70  # Distância máxima (pixels) para considerar posse

    def __init__(self):
        pass

    def atribuir_posse(
        self,
        rastreamento_jogadores: list[dict],
        rastreamento_bola: list[dict],
    ) -> list[int]:
        """
        Retorna uma lista com o time em posse para cada frame.
        Valores possíveis: 0 (sem posse), 1 (Time 1), 2 (Time 2).
        Também injeta 'tem_bola' nos dados dos jogadores.
        """
        posse_por_frame = []

        for indice_frame in range(len(rastreamento_jogadores)):
            info_bola = rastreamento_bola[indice_frame]
            info_jogadores = rastreamento_jogadores[indice_frame]

            # Se não há bola detectada neste frame
            if not info_bola or 1 not in info_bola:
                posse_por_frame.append(0)
                continue

            bbox_bola = info_bola[1]["bbox"]
            posicao_bola = (
                int((bbox_bola[0] + bbox_bola[2]) / 2),
                int(bbox_bola[3]),  # Base da bola (contato com chão)
            )

            menor_distancia = float("inf")
            jogador_mais_proximo = None

            for id_jogador, dados_jogador in info_jogadores.items():
                posicao_pe = obter_posicao_pe(dados_jogador["bbox"])
                distancia = calcular_distancia(posicao_bola, posicao_pe)

                if distancia < menor_distancia:
                    menor_distancia = distancia
                    jogador_mais_proximo = id_jogador

            # Verificar se está dentro do limiar
            if menor_distancia < self.LIMIAR_POSSE and jogador_mais_proximo is not None:
                dados_jogador = info_jogadores[jogador_mais_proximo]
                time_posse = dados_jogador.get("time", 0)
                dados_jogador["tem_bola"] = True
                posse_por_frame.append(time_posse)
            else:
                posse_por_frame.append(0)

        return posse_por_frame

    # ─────────────────────────────────────────────
    # Anotação visual de posse de bola
    # ─────────────────────────────────────────────

    def desenhar_posse_bola(
        self,
        frames: list[np.ndarray],
        posse_por_frame: list[int],
    ) -> list[np.ndarray]:
        """
        Desenha barra de porcentagem de posse acumulada (Time 1 vs Time 2)
        em um retângulo semi-transparente no canto superior do vídeo.
        """
        frames_anotados = []
        total_time1 = 0
        total_time2 = 0

        for indice_frame, frame in enumerate(frames):
            frame_copia = frame.copy()
            time_atual = posse_por_frame[indice_frame]

            if time_atual == 1:
                total_time1 += 1
            elif time_atual == 2:
                total_time2 += 1

            total = total_time1 + total_time2
            if total == 0:
                pct_time1 = 50.0
                pct_time2 = 50.0
            else:
                pct_time1 = (total_time1 / total) * 100
                pct_time2 = (total_time2 / total) * 100

            # Fundo semi-transparente
            overlay = frame_copia.copy()
            cv2.rectangle(overlay, (20, 10), (370, 90), (30, 30, 30), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, frame_copia, 0.3, 0, frame_copia)

            # Título
            cv2.putText(
                frame_copia,
                "POSSE DE BOLA",
                (110, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Barra de porcentagem
            largura_barra = 320
            x_inicio = 35
            y_barra = 50
            altura_barra = 20

            # Time 1 (esquerda)
            largura_time1 = int(largura_barra * pct_time1 / 100)
            cv2.rectangle(
                frame_copia,
                (x_inicio, y_barra),
                (x_inicio + largura_time1, y_barra + altura_barra),
                (0, 100, 255),  # Laranja
                cv2.FILLED,
            )

            # Time 2 (direita)
            cv2.rectangle(
                frame_copia,
                (x_inicio + largura_time1, y_barra),
                (x_inicio + largura_barra, y_barra + altura_barra),
                (255, 100, 0),  # Azul
                cv2.FILLED,
            )

            # Textos de porcentagem
            cv2.putText(
                frame_copia,
                f"Time 1: {pct_time1:.0f}%",
                (35, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 100, 255),
                2,
            )
            cv2.putText(
                frame_copia,
                f"Time 2: {pct_time2:.0f}%",
                (240, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 100, 0),
                2,
            )

            frames_anotados.append(frame_copia)

        return frames_anotados
