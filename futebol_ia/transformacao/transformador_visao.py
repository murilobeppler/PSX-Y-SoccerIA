"""
Módulo Transformador de Visão — Transformação de Perspectiva.

Lógica:
  - Define um polígono (trapézio) na imagem do vídeo que corresponde
    a uma região conhecida do campo real.
  - Mapeia esse trapézio para um retângulo proporcional usando
    cv2.getPerspectiveTransform.
  - Converte posições dos jogadores de pixels para coordenadas
    em metros no campo real.

Dimensões padrão de um campo de futebol:
  - Comprimento: 105 metros
  - Largura: 68 metros
"""

import cv2
import numpy as np


class TransformadorVisao:
    """Transforma coordenadas de pixel para metros usando perspectiva."""

    # Dimensões reais do campo (metros)
    COMPRIMENTO_CAMPO = 105
    LARGURA_CAMPO = 68

    def __init__(self):
        # ── Vértices do trapézio no vídeo (pixels) ──
        # Devem ser ajustados para cada vídeo/câmera
        # Ordem: superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo
        self.vertices_pixel = np.float32([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915],
        ])

        # ── Retângulo de destino proporcional (metros → pixels escalados) ──
        # Escala para manter proporção no mapeamento
        self.largura_destino = int(self.LARGURA_CAMPO)
        self.comprimento_destino = int(self.COMPRIMENTO_CAMPO)

        self.vertices_destino = np.float32([
            [0, self.comprimento_destino],
            [0, 0],
            [self.largura_destino, 0],
            [self.largura_destino, self.comprimento_destino],
        ])

        # Matriz de transformação de perspectiva
        self.matriz_perspectiva = cv2.getPerspectiveTransform(
            self.vertices_pixel, self.vertices_destino
        )

    # ─────────────────────────────────────────────
    # Transformação de ponto individual
    # ─────────────────────────────────────────────

    def transformar_ponto(self, ponto: tuple) -> tuple | None:
        """
        Converte um ponto (x, y) em pixels para coordenadas (x_m, y_m)
        em metros no campo real.
        Retorna None se o ponto estiver fora do polígono de origem.
        """
        # Verificar se o ponto está dentro do polígono de origem
        ponto_array = np.array([ponto], dtype=np.float32)
        dentro = cv2.pointPolygonTest(
            self.vertices_pixel.astype(np.int32), ponto, False
        )

        if dentro < 0:
            return None

        # Aplicar transformação de perspectiva
        ponto_homogeneo = np.array([[[ponto[0], ponto[1]]]], dtype=np.float32)
        ponto_transformado = cv2.perspectiveTransform(ponto_homogeneo, self.matriz_perspectiva)

        x_metros = ponto_transformado[0][0][0]
        y_metros = ponto_transformado[0][0][1]

        return (x_metros, y_metros)

    # ─────────────────────────────────────────────
    # Aplicar transformação a todos os rastreamentos
    # ─────────────────────────────────────────────

    def adicionar_posicao_transformada(
        self, rastreamentos: list[dict]
    ) -> None:
        """
        Para cada objeto rastreado, converte 'posicao_ajustada' (pixels,
        já compensada pela câmera) em 'posicao_transformada' (metros).
        Caso não haja posicao_ajustada, tenta usar o centro da bbox.
        """
        for info_objetos in rastreamentos:
            for _, dados in info_objetos.items():
                posicao = dados.get("posicao_ajustada")

                if posicao is None:
                    # Fallback: usar centro inferior da bbox
                    bbox = dados.get("bbox")
                    if bbox is not None:
                        posicao = (
                            int((bbox[0] + bbox[2]) / 2),
                            int(bbox[3]),
                        )

                if posicao is not None:
                    posicao_metros = self.transformar_ponto(posicao)
                    dados["posicao_transformada"] = posicao_metros
                else:
                    dados["posicao_transformada"] = None
