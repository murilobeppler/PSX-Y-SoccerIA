"""
Módulo Rastreador — Detecção (YOLOv8) + Rastreamento (ByteTrack).

Responsabilidades:
  - Detectar Bola (0), Goleiro (1), Jogador (2), Árbitro (3) via YOLO.
  - Converter goleiros para a classe Jogador antes do rastreamento.
  - Rastrear objetos com ByteTrack (IDs consistentes).
  - Interpolar posições da bola em frames perdidos (motion blur).
  - Desenhar anotações visuais (elipses, triângulos, IDs).
  - Cachear/carregar rastreamentos em pickle para agilizar o desenvolvimento.
"""

import pickle
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

from futebol_ia.utils import obter_centro_bbox, obter_posicao_pe


class Rastreador:
    """Encapsula detecção YOLO e rastreamento ByteTrack."""

    # Mapeamento de classes do modelo treinado
    CLASSE_BOLA = 0
    CLASSE_GOLEIRO = 1
    CLASSE_JOGADOR = 2
    CLASSE_ARBITRO = 3

    def __init__(self, caminho_modelo: str):
        self.modelo = YOLO(caminho_modelo)
        self.rastreador_bytetrack = sv.ByteTrack()

    # ─────────────────────────────────────────────
    # Detecção + Rastreamento
    # ─────────────────────────────────────────────

    def detectar_frames(self, frames: list[np.ndarray]) -> list:
        """Executa detecção YOLO em lotes de 20 frames."""
        deteccoes = []
        tamanho_lote = 20

        for i in range(0, len(frames), tamanho_lote):
            lote = frames[i : i + tamanho_lote]
            resultados_lote = self.modelo.predict(lote, conf=0.1)
            deteccoes.extend(resultados_lote)

        return deteccoes

    def obter_rastreamentos_objetos(
        self,
        frames: list[np.ndarray],
        ler_do_cache: bool = False,
        caminho_cache: str = None,
    ) -> dict:
        """
        Pipeline completo: detecta → converte goleiros → rastreia.
        Retorna dicionário com listas indexadas por frame:
          {
            "jogadores": [{id: {"bbox": [x1,y1,x2,y2]}, ...}, ...],
            "arbitros":  [...],
            "bola":      [...],
          }
        """
        # Tentar carregar do cache
        if ler_do_cache and caminho_cache and os.path.exists(caminho_cache):
            with open(caminho_cache, "rb") as f:
                print(f"[INFO] Rastreamentos carregados do cache: '{caminho_cache}'")
                return pickle.load(f)

        deteccoes = self.detectar_frames(frames)

        rastreamento_jogadores = []
        rastreamento_arbitros = []
        rastreamento_bola = []

        for indice_frame, deteccao in enumerate(deteccoes):
            # Converter de nomes para IDs numéricos
            mapa_classes = deteccao.names   # {0: "ball", 1: "goalkeeper", ...}
            mapa_invertido = {v: k for k, v in mapa_classes.items()}

            # Converter para objeto supervision
            sv_deteccao = sv.Detections.from_ultralytics(deteccao)

            # ── Converter goleiros para jogadores antes do rastreamento ──
            for i, id_classe in enumerate(sv_deteccao.class_id):
                if id_classe == self.CLASSE_GOLEIRO:
                    sv_deteccao.class_id[i] = self.CLASSE_JOGADOR

            # Rastrear com ByteTrack
            sv_deteccao_rastreada = self.rastreador_bytetrack.update_with_detections(sv_deteccao)

            jogadores_frame = {}
            arbitros_frame = {}
            bola_frame = {}

            for det_info in sv_deteccao_rastreada:
                bbox = det_info[0].tolist()
                id_classe = det_info[3]
                id_rastreio = det_info[4]

                if id_classe == self.CLASSE_JOGADOR:
                    jogadores_frame[id_rastreio] = {"bbox": bbox}
                elif id_classe == self.CLASSE_ARBITRO:
                    arbitros_frame[id_rastreio] = {"bbox": bbox}

            # Bola não é rastreada por ByteTrack (é 1 só); pegar maior confiança
            for det_info in sv_deteccao:
                bbox = det_info[0].tolist()
                id_classe = det_info[3]

                if id_classe == self.CLASSE_BOLA:
                    bola_frame[1] = {"bbox": bbox}

            rastreamento_jogadores.append(jogadores_frame)
            rastreamento_arbitros.append(arbitros_frame)
            rastreamento_bola.append(bola_frame)

        rastreamentos = {
            "jogadores": rastreamento_jogadores,
            "arbitros": rastreamento_arbitros,
            "bola": rastreamento_bola,
        }

        # Salvar cache para reutilização
        if caminho_cache:
            os.makedirs(os.path.dirname(caminho_cache), exist_ok=True)
            with open(caminho_cache, "wb") as f:
                pickle.dump(rastreamentos, f)
            print(f"[INFO] Rastreamentos salvos no cache: '{caminho_cache}'")

        return rastreamentos

    # ─────────────────────────────────────────────
    # Interpolação da Bola
    # ─────────────────────────────────────────────

    def interpolar_posicoes_bola(self, rastreamento_bola: list[dict]) -> list[dict]:
        """
        Usa pandas para interpolar bounding boxes da bola em frames
        onde ela não foi detectada (ex: motion blur).
        """
        posicoes_bola = [x.get(1, {}).get("bbox", []) for x in rastreamento_bola]
        df_bola = pd.DataFrame(posicoes_bola, columns=["x1", "y1", "x2", "y2"])

        # Interpolar valores ausentes e preencher extremidades
        df_bola = df_bola.interpolate().bfill()

        rastreamento_bola_interpolado = [
            {1: {"bbox": [row["x1"], row["y1"], row["x2"], row["y2"]]}}
            for _, row in df_bola.iterrows()
        ]

        return rastreamento_bola_interpolado

    # ─────────────────────────────────────────────
    # Compensação de Câmera (chamado pelo principal)
    # ─────────────────────────────────────────────

    def adicionar_posicao_ajustada(
        self, rastreamentos: list[dict], movimento_camera: list[np.ndarray]
    ) -> None:
        """
        Subtrai o deslocamento da câmera da posição dos pés de cada objeto,
        armazenando em 'posicao_ajustada'.
        """
        for indice_frame, info_objetos in enumerate(rastreamentos):
            for id_obj, dados in info_objetos.items():
                posicao_pe = obter_posicao_pe(dados["bbox"])
                deslocamento = movimento_camera[indice_frame]

                posicao_ajustada = (
                    posicao_pe[0] - deslocamento[0],
                    posicao_pe[1] - deslocamento[1],
                )
                dados["posicao_ajustada"] = posicao_ajustada

    # ─────────────────────────────────────────────
    # Anotações Visuais (OpenCV)
    # ─────────────────────────────────────────────

    def desenhar_anotacoes(
        self, frames: list[np.ndarray], rastreamentos: dict
    ) -> list[np.ndarray]:
        """
        Desenha em cada frame:
          - Elipses nos pés dos jogadores/árbitros
          - Retângulo com ID de rastreamento
          - Triângulo invertido sobre a bola
        """
        frames_anotados = []

        for indice_frame, frame in enumerate(frames):
            frame_copia = frame.copy()

            jogadores = rastreamentos["jogadores"][indice_frame]
            arbitros = rastreamentos["arbitros"][indice_frame]
            bola = rastreamentos["bola"][indice_frame]

            # Jogadores
            for id_jogador, dados in jogadores.items():
                cor = dados.get("cor_time", (0, 0, 255))
                self._desenhar_elipse(frame_copia, dados["bbox"], cor, id_jogador)

            # Árbitros
            for id_arbitro, dados in arbitros.items():
                self._desenhar_elipse(frame_copia, dados["bbox"], (0, 255, 255), id_arbitro)

            # Bola
            for _, dados in bola.items():
                self._desenhar_triangulo(frame_copia, dados["bbox"], (0, 255, 0))

            frames_anotados.append(frame_copia)

        return frames_anotados

    def _desenhar_elipse(
        self,
        frame: np.ndarray,
        bbox: list,
        cor: tuple,
        id_rastreio: int = None,
    ) -> None:
        """Desenha elipse nos pés e retângulo com o ID acima do jogador."""
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2
        largura = x2 - x1

        # Elipse na base da bounding box (pés)
        centro_elipse = (cx, int(y2))
        eixos = (int(largura) // 2, int(0.35 * (largura // 2)))
        cv2.ellipse(
            frame,
            center=centro_elipse,
            axes=eixos,
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=cor,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        # Retângulo com ID de rastreamento
        if id_rastreio is not None:
            largura_ret = 40
            altura_ret = 20
            x1_ret = cx - largura_ret // 2
            y1_ret = int(y2) + 5
            x2_ret = x1_ret + largura_ret
            y2_ret = y1_ret + altura_ret

            cv2.rectangle(
                frame,
                (x1_ret, y1_ret),
                (x2_ret, y2_ret),
                cor,
                cv2.FILLED,
            )

            texto = str(id_rastreio)
            tamanho_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]
            x_texto = x1_ret + (largura_ret - tamanho_texto[0]) // 2
            y_texto = y1_ret + (altura_ret + tamanho_texto[1]) // 2

            cv2.putText(
                frame,
                texto,
                (x_texto, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                2,
            )

    def _desenhar_triangulo(
        self,
        frame: np.ndarray,
        bbox: list,
        cor: tuple,
    ) -> None:
        """Desenha triângulo invertido acima da bola para destaque visual."""
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2
        tamanho = 10

        # Triângulo invertido (ponta para baixo, sobre a bola)
        pontos = np.array([
            [cx, y1 - 2],                     # Ponta inferior (em cima da bola)
            [cx - tamanho, y1 - 2 - tamanho * 2],  # Canto esquerdo superior
            [cx + tamanho, y1 - 2 - tamanho * 2],  # Canto direito superior
        ])

        cv2.drawContours(frame, [pontos], 0, cor, cv2.FILLED)
        cv2.drawContours(frame, [pontos], 0, (0, 0, 0), 2)  # Contorno preto
