"""
Módulo Atribuidor de Times — Classificação por cor da camisa via K-Means.

Lógica:
  1. Recorta a metade superior da bounding box do jogador (crop da camisa).
  2. Aplica K-Means (k=2) no crop para separar cor da camisa vs. fundo residual.
  3. No primeiro frame, coleta as cores dominantes de todos os jogadores e
     aplica K-Means (k=2) novamente para definir os dois clusters de times.
  4. Para cada jogador subsequente, classifica-o no time mais próximo.
"""

import numpy as np
from sklearn.cluster import KMeans


class AtribuidorTimes:
    """Identifica e atribui jogadores a um dos dois times usando K-Means na cor da camisa."""

    def __init__(self):
        self.cores_times = {}          # {1: (B, G, R), 2: (B, G, R)}
        self.kmeans_times = None       # KMeans treinado com as 2 cores principais
        self._cache_times = {}         # {id_jogador: time} — cache para consistência

    # ─────────────────────────────────────────────
    # Extração de cor dominante da camisa
    # ─────────────────────────────────────────────

    def _obter_cor_camisa(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        Extrai a cor dominante da camisa de um jogador.
        Usa a metade superior da bbox para ignorar shorts/gramado,
        depois aplica K-Means (k=2) para separar camisa do fundo residual.
        """
        x1, y1, x2, y2 = map(int, bbox)
        altura_bbox = y2 - y1
        metade_superior = y1 + altura_bbox // 2

        # Crop da metade superior (região da camisa)
        crop_camisa = frame[y1:metade_superior, x1:x2]

        # Reshape para lista de pixels (N, 3)
        pixels = crop_camisa.reshape(-1, 3).astype(np.float32)

        # K-Means com k=2 para separar cor da camisa vs. fundo
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0)
        kmeans.fit(pixels)

        # O cluster com mais pixels nos cantos é o fundo;
        # o outro cluster é a camisa
        rotulos = kmeans.labels_
        imagem_rotulada = rotulos.reshape(crop_camisa.shape[0], crop_camisa.shape[1])

        # Verificar qual cluster domina os cantos (fundo/gramado)
        cantos = [
            imagem_rotulada[0, 0],
            imagem_rotulada[0, -1],
            imagem_rotulada[-1, 0],
            imagem_rotulada[-1, -1],
        ]
        cluster_fundo = max(set(cantos), key=cantos.count)
        cluster_camisa = 1 - cluster_fundo

        cor_camisa = kmeans.cluster_centers_[cluster_camisa]
        return cor_camisa

    # ─────────────────────────────────────────────
    # Definição dos dois times (primeiro frame)
    # ─────────────────────────────────────────────

    def definir_cores_times(
        self, frame: np.ndarray, jogadores_frame: dict
    ) -> None:
        """
        Analisa todos os jogadores do primeiro frame para definir
        as duas cores predominantes de times via K-Means (k=2).
        """
        cores_jogadores = []
        for _, dados in jogadores_frame.items():
            cor = self._obter_cor_camisa(frame, dados["bbox"])
            cores_jogadores.append(cor)

        if len(cores_jogadores) < 2:
            print("[AVISO] Menos de 2 jogadores detectados no primeiro frame.")
            return

        cores_array = np.array(cores_jogadores)

        # K-Means global para agrupar em 2 times
        self.kmeans_times = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        self.kmeans_times.fit(cores_array)

        # Armazenar cores representativas de cada time (BGR)
        self.cores_times[1] = tuple(map(int, self.kmeans_times.cluster_centers_[0]))
        self.cores_times[2] = tuple(map(int, self.kmeans_times.cluster_centers_[1]))

        print(f"[INFO] Cores dos times definidas: Time 1={self.cores_times[1]}, Time 2={self.cores_times[2]}")

    # ─────────────────────────────────────────────
    # Classificação individual de jogador
    # ─────────────────────────────────────────────

    def obter_time_jogador(
        self, frame: np.ndarray, bbox: list, id_jogador: int
    ) -> int:
        """
        Classifica um jogador em Time 1 ou Time 2.
        Usa cache por ID para manter consistência entre frames.
        """
        # Retornar do cache se já classificado
        if id_jogador in self._cache_times:
            return self._cache_times[id_jogador]

        cor_camisa = self._obter_cor_camisa(frame, bbox)

        # Predizer o cluster mais próximo
        time_idx = self.kmeans_times.predict(cor_camisa.reshape(1, -1))[0]
        time = time_idx + 1  # Converter 0-indexed para 1-indexed

        # Cachear resultado
        self._cache_times[id_jogador] = time
        return time
