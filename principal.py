"""
Sistema de Análise de Futebol com IA/ML — Orquestrador Principal.

Pipeline completo:
  1. Detecção + Rastreamento (YOLO + ByteTrack)
  2. Interpolação da posição da bola
  3. Atribuição de times (K-Means na cor da camisa)
  4. Atribuição de posse de bola
  5. Compensação de câmera (Optical Flow)
  6. Transformação de perspectiva (pixels → metros)
  7. Cálculo de velocidade e distância
  8. Anotação visual e exportação do vídeo final
"""



from futebol_ia.utils import (
    ler_video,
    salvar_video,
    obter_fps_video,
)
from futebol_ia.rastreadores.rastreador import Rastreador
from futebol_ia.atribuicao_times.atribuidor_times import AtribuidorTimes
from futebol_ia.atribuicao_bola.atribuidor_bola import AtribuidorBola
from futebol_ia.camera.estimador_movimento_camera import EstimadorMovimentoCamera
from futebol_ia.transformacao.transformador_visao import TransformadorVisao
from futebol_ia.velocidade.estimador_velocidade_distancia import EstimadorVelocidadeDistancia


# ─────────────────────────────────────────────
# Configurações globais
# ─────────────────────────────────────────────
CAMINHO_VIDEO_ENTRADA = "videos/entrada.mp4"
CAMINHO_VIDEO_SAIDA   = "videos/saida.avi"
CAMINHO_MODELO_YOLO   = "modelos/melhor_modelo.pt"       # Peso treinado do YOLO
CAMINHO_STUB_TRACKS   = "stubs/rastreamentos_stub.pkl"   # Cache opcional de rastreamentos


def principal():
    """Fluxo principal do sistema de análise."""

    # ── 1. Leitura do vídeo ──
    print("=" * 60)
    print(" SISTEMA DE ANÁLISE DE FUTEBOL COM IA/ML")
    print("=" * 60)

    frames = ler_video(CAMINHO_VIDEO_ENTRADA)
    fps = obter_fps_video(CAMINHO_VIDEO_ENTRADA)

    # ── 2. Detecção e Rastreamento ──
    print("\n[ETAPA 1] Detecção e Rastreamento de objetos...")
    rastreador = Rastreador(caminho_modelo=CAMINHO_MODELO_YOLO)

    # Tenta carregar rastreamentos de cache (evita reprocessar durante desenvolvimento)
    rastreamentos = rastreador.obter_rastreamentos_objetos(
        frames,
        ler_do_cache=True,
        caminho_cache=CAMINHO_STUB_TRACKS,
    )

    rastreamento_jogadores = rastreamentos["jogadores"]
    rastreamento_arbitros  = rastreamentos["arbitros"]
    rastreamento_bola      = rastreamentos["bola"]

    # ── 3. Interpolação da posição da bola ──
    print("[ETAPA 2] Interpolando posição da bola...")
    rastreamento_bola = rastreador.interpolar_posicoes_bola(rastreamento_bola)
    rastreamentos["bola"] = rastreamento_bola  # Sincronizar com o dicionário principal

    # ── 4. Atribuição de times ──
    print("[ETAPA 3] Identificando times por cor da camisa (K-Means)...")
    atribuidor_times = AtribuidorTimes()
    atribuidor_times.definir_cores_times(frames[0], rastreamento_jogadores[0])

    for indice_frame, info_jogadores in enumerate(rastreamento_jogadores):
        for id_jogador, dados_jogador in info_jogadores.items():
            time = atribuidor_times.obter_time_jogador(
                frames[indice_frame],
                dados_jogador["bbox"],
                id_jogador,
            )
            rastreamento_jogadores[indice_frame][id_jogador]["time"] = time
            rastreamento_jogadores[indice_frame][id_jogador]["cor_time"] = (
                atribuidor_times.cores_times[time]
            )

    # ── 5. Atribuição de posse de bola ──
    print("[ETAPA 4] Calculando posse de bola...")
    atribuidor_bola = AtribuidorBola()
    posse_por_time = atribuidor_bola.atribuir_posse(
        rastreamento_jogadores,
        rastreamento_bola,
    )

    # ── 6. Compensação de câmera (Optical Flow) ──
    print("[ETAPA 5] Estimando movimento de câmera (Optical Flow)...")
    estimador_camera = EstimadorMovimentoCamera(frames)
    movimento_camera = estimador_camera.obter_movimento_camera()

    # Subtrair deslocamento da câmera das posições dos jogadores e bola
    rastreador.adicionar_posicao_ajustada(rastreamento_jogadores, movimento_camera)
    rastreador.adicionar_posicao_ajustada(rastreamento_arbitros, movimento_camera)
    rastreador.adicionar_posicao_ajustada(rastreamento_bola, movimento_camera)

    # ── 7. Transformação de perspectiva ──
    print("[ETAPA 6] Aplicando transformação de perspectiva (pixels → metros)...")
    transformador = TransformadorVisao()
    transformador.adicionar_posicao_transformada(rastreamento_jogadores)
    transformador.adicionar_posicao_transformada(rastreamento_arbitros)
    transformador.adicionar_posicao_transformada(rastreamento_bola)

    # ── 8. Velocidade e distância ──
    print("[ETAPA 7] Calculando velocidade e distância percorrida...")
    estimador_vel = EstimadorVelocidadeDistancia(fps=fps)
    estimador_vel.adicionar_velocidade_e_distancia(rastreamento_jogadores)
    estimador_vel.adicionar_velocidade_e_distancia(rastreamento_arbitros)

    # ── 9. Anotação visual ──
    print("[ETAPA 8] Gerando vídeo anotado...")
    frames_anotados = rastreador.desenhar_anotacoes(frames, rastreamentos)

    # Anotar velocidade e distância nos frames
    frames_anotados = estimador_vel.desenhar_velocidade_distancia(
        frames_anotados, rastreamento_jogadores
    )
    frames_anotados = estimador_vel.desenhar_velocidade_distancia(
        frames_anotados, rastreamento_arbitros
    )

    # Anotar posse de bola por time
    frames_anotados = atribuidor_bola.desenhar_posse_bola(
        frames_anotados, posse_por_time
    )

    # ── 10. Salvar vídeo final ──
    salvar_video(frames_anotados, CAMINHO_VIDEO_SAIDA, fps=int(fps))

    print("\n" + "=" * 60)
    print(" PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    principal()
