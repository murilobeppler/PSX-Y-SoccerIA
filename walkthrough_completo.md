# Sistema de Análise de Futebol com IA/ML — Progresso

## Estrutura de Diretórios

```
PS X,Y SOCCER IA/
├── principal.py                          ← ✅ Orquestrador principal (main)
├── requirements.txt                      ← ✅ Dependências do projeto
├── videos/                               ← Vídeos de entrada e saída
├── modelos/                              ← Pesos do YOLO treinado
├── stubs/                                ← Cache de rastreamentos (pkl)
└── futebol_ia/                           ← Pacote principal
    ├── __init__.py
    ├── utils.py                          ← ✅ Funções de vídeo + geometria de bbox
    ├── rastreadores/
    │   ├── __init__.py
    │   └── rastreador.py                 ← ✅ YOLO + ByteTrack + Anotações
    ├── atribuicao_times/
    │   ├── __init__.py
    │   └── atribuidor_times.py           ← ✅ K-Means na cor da camisa
    ├── atribuicao_bola/
    │   ├── __init__.py
    │   └── atribuidor_bola.py            ← ✅ Posse de bola por distância
    ├── camera/
    │   ├── __init__.py
    │   └── estimador_movimento_camera.py  ← ✅ Optical Flow Lucas-Kanade
    ├── transformacao/
    │   ├── __init__.py
    │   └── transformador_visao.py         ← ✅ Perspectiva pixels→metros
    └── velocidade/
        ├── __init__.py
        └── estimador_velocidade_distancia.py ← ✅ Velocidade km/h + Distância m
```

---

## Passo 1 — ✅ Concluído

### [utils.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/futebol_ia/utils.py)
| Função | Descrição |
|---|---|
| `ler_video()` | Lê todos os frames de um vídeo (BGR) |
| `salvar_video()` | Exporta frames como vídeo MP4/AVI |
| `obter_centro_bbox()` | Centro `(cx, cy)` de uma bounding box |
| `obter_largura_bbox()` | Largura da bbox |
| `obter_posicao_pe()` | Ponto inferior-central (pés do jogador) |
| `calcular_distancia()` | Distância euclidiana entre dois pontos |
| `obter_fps_video()` | FPS do arquivo de vídeo |

### [principal.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/principal.py)
Pipeline completo com 10 etapas sequenciais, desde leitura do vídeo até exportação anotada.

---

## Passo 2 — ✅ Concluído

### [rastreador.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/futebol_ia/rastreadores/rastreador.py)

Classe `Rastreador` com os seguintes métodos:

| Método | Descrição |
|---|---|
| `detectar_frames()` | Detecção YOLO em lotes de 20 frames |
| `obter_rastreamentos_objetos()` | Pipeline completo: detecção → conversão goleiro→jogador → ByteTrack. Suporta cache pickle |
| `interpolar_posicoes_bola()` | Interpolação via pandas para frames perdidos |
| `adicionar_posicao_ajustada()` | Subtrai deslocamento de câmera dos pés dos objetos |
| `desenhar_anotacoes()` | Orquestra anotações visuais de todos os objetos |
| `_desenhar_elipse()` | Elipse nos pés + retângulo preenchido com ID |
| `_desenhar_triangulo()` | Triângulo invertido sobre a bola |

**Decisões técnicas:**
- Goleiros são reclassificados como Jogadores (`class_id`) antes do ByteTrack para rastreamento unificado
- A bola não passa pelo ByteTrack (há apenas 1 bola); usa a detecção direta com maior confiança
- Cache pickle evita reprocessamento durante desenvolvimento

---

## Passo 3 — ✅ Concluído

### [atribuidor_times.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/futebol_ia/atribuicao_times/atribuidor_times.py)

Classe `AtribuidorTimes`:

| Método | Descrição |
|---|---|
| `_obter_cor_camisa()` | Crop metade superior da bbox → K-Means(k=2) → separa camisa do fundo pelos cantos |
| `definir_cores_times()` | Coleta cores de todos os jogadores no 1º frame → K-Means(k=2) global → define 2 times |
| `obter_time_jogador()` | Classifica jogador no time mais próximo; mantém cache por ID |

**Decisões técnicas:**
- Dois níveis de K-Means: (1) pixel-level para isolar cor da camisa, (2) player-level para agrupar em 2 times
- Cluster de fundo identificado pela cor dominante nos 4 cantos do crop
- Cache `_cache_times[id_jogador]` evita reclassificação e mantém consistência temporal

### [atribuidor_bola.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/futebol_ia/atribuicao_bola/atribuidor_bola.py)

Classe `AtribuidorBola`:

| Método | Descrição |
|---|---|
| `atribuir_posse()` | Para cada frame: calcula distância bola→pés, atribui posse se < 70px |
| `desenhar_posse_bola()` | Barra de porcentagem acumulada com overlay semi-transparente |

**Decisões técnicas:**
- `LIMIAR_POSSE = 70` pixels como threshold de proximidade
- Posição da bola: ponto inferior-central (contato com chão)
- Porcentagem acumulada frame a frame com visualização estilo broadcast

---

## Passo 4 — ✅ Concluído

### [estimador_movimento_camera.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/futebol_ia/camera/estimador_movimento_camera.py)

Classe `EstimadorMovimentoCamera`:

| Método | Descrição |
|---|---|
| `_criar_mascara_bordas()` | Máscara de 20px nas laterais (fundo estático) |
| `obter_movimento_camera()` | Shi-Tomasi → Lucas-Kanade → mediana do deslocamento por frame |

**Decisões técnicas:**
- Faixa de 20px nas bordas laterais → elementos estáticos (arquibancadas, placas)
- Mediana em vez de média para robustez contra outliers
- Suporta cache pickle para reutilização

### [transformador_visao.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/futebol_ia/transformacao/transformador_visao.py)

Classe `TransformadorVisao`:

| Método | Descrição |
|---|---|
| `transformar_ponto()` | Converte (x,y) pixels → (x_m, y_m) metros via `cv2.perspectiveTransform` |
| `adicionar_posicao_transformada()` | Aplica transformação a todos os objetos rastreados |

**Decisões técnicas:**
- Campo padrão: 105m × 68m
- `vertices_pixel` define o trapézio no vídeo (deve ser calibrado por câmera)
- `cv2.pointPolygonTest` valida se o ponto está dentro da região mapeável
- Fallback para centro inferior da bbox se `posicao_ajustada` não existir

---

## Passo 5 — ✅ Concluído

### [estimador_velocidade_distancia.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/futebol_ia/velocidade/estimador_velocidade_distancia.py)

Classe `EstimadorVelocidadeDistancia`:

| Método | Descrição |
|---|---|
| `adicionar_velocidade_e_distancia()` | Janela de 5 frames → distância (m) → velocidade (km/h) |
| `desenhar_velocidade_distancia()` | Overlay semi-transparente com vel + dist abaixo de cada jogador |

**Decisões técnicas:**
- `JANELA_FRAMES = 5` para suavização de ruídos
- Fórmula: `dist_metros / (num_frames / fps)` → m/s × 3.6 → km/h
- Valores propagados para todos os frames da janela

### Ajustes Finais no [principal.py](file:///c:/Users/muril/Documents/PS%20X,Y%20SOCCER%20IA/principal.py)
- Removidos imports não utilizados (`os`, `numpy`)
- Adicionada sincronização: `rastreamentos["bola"] = rastreamento_bola` após interpolação

---

## 🏆 Projeto Completo

**14 arquivos** | **6 módulos** | ~40KB de código

### Como usar:
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Colocar vídeo em videos/entrada.mp4
# 3. Colocar peso YOLO em modelos/melhor_modelo.pt
# 4. Calibrar vertices_pixel em transformador_visao.py

# 5. Executar
python principal.py
```

> [!IMPORTANT]
> Os `vertices_pixel` em `transformador_visao.py` devem ser calibrados para o ângulo de câmera do seu vídeo específico.
