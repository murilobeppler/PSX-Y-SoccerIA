# Sistema de Análise de Futebol com IA/ML (PSX-Y-SoccerIA)

Este é um sistema modular de análise tática de futebol baseado em visão computacional e aprendizado de máquina. A partir de um vídeo de transmissão aérea ou tática de um jogo, o sistema realiza detecção, rastreamento de jogadores e bola, identificação de times, cálculo de posse de bola, estimativa de deslocamento de câmera, transformação de perspectiva para metros, e cálculo de velocidade e distância percorrida por cada atleta.

---

## 📂 Estrutura de Diretórios e Arquivos

```
PSX-Y-SoccerIA/
├── principal.py                         # Orquestrador principal (Pipeline de execução)
├── requirements.txt                     # Dependências do projeto (PyTorch, Ultralytics, OpenCV, etc.)
├── .gitignore                           # Ignora ambientes virtuais, cache e arquivos pesados de dados
├── README.md                            # Guia rápido e documentação do sistema (Este arquivo)
│
├── videos/                              # Diretório para vídeos de entrada (ex: entrada.mp4) e saída
├── modelos/                             # Diretório para pesos do YOLO (ex: melhor_modelo.pt)
├── stubs/                               # Cache de rastreamento de detecções em formato Pickle (.pkl)
│
└── futebol_ia/                          # Pacote principal da biblioteca
    ├── __init__.py
    ├── utils.py                         # Utilitários auxiliares de vídeo, cálculo geométrico e Bbox
    │
    ├── rastreadores/
    │   ├── __init__.py
    │   └── rastreador.py                # Detecção YOLOv8 + Rastreamento ByteTrack + Anotações
    │
    ├── atribuicao_times/
    │   ├── __init__.py
    │   └── atribuidor_times.py          # Classificação de times por cor de camisa (K-Means)
    │
    ├── atribuicao_bola/
    │   ├── __init__.py
    │   └── atribuidor_bola.py           # Cálculo de posse de bola por proximidade e estatísticas
    │
    ├── camera/
    │   ├── __init__.py
    │   └── estimador_movimento_camera.py # Estabilização de deslocamento via Fluxo Óptico (Lucas-Kanade)
    │
    ├── transformacao/
    │   ├── __init__.py
    │   └── transformador_visao.py        # Projeção de perspectiva (Homografia Homogênea Pixels ➜ Metros)
    │
    └── velocidade/
        ├── __init__.py
        └── estimador_velocidade_distancia.py # Estimativa de velocidade (km/h) e distância (m)
```

---

## 🛠️ Detalhamento dos Módulos e Classes

### 1. Orquestrador Principal (`principal.py`)
Responsável por gerenciar todo o pipeline sequencial de processamento.
*   **Pipeline de 10 Etapas:**
    1. Lê o vídeo original frame por frame.
    2. Inicializa o `Rastreador` e detecta/rastreia todos os elementos (jogadores, árbitro, bola).
    3. Estima o movimento físico da câmera para ajuste de coordenadas.
    4. Corrige as posições dos objetos com base no deslocamento da câmera.
    5. Realiza a interpolação linear das posições da bola para preencher frames com detecção falha.
    6. Inicializa o `TransformadorVisao` para mapear coordenadas da tela para metros reais.
    7. Estima a velocidade média (km/h) e distância percorrida acumulada (m) dos jogadores.
    8. Define as cores dos dois times utilizando K-Means e atribui cada jogador ao seu respectivo time.
    9. Determina a posse da bola a cada frame com base na distância mínima dos jogadores.
    10. Desenha todas as anotações gráficas de saída e exporta o novo vídeo anotado.

### 2. Utilitários Gerais (`futebol_ia/utils.py`)
Funções utilitárias puras sem estado.
*   `ler_video(caminho_video)`: Retorna uma lista de frames OpenCV BGR e metadados.
*   `salvar_video(frames, caminho_saida, fps)`: Codifica a lista de frames em arquivo `.avi` ou `.mp4`.
*   `obter_centro_bbox(bbox)`: Retorna o centro geométrico `(x, y)` de uma Bounding Box.
*   `obter_largura_bbox(bbox)`: Retorna a largura horizontal em pixels da Bounding Box.
*   `obter_posicao_pe(bbox)`: Calcula o ponto inferior central `(x, y)` da Bounding Box, simulando a posição do pé do jogador em contato com o gramado.
*   `calcular_distancia(p1, p2)`: Retorna a distância euclidiana simples entre dois pontos.

### 3. Rastreamento e Detecção (`futebol_ia/rastreadores/rastreador.py`)
Gerencia o modelo de Deep Learning (`ultralytics` YOLOv8) e o rastreador de múltiplos objetos (`supervision` ByteTrack).
*   **Classe `Rastreador`**:
    *   `detectar_frames(frames)`: Executa inferência em lotes (batch size = 20) para otimização de GPU/CPU.
    *   `obter_rastreamentos_objetos(frames, ler_stub, caminho_stub)`: Orquestra o ciclo completo de detecção, aplica o ByteTrack nos jogadores e árbitros, e mantém o rastreamento consistente de IDs. Utiliza cache em arquivo pickle (`.pkl`) na pasta `stubs/` para poupar tempo de inferência nos testes subsequentes.
    *   `interpolar_posicoes_bola(rastreamentos_bola)`: Interpola frames ausentes onde a bola foi obscurecida ou não detectada, usando o algoritmo de interpolação linear do `pandas.DataFrame.interpolate()`.
    *   `desenhar_anotacoes(frames, rastreamentos, posse_bola_por_frame)`: Plota elipses de base translúcida sob os pés dos jogadores/árbitros, caixas com o ID correspondente e um triângulo flutuante indicando a posição da bola.
*   **Especificidade Adotada:** Os goleiros (classe YOLO `1`) são reclassificados temporariamente como jogadores (classe `2`) antes de entrarem no ByteTrack. Isso garante consistência de ID caso o goleiro seja rastreado alternadamente ou cruze com outros atletas.

### 4. Atribuição de Times (`futebol_ia/atribuicao_times/atribuidor_times.py`)
Classifica os jogadores em dois times usando aprendizado não supervisionado sobre as cores das camisas.
*   **Classe `AtribuidorTimes`**:
    *   `_obter_cor_camisa(frame, bbox)`: Realiza o recorte (*crop*) da metade superior da Bounding Box do jogador. Aplica um algoritmo K-Means local ($k=2$) para segmentar os pixels em dois clusters: um representa a camisa do jogador e o outro representa o fundo (gramado/placas). A cor da camisa é determinada descartando o cluster cuja média é mais próxima das cores encontradas nos quatro cantos do crop (que geralmente são grama).
    *   `definir_cores_times(frame, rastreamentos_jogadores)`: Coleta as cores extraídas de todos os jogadores no primeiro frame de vídeo e aplica um K-Means global ($k=2$) para separar as cores em dois grupos representativos de uniformes (Time A e Time B).
    *   `obter_time_jogador(frame, bbox, id_jogador)`: Classifica dinamicamente o time do jogador atribuindo-o ao cluster K-Means mais próximo. As classificações são salvas em um cache interno associado ao ID do jogador, evitando reprocessamento nos frames subsequentes.

### 5. Atribuição de Posse de Bola (`futebol_ia/atribuicao_bola/atribuidor_bola.py`)
Determina qual jogador tem o controle de jogo a cada momento.
*   **Classe `AtribuidorBola`**:
    *   `atribuir_posse(rastreamentos_jogadores, rastreamento_bola)`: Calcula a distância geométrica entre os pés de cada jogador e a posição central inferior da bola. O jogador com a menor distância abaixo do limiar de **70 pixels** (`LIMIAR_POSSE`) recebe a posse de bola no frame correspondente.
    *   `desenhar_posse_bola(frame, num_frame, times_jogadores)`: Renderiza no topo da tela uma barra de progresso gráfica translúcida no estilo transmissão esportiva de TV, indicando a porcentagem acumulada de posse de bola de cada time no decorrer do vídeo.

### 6. Estimativa de Deslocamento de Câmera (`futebol_ia/camera/estimador_movimento_camera.py`)
Mede as translações panorâmicas da câmera de transmissão para isolar a movimentação real dos atletas da movimentação da própria câmera.
*   **Classe `EstimadorMovimentoCamera`**:
    *   `obter_movimento_camera(frames, ler_stub, caminho_stub)`: Detecta pontos de destaque estáticos (Shi-Tomasi) e rastreia seus deslocamentos entre frames adjacentes usando o Fluxo Óptico de Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`).
    *   **Especificidade Adotada:** A busca por pontos de interesse foca exclusivamente em uma faixa de borda estática (20 pixels de margem lateral), que tipicamente engloba placas publicitárias e arquibancadas fixas. O deslocamento final é determinado pela mediana dos vetores de movimento para descartar outliers como a movimentação de pessoas fora de campo.

### 7. Transformação de Visão e Perspectiva (`futebol_ia/transformacao/transformador_visao.py`)
Mapeia as coordenadas 2D distorcidas da câmera para um plano retangular ortogonal representando as dimensões reais do campo em metros.
*   **Classe `TransformadorVisao`**:
    *   `transformar_ponto(ponto)`: Usa homografia de perspectiva (`cv2.perspectiveTransform`) baseada em um trapézio calibrado de pixels mapeado para as dimensões regulamentares de um campo padrão (105m de comprimento por 68m de largura).
    *   `adicionar_posicao_transformada(rastreamentos)`: Adiciona uma chave `"posicao_transformada"` a todas as detecções, representando a posição 2D do jogador em metros (com origem no canto superior esquerdo do campo).

### 8. Velocidade e Distância (`futebol_ia/velocidade/estimador_velocidade_distancia.py`)
Calcula a atividade física dos jogadores com base na variação de suas posições no plano real em metros.
*   **Classe `EstimadorVelocidadeDistancia`**:
    *   `adicionar_velocidade_e_distancia(rastreamentos)`: Calcula a distância percorrida entre frames em metros e deriva a velocidade em km/h.
    *   **Especificidade Adotada:** Para mitigar ruídos e variações bruscas de detecção (ex: jittering), o cálculo utiliza uma janela deslizante de suavização de **5 frames** (`JANELA_FRAMES`). A velocidade calculada para o intervalo é replicada de forma estável para os frames correspondentes da janela.
    *   `desenhar_velocidade_distancia(frames, rastreamentos)`: Desenha overlays de texto abaixo da elipse de cada jogador mostrando a sua velocidade instantânea e a distância acumulada percorrida.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python 3.9+ instalado na máquina.

### 1. Configurar o Ambiente Virtual
Utilizando o PowerShell do Windows, inicialize o ambiente virtual e ative-o:
```powershell
# Criação do venv (se já não estiver criado)
python -m venv soccerIA

# Ativação do ambiente no PowerShell
.\soccerIA\Scripts\Activate.ps1
```

### 2. Instalar as Dependências
Com o ambiente virtual ativado (`(soccerIA)` no início da linha de prompt):
```powershell
pip install -r requirements.txt
```

### 3. Preparar os Arquivos e Pastas de Entrada
Garanta a presença das pastas e arquivos de mídia:
1. Coloque o vídeo original a ser analisado em `videos/entrada.mp4` (ou mude a variável no topo do arquivo `principal.py`).
2. Adicione os pesos do YOLOv8 treinado em `modelos/melhor_modelo.pt`. Se desejar usar um modelo padrão da Ultralytics para testes rápidos, use `yolov8x.pt` (o YOLO fará o download automático se não encontrar localmente).

### 4. Executar
Rode o script principal de processamento:
```powershell
python principal.py
```
O vídeo de saída será gerado em `videos/video_saida.avi` (ou formato definido em `principal.py`).

---

## 📐 Calibração da Perspectiva (Pixels para Metros)

No arquivo `futebol_ia/transformacao/transformador_visao.py`, você encontrará a seguinte variável de calibração padrão:

```python
self.vertices_pixel = np.array([
    [110, 720],   # Canto inferior esquerdo da tela
    [265, 335],   # Canto superior esquerdo da tela
    [1020, 335],  # Canto superior direito da tela
    [1720, 720]   # Canto inferior direito da tela
])
```

> [!WARNING]
> Esses pontos definem os vértices de um trapézio correspondente às linhas de marcação reais de um campo de futebol no vídeo de entrada.
> **Para qualquer novo vídeo que utilize uma câmera ou ângulo diferente, estes 4 pontos devem ser recalibrados** para garantir que as medições de velocidade (km/h) e distância percorrida sejam fisicamente corretas.