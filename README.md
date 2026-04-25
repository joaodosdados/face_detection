# 🧠 Face Recognition com InsightFace + GPU (Windows)

Sistema de **reconhecimento facial em tempo real** com suporte a múltiplas pessoas, embeddings robustos e aceleração por GPU (CUDA).

---

## 🚀 Funcionalidades

- Reconhecimento facial em tempo real via webcam
- Suporte a múltiplas pessoas
- Múltiplas imagens por pessoa para melhorar a robustez
- Embedding médio por identidade
- Comparação por similaridade de cosseno
- Threshold e margem de decisão configuráveis
- Aceleração com GPU NVIDIA via CUDA
- Fallback automático para CPU
- Configuração centralizada via `config.json`
- Exibição de FPS em tempo real

---

## 📦 Estrutura do Projeto

```text
face_detection/
│
├── face_detection.py
├── config.json
├── requirements.txt
└── img/
    └── references/
        ├── joao_lucas/
        │   ├── joao_lucas_1.png
        │   └── joao_lucas_2.png
        │
        └── amanda/
            └── amanda_1.png
```

---

## ⚙️ Requisitos

### Sistema

- Windows 10 ou Windows 11
- GPU NVIDIA opcional, mas recomendada
- CUDA e cuDNN configurados para uso com `onnxruntime-gpu`

### Python

- Python 3.10 a 3.12

---

## 📦 Instalação

### 1. Criar ambiente virtual

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

---

### 2. Criar `requirements.txt`

```txt
insightface==0.7.3
onnxruntime-gpu
opencv-python
numpy
```

---

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## ⚙️ Exemplo de `config.json`

```json
{
  "paths": {
    "reference_dir": "img/references",
    "cuda_path": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin",
    "cudnn_path": "C:/Program Files/NVIDIA/CUDNN/v9.xx/bin/12.9/x64"
  },
  "model": {
    "name": "buffalo_l",
    "use_gpu": true,
    "det_size": [640, 640],
    "detection_threshold": 0.1
  },
  "recognition": {
    "threshold": 0.35,
    "margin": 0.01,
    "process_every_n_frames": 3
  },
  "camera": {
    "index": 0,
    "width": 640,
    "height": 480
  },
  "ui": {
    "show_fps": true,
    "window_name": "Face Recognition"
  }
}
```

---

## ▶️ Executar

```bash
python face_detection.py
```

Pressione `Q` para sair.

---

## 🧠 Como funciona

O pipeline realiza as seguintes etapas:

1. Carrega o modelo `InsightFace`
2. Lê as imagens de referência em `img/references/`
3. Detecta a face em cada imagem de referência
4. Gera embeddings faciais
5. Calcula um embedding médio por pessoa
6. Captura frames da webcam
7. Detecta faces em tempo real
8. Compara cada face com as referências usando similaridade de cosseno
9. Exibe o nome da pessoa ou `Desconhecido`

---

## 📂 Como adicionar novas pessoas

Cada pessoa deve ter uma pasta própria dentro de:

```text
img/references/
```

Exemplo:

```text
img/references/joao_lucas/
    joao_lucas_1.png
    joao_lucas_2.png

img/references/amanda/
    amanda_1.png
```

O nome exibido na tela será baseado no nome da pasta.

Exemplo:

```text
joao_lucas -> Joao Lucas
amanda -> Amanda
```

---

## 🎯 Ajustes importantes

### Threshold de reconhecimento

Controla o quão parecido o rosto precisa ser para ser aceito como uma pessoa conhecida.

| Valor | Comportamento |
| ----- | ------------- |
| 0.30  | Mais permissivo |
| 0.35  | Flexível/equilibrado |
| 0.50  | Mais rigoroso |
| 0.60+ | Muito rigoroso |

Exemplo recomendado para inferência mais flexível:

```json
"threshold": 0.35
```

---

### Margin

Controla a diferença mínima entre o melhor candidato e o segundo melhor candidato.

| Valor | Comportamento |
| ----- | ------------- |
| 0.01  | Mais flexível |
| 0.03  | Moderado |
| 0.05+ | Mais seguro, porém mais rígido |

Exemplo recomendado:

```json
"margin": 0.01
```

---

### Detection threshold

Controla o quão sensível o detector de faces será.

| Valor | Comportamento |
| ----- | ------------- |
| 0.1   | Muito permissivo |
| 0.3   | Equilibrado |
| 0.5+  | Mais rigoroso |

Exemplo recomendado:

```json
"detection_threshold": 0.1
```

---

### det_size

Controla o tamanho usado pelo detector.

| Valor | Uso |
| ----- | --- |
| [640, 640] | Mais rápido |
| [1024, 1024] | Mais preciso, porém mais pesado |
| [1280, 1280] | Mais robusto, porém mais lento |

---

## ⚠️ Atenção: BGR vs RGB

O OpenCV lê imagens em formato **BGR**, enquanto muitos modelos de visão computacional trabalham melhor com **RGB**.

Por isso, antes de passar imagens para o modelo, use:

```python
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
faces = app.get(image_rgb)
```

O mesmo vale para frames da webcam:

```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
last_faces = app.get(frame_rgb)
```

Esse ajuste é essencial para evitar falhas como:

```text
[AVISO] Nenhuma face detectada
```

---

## ⚡ GPU com CUDA

Para validar se a GPU está sendo usada, ao executar o script deve aparecer algo parecido com:

```text
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Se aparecer apenas:

```text
Applied providers: ['CPUExecutionProvider']
```

então a GPU não foi carregada corretamente.

---

## 🧪 Problemas comuns

### Nenhuma face detectada nas referências

Possíveis causas:

- Imagem foi lida em BGR e não convertida para RGB
- Caminho da imagem está incorreto
- Imagem não foi carregada corretamente pelo OpenCV
- `detection_threshold` está muito alto
- `det_size` está muito baixo para o caso

Sugestão:

```python
print(image_path.resolve())
print(image_path.exists())
print(image.shape)
```

---

### Reconhecimento muito rígido

Reduza os valores no `config.json`:

```json
"recognition": {
  "threshold": 0.35,
  "margin": 0.01,
  "process_every_n_frames": 3
}
```

---

### Reconhecimento muito permissivo

Aumente os valores:

```json
"recognition": {
  "threshold": 0.50,
  "margin": 0.05,
  "process_every_n_frames": 3
}
```

---

### GPU não funciona

Verifique:

- CUDA instalado
- cuDNN instalado
- `onnxruntime-gpu` instalado
- Caminhos `cuda_path` e `cudnn_path` corretos no `config.json`

---

## 📈 Performance esperada

| Setup | FPS estimado |
| ----- | ------------ |
| CPU   | 5–15 FPS |
| GPU   | 20–60 FPS |

A performance depende de:

- Modelo usado
- Resolução da webcam
- `det_size`
- Quantidade de faces no frame
- Frequência definida em `process_every_n_frames`

---

## 🚀 Próximos passos

Possíveis melhorias futuras:

- Cache de embeddings para não recalcular referências sempre
- Tracking de faces entre frames
- Banco vetorial com FAISS
- API com FastAPI
- Interface com Streamlit
- Logs estruturados
- Exportação de eventos de reconhecimento
- Deploy em ambiente controlado

---

## 👨‍💻 Autor

[João Lucas Oliveira](https://www.linkedin.com/in/joaodosdados/)
