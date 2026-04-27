# MVP de Controle de Acesso Biométrico

MVP em Python para testes reais de reconhecimento facial com InsightFace, OpenCV, votação temporal, rastreamento leve, filtro de qualidade, entrada por webcam/RTSP, cache de embeddings e logs em CSV.

Este projeto é apenas para avaliação em campo. Ele não abre portas, não libera catracas e não dispara nenhuma ação real de controle de acesso biométrico.

![Dashboard do controle de acesso biométrico](docs/biometric_access.png)

## O Que Este MVP Faz

- Lê vídeo de webcam ou câmera RTSP, incluindo URLs no padrão Intelbras
- Detecta rostos com InsightFace usando o modelo `buffalo_l`
- Converte BGR para RGB antes de toda inferência do modelo
- Mantém os frames em BGR para exibição com OpenCV
- Filtra rostos pequenos ou borrados antes do reconhecimento
- Rastreia rostos entre frames com distância entre centros
- Usa votação temporal em vez de confiar em um único frame
- Confirma uma identidade somente após votos consistentes
- Registra eventos de reconhecimento em CSV
- Usa cache de embeddings para acelerar a inicialização
- Mostra caixas, track IDs, scores, votos e FPS
- Exibe métricas de execução no terminal

Escala-alvo deste MVP: cerca de 10 pessoas cadastradas.

## Estrutura Do Projeto

```text
face_detection/
|-- face_detection.py
|-- dashboard.py
|-- config.example.json
|-- config.json                 # local/privado, ignorado pelo git
|-- requirements.txt
|-- req.txt
|-- README.md
|-- docs/
|   |-- PROJECT_HANDOFF.md
|   `-- biometric_access.png
|-- src/
|   |-- config.py
|   |-- drawing.py
|   |-- logging_utils.py
|   |-- model.py
|   |-- quality.py
|   |-- recognition.py
|   |-- references.py
|   |-- runtime.py
|   |-- tracking.py
|   `-- video.py
|-- web/
|   |-- static/
|   |   |-- app.js
|   |   `-- styles.css
|   `-- templates/
|       `-- index.html
|-- img/
|   `-- references/
|       `-- .gitkeep
|-- data/
|   `-- .gitkeep
`-- logs/
    `-- .gitkeep
```

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

O arquivo `req.txt` foi mantido por compatibilidade com a primeira versão do projeto. Para novas instalações, prefira `requirements.txt`.

## Arquitetura

O projeto tem dois modos de execução que compartilham o mesmo motor de reconhecimento.

```text
face_detection.py
```

Executa o MVP com a janela clássica do OpenCV.

```text
dashboard.py
```

Executa um backend FastAPI local para o dashboard no navegador.

```text
src/runtime.py
```

Contém o loop compartilhado de reconhecimento. Ele abre a fonte de vídeo, executa o InsightFace, atualiza tracking/votação, registra eventos, salva snapshots de desconhecidos e mantém o frame/métricas mais recentes disponíveis para o dashboard.

```text
web/templates/index.html
web/static/app.js
web/static/styles.css
```

Contém o frontend. O navegador lê o stream MJPEG ao vivo em `/video`, consulta `/api/state` a cada segundo, renderiza alertas de desconhecidos e chama `/api/stop` quando o botão `Stop Camera` é clicado.

Rotas do backend:

```text
GET  /           HTML do dashboard
GET  /video      stream MJPEG ao vivo
GET  /api/state  métricas, tracks e eventos recentes em JSON
POST /api/stop   para o runtime da câmera
```

`uvicorn` é o servidor web local usado para executar a aplicação FastAPI.

## Configuração

Crie seu arquivo local a partir do exemplo seguro:

```powershell
Copy-Item config.example.json config.json
```

O `config.json` é ignorado pelo git porque pode conter credenciais de câmera e caminhos locais de CUDA/cuDNN.

### Webcam

```json
"video_source": {
  "type": "webcam",
  "index": 0,
  "width": 1280,
  "height": 720,
  "buffer_size": 1
}
```

### RTSP

```json
"video_source": {
  "type": "rtsp",
  "url": "rtsp://user:password@camera-ip:554/cam/realmonitor?channel=1&subtype=0",
  "buffer_size": 1
}
```

Para câmeras Intelbras, use o stream principal para obter mais detalhes do rosto:

```text
rtsp://user:password@ip:554/cam/realmonitor?channel=1&subtype=0
```

Use `subtype=1` apenas se latência ou banda forem mais importantes que qualidade de reconhecimento.

## Imagens De Referência

Adicione uma pasta por pessoa:

```text
img/references/joao_lucas/
    image_1.jpg
    image_2.png

img/references/maria_silva/
    image_1.jpg
```

O nome da pasta vira o nome exibido:

```text
joao_lucas -> Joao Lucas
maria_silva -> Maria Silva
```

Use imagens nítidas, com boa iluminação e variação de ângulo. O carregador seleciona o maior rosto quando há mais de uma face na imagem, normaliza cada embedding e salva um embedding médio normalizado por pessoa.

## Estabilidade Do Reconhecimento

O reconhecimento é baseado no histórico do track, não em um único frame.

```json
"recognition": {
  "process_every_n_frames": 2,
  "recognition_window_frames": 12,
  "min_votes_to_confirm": 5,
  "min_average_score": 0.35,
  "candidate_min_score": 0.25,
  "max_unknown_frames": 20
}
```

Um track vira `confirmed` somente quando o mesmo nome aparece pelo menos `min_votes_to_confirm` vezes dentro dos últimos `recognition_window_frames`, e esses votos têm score médio maior ou igual a `min_average_score`.

`candidate_min_score` é intencionalmente menor que o score de confirmação para que movimento e blur não descartem frames úteis cedo demais.

## Qualidade Da Face

```json
"quality": {
  "min_face_width": 70,
  "min_face_height": 70,
  "min_blur_score": 25.0,
  "min_detection_score": 0.1
}
```

Rostos abaixo do tamanho mínimo ou abaixo do limite de nitidez são ignorados. O blur é medido com variância do Laplaciano no crop do rosto.

## Tracking

```json
"tracking": {
  "max_center_distance": 180,
  "max_missing_frames": 20
}
```

O tracker é propositalmente leve. Ele associa novas detecções a tracks existentes pela distância entre o centro dos rostos, mantém um histórico curto de predições e remove tracks antigos após muitos frames sem detecção.

## Cache

```json
"cache": {
  "enabled": true,
  "path": "data/embeddings_cache.pkl",
  "force_rebuild": false
}
```

O cache guarda o nome da pessoa, embedding médio, quantidade de imagens e um hash gerado a partir dos caminhos, tamanhos e datas de modificação das imagens de referência.

Defina `force_rebuild` como `true` depois de alterar imagens de referência caso queira forçar a reconstrução dos embeddings.

## Logs

Os eventos são salvos em:

```text
logs/recognition_events.csv
```

Colunas:

```text
timestamp, track_id, name, status, avg_score, votes, frame_number, x1, y1, x2, y2, snapshot_path
```

O logger registra mudanças de status, confirmações e falhas de reconhecimento desconhecido após frames suficientes. Ele evita escrever evento duplicado a cada frame.

Snapshots de rosto são salvos somente para alertas `unknown`:

```text
logs/snapshots/
  unknown/
```

## Executar

```powershell
python face_detection.py
```

Pressione `Q` na janela de vídeo para sair.

O terminal mostra:

- pessoas cadastradas
- fonte de vídeo
- modo GPU/CPU
- tracks ativos
- FPS médio
- reconhecimentos confirmados

## Dashboard

Execute o dashboard local com:

```powershell
uvicorn dashboard:app --host 127.0.0.1 --port 8000
```

Depois abra:

```text
http://127.0.0.1:8000
```

O dashboard mostra:

- câmera ao vivo
- FPS e modo de execução
- cards de alerta `unknown` com snapshot do rosto
- contagem de reconhecimentos confirmados
- quantidade de pessoas cadastradas
- eventos recentes do CSV

O dashboard ainda é apenas para avaliação. Ele exibe e registra o estado do reconhecimento, mas não dispara nenhuma ação real de controle de acesso biométrico.

O backend do dashboard é FastAPI, servido localmente pelo `uvicorn`. O frontend usa HTML, CSS e JavaScript puros para manter o MVP leve e fácil de inspecionar.

Para encerrar corretamente:

1. Clique em `Stop Camera` no dashboard.
2. Volte ao terminal onde o `uvicorn` está rodando.
3. Pressione `Ctrl+C`.

No Windows, se uma câmera ou stream RTSP bloquear o encerramento, pressione `Ctrl+Break` no terminal. Fechar o terminal deve ser a última opção.

## Visualização

- Caixa verde: `confirmed`
- Caixa amarela: `candidate`
- Caixa vermelha: `unknown`

Cada caixa mostra:

- track ID
- nome exibido
- status
- score médio
- quantidade de votos

## Verificar GPU

```powershell
python teste.py
```

Se a GPU estiver disponível, `CUDAExecutionProvider` deve aparecer. Se não aparecer, o MVP ainda pode rodar em CPU, mas o FPS provavelmente será menor.

## Privacidade E Segurança

- Não faça commit de imagens reais de referência.
- Não faça commit de credenciais RTSP reais.
- Não use este MVP para tomar decisões reais de controle de acesso biométrico.
- Colete consentimento dos participantes antes dos testes.
- Trate logs e embeddings como dados sensíveis relacionados à biometria.

O `.gitignore` mantém `config.json`, imagens de referência, logs, dados de runtime, `.venv` e `__pycache__` fora do git, preservando os arquivos `.gitkeep` para manter a estrutura das pastas.

## Autor

[Joao Lucas Oliveira](https://www.linkedin.com/in/joaodosdados/)
