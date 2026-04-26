# Contexto De Retomada Do Projeto

Este documento serve como handoff para retomar o projeto em outro dia com um agente de código ou outro desenvolvedor.

O projeto atual é um MVP de controle de acesso biométrico baseado em reconhecimento facial. Ele ainda não deve acionar portas, catracas, relés ou qualquer mecanismo físico. O objetivo atual é avaliação em campo, estabilidade de reconhecimento, monitoramento visual e geração de evidências para análise.

## Estado Atual

Branch principal:

```text
master
```

Arquitetura atual:

```text
face_detection.py     modo clássico com janela OpenCV
dashboard.py          backend FastAPI para dashboard web
src/runtime.py        motor compartilhado de reconhecimento
src/model.py          carregamento InsightFace/ONNX Runtime
src/references.py     carregamento e cache de embeddings
src/recognition.py    similaridade e matching
src/tracking.py       tracking leve por distância entre centros
src/quality.py        filtros de tamanho, blur e confiança
src/logging_utils.py  CSV de eventos e snapshots unknown
src/video.py          webcam/RTSP
web/                  frontend HTML/CSS/JS
```

Como executar modo OpenCV:

```powershell
python face_detection.py
```

Como executar dashboard:

```powershell
uvicorn dashboard:app --host 127.0.0.1 --port 8000
```

URL local:

```text
http://127.0.0.1:8000
```

## Comportamento Atual Do MVP

- Usa InsightFace com modelo `buffalo_l`.
- Usa OpenCV para captura e visualização.
- Suporta webcam e RTSP.
- Converte BGR para RGB antes da inferência.
- Mantém BGR para exibição.
- Usa cache de embeddings em `data/embeddings_cache.pkl`.
- Usa voting temporal para evitar decisão por frame único.
- Usa tracking leve entre frames.
- Salva eventos em `logs/recognition_events.csv`.
- Salva snapshots somente para alertas `unknown`.
- Dashboard mostra:
  - câmera ao vivo
  - FPS
  - modo GPU/CPU
  - pessoas cadastradas
  - reconhecimentos confirmados
  - alertas `unknown` com snapshot
  - tabela de eventos recentes

Arquivos sensíveis/local-only:

```text
config.json
img/references/*
logs/*
data/*
.venv/
__pycache__/
```

Esses arquivos devem continuar fora do Git.

## Decisão Sobre C/C++

Não há necessidade de reescrever o projeto em C/C++ neste momento.

Motivo:

- A inferência pesada já roda em bibliotecas otimizadas por baixo:
  - ONNX Runtime
  - CUDA
  - OpenCV
  - NumPy
  - InsightFace
- O Python atua principalmente como orquestrador.
- Antes de qualquer reescrita, devemos medir gargalos reais com profiling.

Quando considerar C/C++:

- Muitas câmeras simultâneas.
- Latência crítica comprovada por profiling.
- Hardware embarcado com recursos limitados.
- Integração profunda com SDK nativo de câmera/catraca.
- Parte específica do pipeline provada como gargalo Python.

Recomendação atual:

```text
Continuar em Python + ONNX Runtime GPU + OpenCV.
Otimizar por medição, não por suposição.
```

## Prioridades Para Evoluir Rumo À Produção

### 1. Separar Inferência E Dashboard

Hoje o dashboard inicia o runtime em background no mesmo processo FastAPI.

Para produção, considerar:

- Processo 1: câmera/inferência/tracking/logs.
- Processo 2: API/dashboard.
- Comunicação por:
  - Redis
  - SQLite/Postgres
  - fila local
  - WebSocket
  - arquivos/event stream temporário

Objetivo:

- Evitar que travamentos de câmera afetem o dashboard.
- Permitir reiniciar inferência sem derrubar interface.
- Facilitar múltiplas câmeras no futuro.

### 2. Persistência Real

Hoje eventos são CSV.

Próximo passo recomendado:

- SQLite para MVP avançado.
- Postgres para produção multiusuário/multicâmera.

Tabelas sugeridas:

```text
people
reference_images
recognition_events
unknown_alerts
camera_sources
runtime_metrics
```

Benefícios:

- Consulta histórica melhor.
- Auditoria.
- Relatórios.
- Filtros por data/status/pessoa.
- Retenção controlada.

### 3. Reconexão Automática De Câmera/RTSP

O sistema precisa lidar com:

- câmera offline
- queda de rede
- RTSP travado
- timeout de leitura
- credencial incorreta

Implementar:

- timeout de leitura
- tentativas de reconexão com backoff
- métrica `camera_status`
- alerta visual no dashboard
- log quando a câmera cai e quando volta

### 4. Observabilidade

Adicionar logs estruturados e health checks.

Métricas úteis:

- FPS médio
- latência por frame
- tempo de inferência
- número de faces detectadas
- número de tracks ativos
- uso de GPU/CPU, se viável
- câmera online/offline
- último frame recebido
- último evento registrado

Endpoints sugeridos:

```text
GET /health
GET /api/metrics
GET /api/events
GET /api/camera/status
```

### 5. Segurança Do Dashboard

Antes de uso real, adicionar:

- autenticação
- senha por ambiente
- sessão simples
- proteção das rotas `/snapshots`
- não expor dashboard fora da rede local sem segurança

Configurações sensíveis devem sair do JSON local e ir para:

```text
.env
variáveis de ambiente
secret manager, se houver infraestrutura
```

### 6. Política De Retenção De Dados

Snapshots e logs são dados sensíveis relacionados à biometria.

Definir:

- por quanto tempo guardar snapshots unknown
- por quanto tempo guardar logs
- quem pode acessar
- como excluir dados antigos
- como anonimizar eventos de teste

Implementar job de limpeza:

```text
delete snapshots older than N days
archive logs older than N days
```

### 7. Gestão De Pessoas Cadastradas

Hoje as referências são pastas em `img/references/{person_name}/`.

Próximo passo:

- página no dashboard para pessoas cadastradas
- upload de imagens
- remover pessoa
- visualizar imagens de referência
- rebuild de cache pelo dashboard
- validação de qualidade das imagens enviadas

Essa feature deve preservar privacidade e evitar commit de imagens reais.

### 8. Melhorar Alertas Unknown

Estado atual:

- snapshots só para `unknown`
- painel lateral mostra somente alertas `unknown`

Melhorias possíveis:

- agrupar múltiplos unknown do mesmo track
- evitar spam de snapshots
- botão para marcar alerta como revisado
- classificação manual posterior:
  - conhecido
  - falso positivo
  - visitante
  - ignorar
- adicionar observação manual

### 9. Configuração Pelo Dashboard

Hoje a configuração principal vem de `config.json`.

Para produção/MVP avançado:

- painel read-only primeiro
- depois edição controlada

Configurações candidatas:

- fonte de vídeo
- RTSP URL
- `process_every_n_frames`
- `recognition_window_frames`
- `min_votes_to_confirm`
- `min_average_score`
- `min_face_width`
- `min_face_height`
- `min_blur_score`
- `max_center_distance`

Evitar editar pelo dashboard inicialmente:

- caminhos CUDA/cuDNN
- configurações sensíveis sem autenticação

### 10. Empacotamento E Execução Como Serviço

Para rodar em ambiente real, evitar depender de terminal manual.

Opções:

- serviço Windows
- Docker
- script PowerShell de inicialização
- NSSM para Windows service
- systemd se for Linux

Também documentar:

- instalação
- atualização
- reinício
- onde ficam logs
- como parar com segurança

## Ordem Recomendada De Implementação

Sugestão de roadmap:

1. Reconexão automática RTSP e status da câmera.
2. SQLite para eventos e unknown alerts.
3. Health checks e métricas.
4. Autenticação básica no dashboard.
5. Retenção automática de logs/snapshots.
6. Página de revisão de unknown alerts.
7. Página de pessoas cadastradas.
8. Upload/rebuild de referências pelo dashboard.
9. Separar processo de inferência e dashboard.
10. Empacotar como serviço.

## Cuidados Importantes

- Não transformar este MVP em controle de acesso real sem camadas adicionais de segurança.
- Não acionar hardware físico ainda.
- Não versionar `config.json` real.
- Não versionar imagens reais.
- Não versionar logs/snapshots reais.
- Sempre testar com consentimento dos participantes.
- Validar taxas de falso positivo e falso negativo em campo.
- Medir performance antes de otimizar.

## Pontos De Atenção Técnicos

- `Ctrl+C` pode travar no Windows se RTSP/câmera estiver bloqueando leitura.
- Usar `Stop Camera` no dashboard antes de encerrar `uvicorn`.
- Se necessário, usar `Ctrl+Break`.
- `config.example.json` deve permanecer sem credenciais reais.
- Ao alterar imagens de referência, usar `force_rebuild: true` temporariamente.
- O cache pode mascarar mudanças se os arquivos não mudarem de metadata.

## Próximo Agente Deve Começar Por

1. Ler `README.md`.
2. Ler este arquivo.
3. Conferir `git status --short --branch`.
4. Rodar:

```powershell
python -m compileall face_detection.py dashboard.py src
```

5. Testar dashboard:

```powershell
uvicorn dashboard:app --host 127.0.0.1 --port 8000
```

6. Confirmar que dados sensíveis seguem ignorados:

```powershell
git status --ignored --short
```

## Estado Desejado Antes De Produção Real

Antes de qualquer uso real em controle de acesso:

- autenticação habilitada
- RTSP com reconexão
- banco de dados
- política de retenção
- logs auditáveis
- snapshots protegidos
- revisão humana de unknown
- testes documentados com participantes
- métricas de acurácia e latência
- nenhuma ação física automatizada sem validação formal
