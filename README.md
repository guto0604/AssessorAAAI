# AssessorAAAI

## Configuração

Crie um `.env` com:

```bash
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=poc_datamasters
# opcional
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

## Observabilidade (LangSmith)

Os fluxos de **PITCH** e **REUNIÃO** usam run config com `run_name`, `tags` e `metadata` para facilitar o rastreamento.

### Como validar no LangSmith

1. Configure as variáveis acima.
2. Execute o app:
   ```bash
   streamlit run app.py
   ```
3. Rode um fluxo completo de Pitch (passos 1, 4, 5, 7 e opcionalmente 8).
4. Rode um fluxo de Reunião (upload de áudio e geração de resumo).
5. Na UI do LangSmith, abra o projeto `poc_datamasters` e confira:
   - Run pai de cada fluxo (`pitch_step_*` e `meeting_end_to_end`);
   - Child runs de prompt/model/parser/tools;
   - Tags e metadata (`feature`, `step`, `parent_run_id`).

## Testes locais

```bash
python -m unittest discover -s tests -p 'test_*.py'
```


## Rodando com Docker

A imagem foi adaptada para usar **uv** no lugar de `pip` na etapa de instalação de dependências, o que normalmente reduz bastante o tempo de build em comparação ao `pip install -r requirements.txt`.

### Pré-requisitos

- Docker Desktop (Windows/Mac) ou Docker Engine + Docker Compose (Linux).
- Arquivo `.env` configurado na raiz do projeto (mesmas variáveis da seção de configuração).

### Build e execução (Docker Compose)

```bash
docker compose up --build
```

O Streamlit ficará disponível em `http://localhost:8501`.

> Observação: o `docker-compose.yml` está em modo mais portátil (sem volume bind por padrão), para reproduzir melhor o ambiente entre máquinas.

### Execução em modo detached

```bash
docker compose up --build -d
```

Para parar:

```bash
docker compose down
```

### Build e execução (somente Docker)

```bash
docker build -t assessor-aaai:latest .
docker run --rm -p 8501:8501 --env-file .env assessor-aaai:latest
```


### Desenvolvimento com hot reload (opcional)

Se quiser refletir mudanças do código local imediatamente no container, rode com bind mount manual:

```bash
docker run --rm -p 8501:8501 --env-file .env -v ${PWD}:/app assessor-aaai:latest
```

No PowerShell (Windows), use `${PWD}` normalmente; no CMD, substitua por `%cd%`.
