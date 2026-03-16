# AssessorAAAI

Aplicação em **Streamlit** para apoiar assessores na jornada comercial com uso de IA.

## Objetivo do projeto

O AssessorAAAI ajuda o assessor a:
- entender melhor o cliente e sua carteira;
- criar abordagens comerciais com mais consistência;
- transformar dados e reuniões em ações práticas;
- consultar documentos internos via IA.

---

## Telas da aplicação (resumo simples)

A aplicação possui um seletor de cliente na barra lateral e abas principais:

- **🏠 Início**
  - visão geral do propósito da ferramenta e de como navegar nas funcionalidades.

- **👤 Visualização clientes**
  - painel com resumo patrimonial, objetivos, alocação, liquidez, perfil de risco e oportunidades.

- **🚀 Voz do Assessor (Pitch)**
  - fluxo guiado para montar pitch comercial:
    - intenção de contato,
    - seleção de fontes,
    - opções de narrativa,
    - geração e ajuste final do texto.

- **📝 Reuniões**
  - gravação/upload de áudio da reunião,
  - transcrição automática,
  - resumo com próximos passos,
  - histórico por cliente.

- **📊 Talk to your Data**
  - perguntas em linguagem natural para gerar SQL,
  - execução da consulta,
  - resposta explicada e visualização de dados.

- **🤖 Pergunte à IA**
  - upload de arquivos para base vetorial (knowledge base),
  - perguntas e respostas com contexto dos documentos.

- **⚙️ Configurações**
  - configuração de chaves (sessão),
  - teste de tracing (LangSmith),
  - reindexação da base vetorial.

---

## Como executar com Docker (principal)

### 1) Pré-requisitos

- Docker instalado (Docker Desktop ou Docker Engine + Compose).
- Arquivo `.env` na raiz do projeto.

Exemplo de `.env`:

```env
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=poc_datamasters
# opcional
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

> Se usar recursos de busca externa, também pode ser necessário:
>
> `TAVILY_API_KEY=...`

### 2) Subir aplicação com Docker Compose

```bash
docker compose up --build
```

A aplicação ficará disponível em:

- `http://localhost:8501`

### 3) Rodar em segundo plano (detached)

```bash
docker compose up --build -d
```

### 4) Parar aplicação

```bash
docker compose down
```

---

## Execução alternativa (sem Compose)

```bash
docker build -t assessor-aaai:latest .
docker run --rm -p 8501:8501 --env-file .env assessor-aaai:latest
```

---

## Desenvolvimento com hot reload (opcional)

Para refletir mudanças locais no container:

```bash
docker run --rm -p 8501:8501 --env-file .env -v ${PWD}:/app assessor-aaai:latest
```

No Windows CMD, troque `${PWD}` por `%cd%`.

---

## Execução local (sem Docker)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Testes

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
