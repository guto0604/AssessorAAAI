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
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### 2) Subir aplicação com Docker Compose

```bash
docker compose up --build
```

A aplicação ficará disponível em:

- `http://localhost:8501`
