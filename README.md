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
