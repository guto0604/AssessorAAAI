import unittest
from unittest.mock import patch

from core.auto_pitch import (
    build_auto_pitch_signal_summary,
    build_priority_candidates,
    generate_auto_pitch_communication,
    generate_auto_pitch_priorities,
)
from core.journey_ranker import rank_journeys
from core.source_selector import select_sources_step4
from core.pitch_structurer import build_pitch_options_step5
from core.pitch_writer import generate_final_pitch_step7, generate_prompt_to_pitch, revise_pitch_step8
from core.meetings import summarize_transcript


class MiniDF:
    def __init__(self, rows):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            rows: Valor de entrada necessário para processar 'rows'.
        """
        self.rows = rows
        self.empty = len(rows) == 0

    def iterrows(self):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        for i, r in enumerate(self.rows):
            yield i, r

    def __getitem__(self, cols):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            cols: Valor de entrada necessário para processar 'cols'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        return MiniDF([{k: r.get(k) for k in cols} for r in self.rows])

    def to_dict(self, orient="records"):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            orient: Valor de entrada necessário para processar 'orient'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        return self.rows


class _Msg:
    def __init__(self, content):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            content: Valor de entrada necessário para processar 'content'.
        """
        self.content = content


class _Choice:
    def __init__(self, content):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            content: Valor de entrada necessário para processar 'content'.
        """
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            content: Valor de entrada necessário para processar 'content'.
        """
        self.choices = [_Choice(content)]


class _Completions:
    def create(self, **kwargs):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            kwargs: Parâmetros adicionais repassados para a chamada interna.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        system = kwargs["messages"][0]["content"]
        if "ranquear jornadas" in system:
            return _Resp('{"ranking":[{"jornada_id":"J1","nome_jornada":"Teste","score":0.9}]}')
        if "seleção de fontes" in system:
            return _Resp('{"data_sources":["investimentos_do_cliente"],"products_selected_ids":["P1","P2","P3"],"kb_files_selected":["knowledge_base/documentos/explicacao_cdb.txt"],"reasoning_short":"ok"}')
        if "estrategista de conteúdo" in system:
            return _Resp('{"blocos_conteudo":[{"id":"diagnostico_alocacao","titulo":"Diagnóstico de alocação","itens":[{"id":"diagnostico_alocacao_i1","texto":"x"},{"id":"diagnostico_alocacao_i2","texto":"y"}]},{"id":"proximos_passos","titulo":"Próximos passos recomendados","itens":[{"id":"proximos_passos_i1","texto":"z"}]}],"tom_sugerido":{"principal":{"id":"t1","texto":"consultivo"},"alternativas":[{"id":"t2","texto":"direto"},{"id":"t3","texto":"leve"}]},"tamanho_pitch":{"principal":{"id":"l1","texto":"Médio"},"alternativas":[{"id":"l2","texto":"Pequeno"},{"id":"l3","texto":"Longo"}]}}')
        if "estrategista de auto-pitch" in system:
            return _Resp('{"resumo_executivo":"ok","prioridades":[{"priority_rank":1,"priority_id":"p1","categoria":"oferta_produto","titulo":"Oferta de caixa","objetivo":"Alocar caixa","porque_agora":"Há liquidez disponível","sinais_dados":["Caixa alto"],"abordagem_recomendada":"Contato consultivo","canal_recomendado":"WhatsApp","tom":"consultivo","products_selected_ids":["P1"],"kb_files_selected":["knowledge_base/produtos/explicacao_cdb.txt"]},{"priority_rank":2,"priority_id":"p2","categoria":"contato_padrão","titulo":"Check-in consultivo","objetivo":"Retomar contato","porque_agora":"Manter relacionamento","sinais_dados":["Sem follow-up"],"abordagem_recomendada":"Atualização","canal_recomendado":"Ligação","tom":"leve","products_selected_ids":[],"kb_files_selected":[]},{"priority_rank":3,"priority_id":"p3","categoria":"rebalanceamento","titulo":"Rever carteira","objetivo":"Reduzir concentração","porque_agora":"Carteira concentrada","sinais_dados":["Concentração alta"],"abordagem_recomendada":"Rebalancear","canal_recomendado":"WhatsApp","tom":"consultivo","products_selected_ids":["P2"],"kb_files_selected":[]}]}')
        if "estrategista comercial para assessoria de investimentos" in system:
            return _Resp('{"resumo_estrategico":"Prioridade escolhida","racional_argumentativo":["Bullet 1","Bullet 2"],"provas_evidencias":["Prova 1"],"mensagem_principal":"Mensagem pronta","mensagem_follow_up":"Follow-up pronto","cta":"Agendar conversa","observacoes_assessor":["Obs 1"]}')
        if "mensagem comercial final" in system:
            return _Resp("pitch direto")
        if "escrevendo uma mensagem" in system:
            return _Resp("pitch final")
        if "revisor de texto comercial" in system:
            return _Resp("pitch revisado")
        return _Resp("resumo")


class _Chat:
    completions = _Completions()


class _FakeClient:
    chat = _Chat()


class FlowTests(unittest.TestCase):
    @patch("langchain_openai.get_openai_client", return_value=_FakeClient())
    def test_pitch_langchain_steps(self, _mock_client):
        """Executa uma etapa de construção do pitch comercial personalizado para o cliente.

        Args:
            _mock_client: Valor de entrada necessário para processar '_mock_client'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        jornadas_df = MiniDF([
            {
                "Jornada_ID": "J1",
                "Nome_Jornada": "A",
                "Categoria": "C",
                "Objetivo_Principal": "O",
                "Descricao_Resumida": "D",
                "Topicos_LLM": "Diagnóstico de alocação; Próximos passos recomendados",
            }
        ])
        cliente = {"Nome": "Cli"}
        jornada = {
            "jornada_id": "J1",
            "topicos_llm": ["Diagnóstico de alocação", "Próximos passos recomendados"],
        }

        r1 = rank_journeys(cliente, "meta", jornadas_df)
        self.assertEqual(r1["ranking"][0]["jornada_id"], "J1")

        r1m = rank_journeys(cliente, "meta", jornadas_df, include_api_metrics=True)
        self.assertEqual(r1m["result"]["ranking"][0]["jornada_id"], "J1")
        self.assertIn("model", r1m["api_metrics"])

        produtos_df = MiniDF([
            {"Produto_ID": "P1", "Nome_Produto": "Prod", "Categoria": "R", "Subcategoria": "S", "Risco_Nivel (1-5)": 2, "Suitability_Ideal": "Conservador"},
            {"Produto_ID": "P2", "Nome_Produto": "Prod2", "Categoria": "R", "Subcategoria": "S", "Risco_Nivel (1-5)": 2, "Suitability_Ideal": "Conservador"},
            {"Produto_ID": "P3", "Nome_Produto": "Prod3", "Categoria": "R", "Subcategoria": "S", "Risco_Nivel (1-5)": 2, "Suitability_Ideal": "Conservador"},
        ])
        inv_df = MiniDF([{"Produto": "X", "Categoria": "Y", "Valor_Investido": 100}])
        r4 = select_sources_step4(cliente, "meta", jornada, {}, produtos_df, inv_df)
        self.assertEqual(len(r4["products_selected_ids"]), 3)

        r4m = select_sources_step4(cliente, "meta", jornada, {}, produtos_df, inv_df, include_api_metrics=True)
        self.assertEqual(len(r4m["result"]["products_selected_ids"]), 3)
        self.assertIn("total_tokens", r4m["api_metrics"])

        r5 = build_pitch_options_step5(cliente, "meta", jornada, {}, inv_df, produtos_df, [])
        self.assertIn("blocos_conteudo", r5)
        self.assertEqual(r5["blocos_conteudo"][0]["titulo"], "Diagnóstico de alocação")

        r5m = build_pitch_options_step5(cliente, "meta", jornada, {}, inv_df, produtos_df, [], include_api_metrics=True)
        self.assertIn("blocos_conteudo", r5m["result"])
        self.assertIn("latency_ms", r5m["api_metrics"])

        r7 = generate_final_pitch_step7(cliente, "meta", jornada, {"blocos_conteudo": []})
        self.assertEqual(r7, "pitch final")

        r7_direct = generate_prompt_to_pitch(cliente, "meta")
        self.assertEqual(r7_direct, "pitch direto")

        r8 = revise_pitch_step8("abc", "editar")
        self.assertEqual(r8, "pitch revisado")

        signal_summary = build_auto_pitch_signal_summary(
            {
                "Nome": "Cli",
                "Perfil_Suitability": "Conservador",
                "Patrimonio_Investido_Conosco": 1000,
                "Patrimonio_Investido_Outros": 800,
                "Dinheiro_Disponivel_Para_Investir": 200,
                "Ultima_Interacao_Dias": 60,
                "Aniversario_Proximo_Dias": 5,
            },
            {"spread_vs_cdi_12m": -0.02},
            inv_df,
        )
        self.assertEqual(signal_summary["cliente"]["nome"], "Cli")

        candidates = build_priority_candidates(signal_summary)
        self.assertTrue(len(candidates) >= 3)

        auto_priorities = generate_auto_pitch_priorities(
            cliente_info={
                "Nome": "Cli",
                "Perfil_Suitability": "Conservador",
                "Patrimonio_Investido_Conosco": 1000,
                "Patrimonio_Investido_Outros": 800,
                "Dinheiro_Disponivel_Para_Investir": 200,
                "Ultima_Interacao_Dias": 60,
                "Aniversario_Proximo_Dias": 5,
            },
            carteira_summary={"spread_vs_cdi_12m": -0.02},
            investimentos_cliente_df=inv_df,
            produtos_df=produtos_df,
            prompt_assessor="",
        )
        self.assertEqual(len(auto_priorities["prioridades"]), 3)
        self.assertEqual(auto_priorities["prioridades"][0]["priority_id"], "p1")
        self.assertNotIn("score_prioridade", auto_priorities["prioridades"][0])
        self.assertNotIn("confianca", auto_priorities["prioridades"][0])

        auto_comm = generate_auto_pitch_communication(
            cliente_info={"Nome": "Cli", "Perfil_Suitability": "Conservador"},
            carteira_summary={"spread_vs_cdi_12m": -0.02},
            investimentos_cliente_df=inv_df,
            produtos_df=produtos_df,
            selected_priority=auto_priorities["prioridades"][0],
        )
        self.assertEqual(auto_comm["mensagem_principal"], "Mensagem pronta")

    @patch("langchain_openai.get_openai_client", return_value=_FakeClient())
    def test_meeting_summary(self, _mock_client):
        """Executa uma etapa do fluxo de reuniões, incluindo registro, transcrição ou sumarização.

        Args:
            _mock_client: Valor de entrada necessário para processar '_mock_client'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        out = summarize_transcript({"Nome": "Cli"}, "texto")
        self.assertTrue(isinstance(out, str))


if __name__ == "__main__":
    unittest.main()
