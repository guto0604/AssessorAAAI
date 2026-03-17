import unittest
from unittest.mock import patch

from core.journey_ranker import rank_journeys
from core.source_selector import select_sources_step4
from core.pitch_structurer import build_pitch_options_step5
from core.pitch_writer import generate_final_pitch_step7, revise_pitch_step8
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
            return _Resp('{"diagnostico":[{"id":"d1","texto":"x"},{"id":"d2","texto":"y"},{"id":"d3","texto":"z"}],"pontos_prioritarios":[{"id":"p1","texto":"a"},{"id":"p2","texto":"b"},{"id":"p3","texto":"c"}],"gatilhos_comerciais":[{"id":"g1","texto":"g"}],"objecoes_e_respostas":[{"id":"o1","objecao":"o","resposta":"r"}],"produtos_sugeridos":[{"id":"s1","produto_id":"P1","texto":"t"}],"tom_sugerido":{"principal":{"id":"t1","texto":"consultivo"},"alternativas":[{"id":"t2","texto":"direto"},{"id":"t3","texto":"leve"}]},"tamanho_pitch":{"principal":{"id":"l1","texto":"Médio"},"alternativas":[{"id":"l2","texto":"Pequeno"},{"id":"l3","texto":"Longo"}]}}')
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
            {"Jornada_ID": "J1", "Nome_Jornada": "A", "Categoria": "C", "Objetivo_Principal": "O", "Descricao_Resumida": "D"}
        ])
        cliente = {"Nome": "Cli"}

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
        r4 = select_sources_step4(cliente, "meta", {"jornada_id": "J1"}, {}, produtos_df, inv_df)
        self.assertEqual(len(r4["products_selected_ids"]), 3)

        r4m = select_sources_step4(cliente, "meta", {"jornada_id": "J1"}, {}, produtos_df, inv_df, include_api_metrics=True)
        self.assertEqual(len(r4m["result"]["products_selected_ids"]), 3)
        self.assertIn("total_tokens", r4m["api_metrics"])

        r5 = build_pitch_options_step5(cliente, "meta", {"jornada_id": "J1"}, {}, inv_df, produtos_df, [])
        self.assertIn("diagnostico", r5)

        r5m = build_pitch_options_step5(cliente, "meta", {"jornada_id": "J1"}, {}, inv_df, produtos_df, [], include_api_metrics=True)
        self.assertIn("diagnostico", r5m["result"])
        self.assertIn("latency_ms", r5m["api_metrics"])

        r7 = generate_final_pitch_step7(cliente, "meta", {"jornada_id": "J1"}, {"diagnostico": []})
        self.assertEqual(r7, "pitch final")

        r8 = revise_pitch_step8("abc", "editar")
        self.assertEqual(r8, "pitch revisado")

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
