import unittest
from datetime import date

import duckdb

from app import run_duckdb_query, sanitize_duckdb_sql, validate_read_only_sql


class DuckDBQueryRunnerTests(unittest.TestCase):
    def test_required_selects_execute(self):
        """Valida o comportamento esperado deste fluxo por meio de um teste automatizado.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        queries = [
            "SELECT Cliente_ID, Nome, Renda_Mensal_Estimada FROM Informacoes_Cliente ORDER BY Renda_Mensal_Estimada DESC LIMIT 10",
            "SELECT Nome FROM Informacoes_Cliente WHERE EXTRACT(MONTH FROM Data_Nascimento) = EXTRACT(MONTH FROM CURRENT_DATE) LIMIT 10",
            "SELECT COUNT(*) FROM Informacoes_Cliente",
            "SELECT * FROM Informacoes_Cliente LIMIT 5",
        ]
        for sql in queries:
            df = run_duckdb_query(sql)
            self.assertIsNotNone(df)

    def test_strftime_now_is_normalized(self):
        """Valida o comportamento esperado deste fluxo por meio de um teste automatizado.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        sql = "SELECT Nome FROM Informacoes_Cliente WHERE strftime('%m', Data_Nascimento) = strftime('%m', 'now') LIMIT 10;"
        normalized = sanitize_duckdb_sql(sql)
        self.assertIn("EXTRACT(MONTH FROM Data_Nascimento)", normalized)
        self.assertIn(f"EXTRACT(MONTH FROM DATE '{date.today().isoformat()}')", normalized)
        df = run_duckdb_query(sql)
        self.assertIsNotNone(df)

    def test_blocks_non_read_only_sql(self):
        """Trata consultas e resultados de dados para o fluxo Talk to Data com segurança.

        Returns:
            Consulta validada ou resultado tabular da execução, conforme a etapa.
        
        """
        with self.assertRaises(ValueError):
            validate_read_only_sql("DROP TABLE Informacoes_Cliente")

        with self.assertRaises(ValueError):
            validate_read_only_sql("SELECT * FROM Informacoes_Cliente; SELECT 1")

    def test_optional_existing_connection(self):
        """Valida o comportamento esperado deste fluxo por meio de um teste automatizado.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        with duckdb.connect(database=":memory:") as con:
            df = run_duckdb_query("SELECT COUNT(*) AS total FROM Informacoes_Cliente", conn=con)
        self.assertIn("total", df.columns)


if __name__ == "__main__":
    unittest.main()
