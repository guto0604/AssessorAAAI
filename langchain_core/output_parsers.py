from .runnables import Runnable


class StrOutputParser(Runnable):
    def invoke(self, input_value, config=None):
        """Invoke.

        Args:
            input_value: Descrição do parâmetro `input_value`.
            config: Descrição do parâmetro `config`.

        Returns:
            Valor de retorno da função.
        """
        if isinstance(input_value, str):
            return input_value
        return getattr(input_value, "content", str(input_value))
