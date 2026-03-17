from .runnables import Runnable


class ToolRunnable(Runnable):
    def __init__(self, fn):
        """  init  .

        Args:
            fn: Descrição do parâmetro `fn`.
        """
        self.fn = fn

    def invoke(self, input_value, config=None):
        """Invoke.

        Args:
            input_value: Descrição do parâmetro `input_value`.
            config: Descrição do parâmetro `config`.

        Returns:
            Valor de retorno da função.
        """
        return self.fn(input_value)


def tool(name=None):
    """Tool.

    Args:
        name: Descrição do parâmetro `name`.

    Returns:
        Valor de retorno da função.
    """
    def decorator(fn):
        """Decorator.

        Args:
            fn: Descrição do parâmetro `fn`.

        Returns:
            Valor de retorno da função.
        """
        return ToolRunnable(fn)
    return decorator
