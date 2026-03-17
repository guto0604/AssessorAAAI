from .runnables import Runnable


class ToolRunnable(Runnable):
    def __init__(self, fn):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            fn: Valor de entrada necessário para processar 'fn'.
        """
        self.fn = fn

    def invoke(self, input_value, config=None):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            input_value: Valor de entrada necessário para processar 'input_value'.
            config: Valor de entrada necessário para processar 'config'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        return self.fn(input_value)


def tool(name=None):
    """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

    Args:
        name: Valor de entrada necessário para processar 'name'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    def decorator(fn):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            fn: Valor de entrada necessário para processar 'fn'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        return ToolRunnable(fn)
    return decorator
