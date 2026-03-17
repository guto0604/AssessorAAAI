from __future__ import annotations

from dataclasses import dataclass


class Runnable:
    def invoke(self, input_value, config=None):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            input_value: Valor de entrada necessário para processar 'input_value'.
            config: Valor de entrada necessário para processar 'config'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        raise NotImplementedError

    def __or__(self, other):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            other: Valor de entrada necessário para processar 'other'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        return RunnableSequence([self, other])


class RunnableSequence(Runnable):
    def __init__(self, steps):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            steps: Valor de entrada necessário para processar 'steps'.
        """
        self.steps = []
        for step in steps:
            if isinstance(step, RunnableSequence):
                self.steps.extend(step.steps)
            else:
                self.steps.append(step)

    def invoke(self, input_value, config=None):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            input_value: Valor de entrada necessário para processar 'input_value'.
            config: Valor de entrada necessário para processar 'config'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        x = input_value
        for step in self.steps:
            x = step.invoke(x, config=config)
        return x


class RunnableLambda(Runnable):
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


class RunnablePassthrough(Runnable):
    def invoke(self, input_value, config=None):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            input_value: Valor de entrada necessário para processar 'input_value'.
            config: Valor de entrada necessário para processar 'config'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        return input_value


class RunnableParallel(Runnable):
    def __init__(self, **kwargs):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            kwargs: Parâmetros adicionais repassados para a chamada interna.
        """
        self.mapping = kwargs

    def invoke(self, input_value, config=None):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            input_value: Valor de entrada necessário para processar 'input_value'.
            config: Valor de entrada necessário para processar 'config'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        out = {}
        for k, r in self.mapping.items():
            out[k] = r.invoke(input_value, config=config)
        return out


@dataclass
class RunnableConfig:
    run_name: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None
