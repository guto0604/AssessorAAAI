from __future__ import annotations

from dataclasses import dataclass


class Runnable:
    def invoke(self, input_value, config=None):
        """Invoke.

        Args:
            input_value: Descrição do parâmetro `input_value`.
            config: Descrição do parâmetro `config`.

        Returns:
            Valor de retorno da função.
        """
        raise NotImplementedError

    def __or__(self, other):
        """  or  .

        Args:
            other: Descrição do parâmetro `other`.

        Returns:
            Valor de retorno da função.
        """
        return RunnableSequence([self, other])


class RunnableSequence(Runnable):
    def __init__(self, steps):
        """  init  .

        Args:
            steps: Descrição do parâmetro `steps`.
        """
        self.steps = []
        for step in steps:
            if isinstance(step, RunnableSequence):
                self.steps.extend(step.steps)
            else:
                self.steps.append(step)

    def invoke(self, input_value, config=None):
        """Invoke.

        Args:
            input_value: Descrição do parâmetro `input_value`.
            config: Descrição do parâmetro `config`.

        Returns:
            Valor de retorno da função.
        """
        x = input_value
        for step in self.steps:
            x = step.invoke(x, config=config)
        return x


class RunnableLambda(Runnable):
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


class RunnablePassthrough(Runnable):
    def invoke(self, input_value, config=None):
        """Invoke.

        Args:
            input_value: Descrição do parâmetro `input_value`.
            config: Descrição do parâmetro `config`.

        Returns:
            Valor de retorno da função.
        """
        return input_value


class RunnableParallel(Runnable):
    def __init__(self, **kwargs):
        """  init  .

        Args:
            kwargs: Descrição do parâmetro `kwargs`.
        """
        self.mapping = kwargs

    def invoke(self, input_value, config=None):
        """Invoke.

        Args:
            input_value: Descrição do parâmetro `input_value`.
            config: Descrição do parâmetro `config`.

        Returns:
            Valor de retorno da função.
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
