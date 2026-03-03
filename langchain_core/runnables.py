from __future__ import annotations

from dataclasses import dataclass


class Runnable:
    def invoke(self, input_value, config=None):
        raise NotImplementedError

    def __or__(self, other):
        return RunnableSequence([self, other])


class RunnableSequence(Runnable):
    def __init__(self, steps):
        self.steps = []
        for step in steps:
            if isinstance(step, RunnableSequence):
                self.steps.extend(step.steps)
            else:
                self.steps.append(step)

    def invoke(self, input_value, config=None):
        x = input_value
        for step in self.steps:
            x = step.invoke(x, config=config)
        return x


class RunnableLambda(Runnable):
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, input_value, config=None):
        return self.fn(input_value)


class RunnablePassthrough(Runnable):
    def invoke(self, input_value, config=None):
        return input_value


class RunnableParallel(Runnable):
    def __init__(self, **kwargs):
        self.mapping = kwargs

    def invoke(self, input_value, config=None):
        out = {}
        for k, r in self.mapping.items():
            out[k] = r.invoke(input_value, config=config)
        return out


@dataclass
class RunnableConfig:
    run_name: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None
