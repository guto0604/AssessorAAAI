from .runnables import Runnable


class ToolRunnable(Runnable):
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, input_value, config=None):
        return self.fn(input_value)


def tool(name=None):
    def decorator(fn):
        return ToolRunnable(fn)
    return decorator
