from .runnables import Runnable


class StrOutputParser(Runnable):
    def invoke(self, input_value, config=None):
        if isinstance(input_value, str):
            return input_value
        return getattr(input_value, "content", str(input_value))
