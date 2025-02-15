class VLM:
    def __init__(self, model_name=""):
        self.model_name = model_name
        self.conversation = []

    def ask(self, frame):
        # throw NotImplementedError("Should be implemented in subclasses")
        raise NotImplementedError("Should be implemented in subclasses")
