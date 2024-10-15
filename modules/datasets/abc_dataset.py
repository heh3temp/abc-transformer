class ABCDataset:
    def __init__(self, path: str) -> None:

        with open(path, "r") as f:
            self._corpus = list(f.read().strip())

    @property
    def corpus(self) -> str:
        return self._corpus