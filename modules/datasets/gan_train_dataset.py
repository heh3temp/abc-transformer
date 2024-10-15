from .train_dataset import TrainDataset


class GanTrainDataset(TrainDataset):

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.context_size]

        return chunk, chunk