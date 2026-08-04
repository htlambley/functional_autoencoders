class SamplerBase:
    def __init__(self, autoencoder, state):
        self.autoencoder = autoencoder
        self.state = state

    def sample(self, x, key):
        raise NotImplementedError()

    def fit(self, train_dataloader):
        raise NotImplementedError()