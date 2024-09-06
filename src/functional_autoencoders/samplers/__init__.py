class SamplerBase:
    def __init__(self, autoencoder, state):
        self.autoencoder = autoencoder
        self.state = state

    def sample(self, x):
        raise NotImplementedError()
