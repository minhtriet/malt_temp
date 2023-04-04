from simulai.regression import DenseNetwork

class BeerPINN:
    def __init__(self, input_labels=['x', 't'], output_labels=['u']):
        self.input_labels = input_labels
        self.output_labels = output_labels
        # Configuration for the fully-connected network
        config = {
            "layers_units": [128, 128, 128, 128],
            "activations": "tanh",
            "input_size": len(self.input_labels),
            "output_size": len(self.output_labels),
            "name": "beerPINN"
        }
        # Instantiating and training the surrogate model
        self.net = DenseNetwork(**config)

    def train(self):
        pass
