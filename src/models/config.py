class Config(object):

    def __init__(self):
        self.semodule_reduction = 16
        self.gate_reduction = 8
        self.dense_connections = 4
        self.skip_connections = 16
        self.dropblock_size = 7
        self.save_weights = False

    def load(self, params):
        self.semodule_reduction = params.semodule_reduction
        self.gate_reduction = params.gate_reduction
        self.dense_connections = params.dense_connections
        self.skip_connections = params.skip_connections
        self.dropblock_size = params.dropblock_size


CONFIG = Config()
