from abc import ABC, abstractmethod
#todo: add optimizer import

class Base_layer(ABC):
    def __init__(self, optimizer=None):
        self.X = []
        self.train_mode = True
        self.optimizer = optimizer # todo: make init from name
        self.grads = {}
        self.params = {}
        self.derivatives = {}
        self.name = "abstract"

        super.__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, **kwargs):
        raise NotImplementedError

    def freeze(self):
        self.train_mode = False

    def unfreeze(self):
        self.train_mode = True

    def clear_grads(self):
        assert self.train_mode, "Layer is frozen"
        self.X = []
        self.derivatives = {k: [] for k in self.derivatives}
        self.grads = {k: [] for k in self.grads}

    def update(self):
        assert self.train_mode, "Layer is frozen"
        self.optimizer.step()
        for k, v in self.grads.items():
            if k in self.params: # have grads -> able to update weights
                self.params = self.optimizer(self.params[k], v, k)
        self.clear_grads()

    def set_params(self, inp_dict):
        # todo
        pass

    def dump(self):
        # todo
        pass


class Fully_connected_layer(Base_layer):
    def __init__(self, out_n, activ_func=None, init_weights_mode=None, optimizer=None):
        super(Fully_connected_layer, self).__init__(optimizer)
        self.init_weights_mode = init_weights_mode
        self.in_n = None
        self.out_n = out_n
        self.activ_func = activ_func # todo: create mapper from name to Activation
        self.params = {"W" : None, "b" : None}
        self.is_init = False

    def _init_params(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def _core_forward(self):
        pass

    def _core_backward(self):
        pass
