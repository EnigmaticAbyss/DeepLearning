class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.testing_phase=False  
        self.weights = None
    def forward(self, input_tensor):
 
        raise NotImplementedError("Forward method not implemented")

    def backward(self, error_tensor):

        raise NotImplementedError("Backward method not implemented")