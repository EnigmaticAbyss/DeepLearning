# the annotation is like every row is set of features so the 10th row is the 10th group of baches


class Sgd:
    
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate
    def  calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights
