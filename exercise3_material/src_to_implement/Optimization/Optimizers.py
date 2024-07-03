# the annotation is like every row is set of features so the 10th row is the 10th group of baches



import numpy as np
import copy


class Optimizer:



    def __init__(self):
        self.regularizer = None
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
class Sgd(Optimizer):
    


    def __init__(self, learning_rate:float):
        super().__init__()
        self.learning_rate = learning_rate


    def  calculate_update(self, weight_tensor, gradient_tensor):
        originalWeight=copy.deepcopy(weight_tensor)
        weight_tensor -=  self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(originalWeight) 
        return weight_tensor
class SgdWithMomentum(Optimizer):


    def __init__(self,learning_rate,momentum_rate):
        self.learning_rate=learning_rate
        self.momentum_rate=momentum_rate
        super().__init__()


        self.vn_1=0



    def calculate_update(self,weight_tensor,gradient_tensor):

        self.vn_1= (self.momentum_rate* self.vn_1) - (self.learning_rate* gradient_tensor)
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor) 



        return weight_tensor+self.vn_1



class Adam(Optimizer):



    def __init__(self,learning_rate,mu,rho):
        super().__init__()
        self.learning_rate=learning_rate
        self.mu=mu



        self.mu_1= 1-mu
        self.rho=rho



        self.rho_1=1-rho



        self.v=0



        self.r=0



        self.k=1



    def calculate_update(self,weight_tensor,gradient_tensor):

        self.v= self.mu*self.v + self.mu_1*gradient_tensor



        self.r=self.rho*self.r + self.rho_1* gradient_tensor*gradient_tensor



        v_hat = self.v / (1 - np.power(self.mu, self.k))



        r_hat = self.r / (1 - np.power(self.rho, self.k))


        originalWeight=copy.deepcopy(weight_tensor)
        self.k += 1
        weight_tensor-= self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))
        if self.regularizer is not None:
             weight_tensor-= self.learning_rate * self.regularizer.calculate_gradient(originalWeight) 


        return weight_tensor


