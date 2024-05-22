import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        pass
    def forward(self,prediction_tensor, label_tensor):
        res=[]
        prediction_tensor[prediction_tensor==0]=np.finfo(np.float64).eps
        for i in range(prediction_tensor.shape[0]):
            q=prediction_tensor[i]
            p=label_tensor[i]
            loss= p*np.log(q)
            res.append(-np.sum(loss))

        return np.array(res)
    def backward(self,label_tensor):
        
        return