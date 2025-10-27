import torch as nn #type: ignore

class resnet:
    
    def __init__(self):
        self.w1 = nn.randn(4,6) # input ,hidden
        self.w2 = nn.randn(6,4) # hidden , hidden
        self.w3 = nn.randn(4,2) # hidden , output

    def forwardPass(self,x):
        layer1 = x @ self.w1
        activatedLayer1 = nn.tanh(layer1) # (6,6)

        layer2 = activatedLayer1 @ self.w2
        activatedLayer2 = nn.relu(layer2) #(6,4)

        output = activatedLayer2 @ self.w3 #(6,2)

        return layer1,layer2,activatedLayer1,activatedLayer2,output
    
    def backwardPass(self,layer1,layer2,activatedLayer1,activatedLayer2,output,target,x):
        deltaO = 2 *(output - target)
        deltaW3 =  activatedLayer2.T @ deltaO # diffrentiation of loss function(partially) wrt to W3
        deltaA2 = deltaO @ self.w3.T
        deltaRelu = (layer2>0).float()
        deltaW2 =  (activatedLayer1.T @ deltaA2) * deltaRelu      # diffrentiation of loss function(partially) wrt to W2
        deltaW1 =                                               # diffrentiation of loss function(partially) wrt to W1


        return deltaW3 , deltaW2,deltaW1

    def Loss(self,output,target):
        return (output - target)**2


x = nn.randn(6,4)