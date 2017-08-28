"""
The MIT License (MIT)

Copyright (c) 2017 Eduardo Henrique Vieira dos Santos

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import copy

class AnnyBee(object):
    """docstring for AnnBP"""
    def __init__(self, arg):
        super(AnnyBee, self).__init__()
        self.arg = arg
        self.synapses = self.randomSynapses(arg[0])
        self.deservedInputs = arg[1]
        self.deservedOutputs = arg[2]
        self.min_error = arg[3]
    
    def save(name, ob):
        #save object
        pickle_out = open(name, "wb")
        pickle.dump(ob, pickle_out)
        pickle_out.close()

    def load(name):
        pickle_in = open(name, "rb")
        z = pickle.load(pickle_in)
        pickle_in.close()
        return z

    def activationFunction(self, x, d=False):
        if(d):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    def activateNet(self, inputList = None, synapsesList = None):
        if inputList is None:
            inputList = self.deservedInputs
        if synapsesList is None:
            synapsesList = self.synapses
        np.random.seed(1)
        l = []
        for i in xrange(len(synapsesList)+1):
            if i == 0:
                l.append(inputList)
            else:
                l.append(self.activationFunction(np.dot(l[i-1],synapsesList[i-1])))
        return l

    def randomSynapses(self, listOfLayerHeights):
        np.random.seed(1)
        synapses = []
        for i in xrange(len(listOfLayerHeights)):
            if i != 0:
                synapses.append(2*np.random.random((listOfLayerHeights[i-1],listOfLayerHeights[i])) - 1)
        return synapses

    def rateError(self, layersActivations, deservedOutputs = None, synapsesList = None):
        if deservedOutputs is None:
            deservedOutputs = copy.deepcopy(self.deservedOutputs)
        if synapsesList is None:
            synapsesList = copy.deepcopy(self.synapses)
        error = []
        delta = []
        layersActivations.reverse()
        synapsesList.reverse()
        for i in xrange(len(synapsesList)):
            if i == 0:
                error.append(deservedOutputs - layersActivations[i])
                delta.append(error[0]*self.activationFunction(layersActivations[0],True))
            else:
                error.append(delta[i-1].dot(synapsesList[i-1].T))
                delta.append(error[i]*self.activationFunction(layersActivations[i],True))
        layersActivations.reverse()
        synapsesList.reverse()
        error.reverse()
        delta.reverse()
        return delta, error

    def assignDelta(self, synapsesList, layersActivations, deltaList):
        for i in xrange(len(synapsesList)):
            synapsesList[i] += layersActivations[i].T.dot(deltaList[i])

    def learnBP(self, synapsesList = None, inputList = None, deservedOutputs = None):
        if synapsesList is None:
            synapsesList = self.synapses
        if inputList is None:
            inputList = self.deservedInputs
        if deservedOutputs is None:
            deservedOutputs = self.deservedOutputs
        loop = True
        j = 0
        while loop:
            layersActivations = self.activateNet(inputList, synapsesList)
            delta = self.rateError(layersActivations, deservedOutputs,synapsesList)
            self.assignDelta(synapsesList, layersActivations, delta[0])
            error = np.mean(np.abs(delta[-1][1]))
            if (j% 10000) == 0:
                print "Error:" + str(error)
                j = 0
            if (self.min_error > error):
                loop = False
            j = j + 1
"""
#Input
xInput = np.array([[ 0,0 ],
            [   0,1 ],
            [   1,0 ],
            [   1,1 ]])
#Output          
yTargetOutput = np.array([[  0,1 ],
			[    1,0 ],
			[    1,0 ],
			[    0,1 ]])

MinimalError = 0.001

#Inicialization args[]:
#[len(inputs), len(hidden1), len(hidden2)... len(output)
#Input numpy.array()
#Output numpy.array()
#Minimal error
ann = AnnyBee([[2,3,3,2],xInput,yTargetOutput,MinimalError])

#Learn
ann.learnBP()

#Show output
print ann.activateNet()[-1]
"""