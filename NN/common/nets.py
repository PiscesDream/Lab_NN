import numpy as np
from theano.sandbox.cuda.var import CudaNdarraySharedVariable

class NetModel(object):
    def __init__(self):
        pass

    def save_params(self, filename, verbose=0):
        '''
            WARNING: Regradless of structure, this method won't store the structure (shapes, types).
        '''
        params = self.__dict__.get('params', [])
        if params == []:
            raise Exception('Error: No parameters to store')
        if len(params) != len(set(params)):
            raise Exception('Error: Duplicated parameter names detected')

        data = {}
        for param in params:
            if verbose: print 'saving {} ...'.format(param.name)  
            if isinstance(param, CudaNdarraySharedVariable):
                data[param.name] = param.get_value()
        np.savez_compressed(open(filename, 'w'), **data)
        if verbose: print 'Done.'


    def load_params(self, filename, verbose=0): 
        '''
            WARNING: Regradless of structure, this method won't check the structure (shapes, types) before loading.
            The parameters of current model must have the same name and types with the loaded file.
        '''
        params = self.__dict__.get('params', [])
        if params == []:
            raise Exception('Error: No parameters to load')
        if len(params) != len(set(params)):
            raise Exception('Error: Duplicated parameter names detected')

        data = np.load(open(filename, 'r'))
        for param in params:
            if verbose: print 'loading{} ...'.format(param.name)  
            if isinstance(param, CudaNdarraySharedVariable):
                param.set_value(data[param.name])
        if verbose: print 'Done.'




