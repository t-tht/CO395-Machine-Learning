The structure and design of this part are as suggested in the specification.

The Preprocessor class by default scales features to the range [0,2]. This can be 
changed in the initializer by specifying the high and low values.

For example, scaling to the range [0, 5] would be 

preproc = Preprocessor(dataset, high=5, low=0)
