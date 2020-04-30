# Testing Tool for Throughput and Latency

## Quick Start 

You can start using the tool to test a serving model easily. 

First, you need to implement two Python classes: `BaseDataWrapper` and `BaseModelInspector`.

- `BaseDataWrapper`: used for wrapping the testing data, implement that if you need to define a custom data preprocessing method.
- `BaseModelInspector`: inspector class, you can implement it and define the `infer` method addcording to your serving platform.

## Sample

implement a custom BaseDataWrapper:

```python
from metric import BaseDataWrapper, BaseModelInspector

class TestDataWrapper(BaseDataWrapper):
    '''
    implement the BaseDataWrapper and define the preprocessing method by yourself.
    the preprocessing time will be recorded.
    '''
    def __init__(self, meta_data_url, raw_data, batch_size=None):
        super().__init__(meta_data_url, raw_data, batch_size)

    def data_preprocess(self):
        # do something...
        return self.precessed_data 

```

implement a custom BaseModelInspector:

```python

class TestModelInspector(BaseModelInspector):
    def __init__(self, data_wrapper:BaseDataWrapper, threads=None, asynchronous=None, percentile=None, sla=None):
        super().__init__(data_wrapper=data_wrapper, threads=threads, asynchronous=asynchronous, percentile=percentile, sla=sla)

    def server_batch_request(self):
        '''
        should batch the data from server side, leave blank if don't.
        '''
        pass

    def setup_inference(self):
        '''
        setup inference method, you can setup some requests here, implemented from parent class.
        '''

    def infer(self, input_batch):
        '''
        inference method, implemented from parent class.
        '''
        pass
```

After passing the TestDataWrapper instance to a TestModelInspector instance, you can call `run_model()` to start testing, the results will be recorded by the `hysia.util.Logger` class.

```python
    testDataWrapper = TestDataWrapper(meta_data_url=meta_data_url, raw_data=fake_image_data, batch_size=16) # set batch size here.
    testModelInspector = TestModelInspector(testDataWrapper)
    testModelInspector.run_model() 
```

For a full sample code, please refer to [here](./sample.py).

## Metrics

By running once, you can get some performance informations.

- a latency list contains every batch's latency.
- a through list contains every batch's throughput.
- overall latency
- overall throughput
- 25th-percentile latiency
- 50th-percentile latiency
- 75th-percentile latiency

And we do also support a custom percentile, you can setup that by passing a `percentile` in the `BaseModelInspector` constructor. 