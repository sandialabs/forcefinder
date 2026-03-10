# Source Path Receiver Data Object
The `SourcePathReceiverData` object is intended to store all of the data for a `SourcePathReceiver` object. The key differences between the `SourcePathReceiverData` object and `SourcePathReceiver` object are:

- The `SourcePathReceiverData` object only requires responses to initialize the object, all the other pieces of data can be left as `None`
- There are several differences (compared to the `SourcePathReceiver` object) in how the data is store to minimize the duplicated data 
- The individual pieces of data (the responses, FRFs, and transformations) can be labeled, as shown below:

```{code-block} python
spr_data_object = ff.LinearSourcePathReceiverData(frfs = {'frf_run_1':frf_data},
                                                  target_response = {'response_run_1':response_data})
```
- The labeled data can be left as `None` (so it can be assigned later in the `SourcePathReceiverSet` object)