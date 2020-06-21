# torch_converter

Utilities for converting PyTorch models saved in Python 2 to models compatible with Python 3.

There aren't many PyTorch 2 models available online and the format they were pickled in can vary greatly.
As a result, the functions are defined individually rather than as a single function to convert a model.

# Example Usage

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        ...
```

```python
model = torch2to3.load_model('./models/alexnet.pth')
model.keys() # dict_keys(['optimizer', 'epoch', 'state_dict', 'best_prec1'])
state_dict = torch2to3.byte_convert(model['state_dict'])
torch2to3.save_model(AlexNet, state_dict)
```