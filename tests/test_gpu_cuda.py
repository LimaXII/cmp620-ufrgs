import torch

# Program used to test CUDA/GPU.
# Youtube link: https://www.youtube.com/watch?v=GMSjDTU8Zlc

# Should print 'true'.
print(torch.cuda.is_available())

# Should print '0'
print(torch.cuda.current_device())

# Should print device object.
print(torch.cuda.device(0))

# Should print '1'
print(torch.cuda.device_count())

# Should print GPU NAME. (For me: NVIDIA GeForce GTX 1060 6GB)
print(torch.cuda.get_device_name())