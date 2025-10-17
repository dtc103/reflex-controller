from ..modules.utils.ring_buffer import FiFoRingBuffer
import torch

device = "cuda:0"

def main():
    buffer = FiFoRingBuffer(10, 3, 12, device)

    print(buffer.size())

    buffer.append([0], torch.ones(1, 12, device=device))
    print(buffer.size())

    buffer.append([0, 2], torch.full((2, 12), 10, device=device).float())
    print(buffer._storage)
    print(buffer.size())

    print(buffer.pop([0, 2]))

    buffer.append([1, 3], torch.rand(12, device=device))
    print(buffer._storage)

    buffer.append([1, 3], torch.rand(12, device=device))
    
    buffer.append([1, 3], torch.rand(12, device=device))

    buffer.append([1, 3], torch.full((12,), 99.0, device=device))
    print(buffer._storage)

    print(buffer.pop([1, 5]))
    assert torch.any(torch.isnan(buffer.pop([1, 5])[1])) == True
    assert torch.any(torch.isnan(buffer.pop([1, 5])[0])) == False


if __name__ == "__main__":
    main()