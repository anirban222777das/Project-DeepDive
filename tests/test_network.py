import torch
from src.network import SeaquestCNN

def test_tensor_shapes_and_forward_pass():
    model = SeaquestCNN(input_channels=4, num_actions=18)
    
    # Test batch size 1
    # State from preprocessor is uint8
    batch_1 = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
    out_1 = model(batch_1)
    
    assert out_1.shape == (1, 18), f"Expected shape (1, 18), got {out_1.shape}"
    assert out_1.dtype == torch.float32, f"Expected output dtype float32, got {out_1.dtype}"
    print("Test Batch Size 1: Passed")

    # Test batch size 8
    batch_8 = torch.ones((8, 4, 84, 84), dtype=torch.uint8) * 128
    out_8 = model(batch_8)
    
    assert out_8.shape == (8, 18), f"Expected shape (8, 18), got {out_8.shape}"
    assert out_8.dtype == torch.float32
    print("Test Batch Size 8: Passed")
    
    # Test batch size 32
    batch_32 = torch.randint(0, 255, (32, 4, 84, 84), dtype=torch.uint8)
    out_32 = model(batch_32)
    
    assert out_32.shape == (32, 18), f"Expected shape (32, 18), got {out_32.shape}"
    print("Test Batch Size 32: Passed")

def test_intermediate_dimensions():
    model = SeaquestCNN()
    
    # We will manually pass a tensor through the layers to verify intermediate shapes
    x = torch.zeros((1, 4, 84, 84), dtype=torch.float32)
    
    x1 = model.conv1(x)
    assert x1.shape == (1, 32, 20, 20), f"Conv1 expected (1, 32, 20, 20), got {x1.shape}"
    
    x2 = model.conv2(x1)
    assert x2.shape == (1, 64, 9, 9), f"Conv2 expected (1, 64, 9, 9), got {x2.shape}"
    
    x3 = model.conv3(x2)
    assert x3.shape == (1, 64, 7, 7), f"Conv3 expected (1, 64, 7, 7), got {x3.shape}"
    
    x_flat = x3.reshape(x3.size(0), -1)
    assert x_flat.shape == (1, 3136), f"Flatten expected (1, 3136), got {x_flat.shape}"
    
    x_fc1 = model.fc1(x_flat)
    assert x_fc1.shape == (1, 512), f"FC1 expected (1, 512), got {x_fc1.shape}"
    
    x_out = model.fc2(x_fc1)
    assert x_out.shape == (1, 18), f"Output expected (1, 18), got {x_out.shape}"
    
    print("Test Intermediate Dimensions: Passed")

def test_device_support():
    model = SeaquestCNN()
    x = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
    
    # Test CPU
    model.to("cpu")
    out_cpu = model(x.to("cpu"))
    assert out_cpu.shape == (1, 18)
    print("Test CPU Device: Passed")
    
    # Test MPS if available
    if torch.backends.mps.is_available():
        model.to("mps")
        out_mps = model(x.to("mps"))
        assert out_mps.shape == (1, 18)
        print("Test MPS Device: Passed")
    else:
        print("Test MPS Device: Skipped (MPS not available)")

def run_all_tests():
    print("Running Network Unit Tests...")
    test_tensor_shapes_and_forward_pass()
    test_intermediate_dimensions()
    test_device_support()
    print("All network tests passed successfully!\n")

if __name__ == "__main__":
    run_all_tests()
