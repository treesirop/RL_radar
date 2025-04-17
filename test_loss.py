import torch
import torch.nn as nn
from loss import RegLoss30, RegLoss60, RegLoss90, RegLoss120, MyLoss30, MyLoss60, MyLoss90, MyLoss120

# 测试数据
batch_size = 2
channels = 4
width = 350
height = 270
output_reg = torch.randn(batch_size, channels, width, height)
label_reg = torch.randn(batch_size, channels, width, height)

# 测试RegLoss系列
def test_reg_loss():
    print("Testing RegLoss time weights:")
    
    loss30 = RegLoss30()
    loss60 = RegLoss60()
    loss90 = RegLoss90()
    loss120 = RegLoss120()
    
    print(f"RegLoss30 weights: {loss30.time_weights['30']}")
    print(f"RegLoss60 weights: {loss60.time_weights['60']}")
    print(f"RegLoss90 weights: {loss90.time_weights['90']}")
    print(f"RegLoss120 weights: {loss120.time_weights['120']}")
    
    print(f"\nRegLoss30 loss: {loss30(output_reg, label_reg)}")
    print(f"RegLoss60 loss: {loss60(output_reg, label_reg)}")
    print(f"RegLoss90 loss: {loss90(output_reg, label_reg)}")
    print(f"RegLoss120 loss: {loss120(output_reg, label_reg)}")

# 测试MyLoss系列
def test_my_loss():
    print("\nTesting MyLoss time weights:")
    
    loss30 = MyLoss30()
    loss60 = MyLoss60()
    loss90 = MyLoss90()
    loss120 = MyLoss120()
    
    print(f"MyLoss30 weights: {loss30.time_weights['30']}")
    print(f"MyLoss60 weights: {loss60.time_weights['60']}")
    print(f"MyLoss90 weights: {loss90.time_weights['90']}")
    print(f"MyLoss120 weights: {loss120.time_weights['120']}")
    
    print(f"\nMyLoss30 loss: {loss30(output_reg, label_reg)}")
    print(f"MyLoss60 loss: {loss60(output_reg, label_reg)}")
    print(f"MyLoss90 loss: {loss90(output_reg, label_reg)}")
    print(f"MyLoss120 loss: {loss120(output_reg, label_reg)}")

if __name__ == "__main__":
    test_reg_loss()
    test_my_loss()