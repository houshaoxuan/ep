import time
import torch

start_time = time.time()
# 训练循环...
end_time = time.time()
print(f"训练时间: {end_time - start_time}秒")

# 内存使用
if torch.cuda.is_available():
    print(f"GPU内存使用量: {torch.cuda.memory_allocated() / 1024**2} MB")