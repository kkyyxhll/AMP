# 分布式训练 && 混合精度训练
## 混合精度训练
### 混合精度训练原理简介
具体理论知识在https://ocn20gn5ewwp.feishu.cn/wiki/TnsiwvljGi2Tonk6W5tcLdiNn80?from=from_copylink  
通常训练使用固定数据类型`float32`, 混合精度训练即引入`float16`。
精度越高，能表示的数值范围越大，
- float32: 1位符号位， 8位指数位， 23位尾数位
- float16: 1位符号位， 5位指数位， 10位尾数位
浮点数存在两个问题  
1. 数值下溢
2. 舍入误差  

因此，
一次训练迭代可以抽象为3步：
1. `forward(x)`
2. `loss.backward()`
3. `optimizer.step()`
   
训练过程中，需要存储模型参数(`parameters`)、优化器状态(`optimizer_states`),
梯度(`gradients`)和中间的激活值(用于反向梯度计算、链式法则)。  
我们期望模型精度高，因此模型参数使用`fp32`存储，而中间的激活值与`batch_size`成正比，
占内存的大头。因此`forward`阶段，转换为`fp16`计算。  
对应伪代码的`with torch.amp.autocast(device_type=device, dtype=torch.float16, ):`
`autocast`会自动转换，将矩阵运算如(`nn.Linear(), conv()`)等转为`float16`。  

Q: 为什么`backward`不包在`with autocast`中?  
A: 因此PyTorch的`backward`的数据类型由`forward`确定。  
Q: `scale`和`unscale_`的作用?  
A: `fp16`能表示的数值范围有限，梯度通常很小，很容易数值下溢。而梯度反向传播，`Loss`
放缩，等价于梯度放缩。从而将本身可能下溢的值变大，从而不再下溢。而放缩系数也不是越大越好，
因为会上溢，出现`inf`或`NaN`, 因此最优放缩系数就是不出现`inf`或`NaN`的最大放缩系数。
`scaler`会设定一个初始放缩系数，如果梯度中检测到`inf`或`NaN`，缩小1倍， 此次`step` 不可靠,不做`optimizer.step()` 。若2000轮次后没检测到
，说明放缩系数可以变大，扩大一倍。这个由`scaler.update()`来做。
Q: `scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`是什么？
A: 梯度裁减， 因为梯度放缩，直接裁减肯定不行，因此需要反放缩，并且`scaler`会记录这次`unscale_`，从而在`scaler.step(optimizer)`不会再次`unscale_`

混合精度训练伪代码
```
device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.amp.GradScaler(device=device)

with torch.amp.autocast(device_type=device, dtype=torch.float16, ):
  pred_label = model(data)
  loss = criterion(pred_label, label)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```
## 分布式训练
```
import torch.distributed as dist
import torch.utils.data.distributed

