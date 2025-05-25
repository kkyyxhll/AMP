# 分布式训练 && 混合精度训练
## 混合精度训练
```混合精度训练伪代码
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
