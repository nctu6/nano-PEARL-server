# Auto Set Gamma 修复说明

## 问题分析

### 问题 1: Batch 1 吞吐量异常低
**原因**: 每个 batch size 测试前没有独立的预热步骤
- 第一个 batch 直接开始测试，GPU 可能还没有完全预热
- Torch 编译、CUDA 内核初始化等都在第一次运行时发生
- 只有全局的 SKIP_FIRST_STEPS=5 不够

**解决方案**: 
- 增加 `WARMUP_STEPS = 10`
- **每个 batch size 测试前**都先运行 10 步预热
- 预热步骤不计入性能统计

### 问题 2: MAT 计算错误
**原因**: MAT (Mean Accepted Tokens) 计算方式有误

原代码:
```python
# 错误：包含了 prefill 的所有 tokens
step_tokens.append(abs(num_tokens) if num_tokens < 0 else num_tokens)
mat = total_tokens / effective_steps  # 结果等于 batch_size
```

**`num_tokens` 的含义**:
- Prefill: `num_tokens = sum(len(seq) for seq in seqs)` (所有序列总长度)
- Decode: `num_tokens = -len(seqs)` (负的序列数量)

对于 batch_size=256, MAX_SEQ_LEN=256:
- Prefill 第一步: `num_tokens = 256 * 256 = 65536`
- 之后所有 decode 步: `num_tokens = -256`

原来的 MAT 计算:
```python
total_tokens = 65536 + 256*29 = 72960  # (1 prefill + 29 decode)
mat = 72960 / 30 = 2432  # 完全错误！
```

**正确的 MAT 计算**:
MAT 应该只统计 **decode 步骤**，表示平均每步每个序列生成的 token 数。

修复后:
```python
# 只统计 decode 步骤 (num_tokens < 0)
if num_tokens < 0:
    decode_step_tokens.append(abs(num_tokens))

# MAT = 总 decode tokens / decode 步数 / batch_size
mat = total_decode_tokens / num_decode_steps / batch_size
```

对于正常的 decode (非 speculative):
- 每步每个序列生成 1 token
- `total_decode_tokens = 256 * 29 = 7424` (29 decode 步)
- `mat = 7424 / 29 / 256 ≈ 1.0` ✅

### 问题 3: Final Gamma Configuration
**说明**: 这不是问题，而是设计特性！

Gamma **必须**根据 batch size 动态选择:
- Batch 1-2: Gamma = 10 (draft 很快，可以多猜几个)
- Batch 64: Gamma = 8 (draft 开始变慢)
- Batch 256: Gamma = 5 (draft 很慢，少猜一点)

**原因**:
- 小 batch: draft model 很快 → 可以生成更多候选 tokens
- 大 batch: draft model 变慢 → 减少候选数量，避免浪费计算

标题已更新为: `"Final Gamma Configuration (Dynamic per Batch Size)"`

## 预期修复后的输出

```
Auto Set Gamma: 100%|██████████| 9/9 [00:15<00:00,  1.7s/it]
====================================================================================================
Auto Set Gamma Results:
====================================================================================================
Batch:   1 | Draft: 250.00 tok/s | Target: 25.84 tok/s | Gamma: 10 | MAT:  1.00 | Throughput:   25.84 tok/s | Per-req:  25.84 tok/s
Batch:   2 | Draft: 245.00 tok/s | Target: 25.60 tok/s | Gamma: 10 | MAT:  1.00 | Throughput:   51.20 tok/s | Per-req:  25.60 tok/s
Batch:   4 | Draft: 240.00 tok/s | Target: 25.40 tok/s | Gamma:  9 | MAT:  1.00 | Throughput:  101.60 tok/s | Per-req:  25.40 tok/s
Batch:   8 | Draft: 235.00 tok/s | Target: 25.20 tok/s | Gamma:  9 | MAT:  1.00 | Throughput:  201.60 tok/s | Per-req:  25.20 tok/s
Batch:  16 | Draft: 230.00 tok/s | Target: 25.00 tok/s | Gamma:  9 | MAT:  1.00 | Throughput:  400.00 tok/s | Per-req:  25.00 tok/s
Batch:  32 | Draft: 220.00 tok/s | Target: 24.50 tok/s | Gamma:  9 | MAT:  1.00 | Throughput:  784.00 tok/s | Per-req:  24.50 tok/s
Batch:  64 | Draft: 200.00 tok/s | Target: 23.50 tok/s | Gamma:  8 | MAT:  1.00 | Throughput: 1504.00 tok/s | Per-req:  23.50 tok/s
Batch: 128 | Draft: 160.00 tok/s | Target: 21.60 tok/s | Gamma:  7 | MAT:  1.00 | Throughput: 2764.80 tok/s | Per-req:  21.60 tok/s
Batch: 256 | Draft: 115.00 tok/s | Target: 21.60 tok/s | Gamma:  5 | MAT:  1.00 | Throughput: 5529.60 tok/s | Per-req:  21.60 tok/s
====================================================================================================

Final Gamma Configuration (Dynamic per Batch Size):
--------------------------------------------------
  Batch   1 -> Gamma 10
  Batch   2 -> Gamma 10
  Batch   4 -> Gamma  9
  Batch   8 -> Gamma  9
  Batch  16 -> Gamma  9
  Batch  32 -> Gamma  9
  Batch  64 -> Gamma  8
  Batch 128 -> Gamma  7
  Batch 256 -> Gamma  5
--------------------------------------------------
```

## 关键改进

1. ✅ **每个 batch size 独立预热 10 步** → Batch 1 吞吐量会正常
2. ✅ **MAT 只统计 decode 步骤** → MAT ≈ 1.0 (非 speculative 模式)
3. ✅ **吞吐量计算更准确** → 只计算 decode tokens，排除 prefill
4. ✅ **说明 gamma 是动态的** → 用户理解这是设计特性

## 注意事项

当前代码中 MAT ≈ 1.0 是因为:
- 在 Auto Set Gamma **profiling 阶段**不使用 speculative decoding
- 只是测量 draft 和 target 的原始速度
- 实际运行时，使用 speculative decoding，MAT 应该 ≈ 7-8

如果要在 profiling 时测试真实的 MAT，需要:
1. 启用 PEARL verification
2. 运行 draft → target 的完整 speculative flow
3. 但这会让 profiling 时间更长
