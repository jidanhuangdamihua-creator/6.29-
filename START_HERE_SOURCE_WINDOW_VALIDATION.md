# 🎯 D4 Source Window Validation - Quick Start

## 📋 Overview

验证D4实验中4个target store（166, 155, 240, 293）在3种source窗口长度（180, 210, 300天）下的K≥3可行性。

## ⚡ Quick Start（修复环境后）

```bash
cd "/Users/ming/Desktop/复现实验/保留的复现实验修改rfe"

# 1. 修复Python环境
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow numpy

# 2. 运行验证
python validate_source_window_solidified.py

# 3. 查看结果
cat outputs/source_window_validation/source_window_validation_report.md
```

## 📁 创建的文件

### 核心脚本
- `validate_source_window_solidified.py` ⭐ **推荐**：使用固化数据，更快
- `validate_source_window_matrix.py` - 使用原始数据，更完整

### 文档
- **`SUMMARY_SOURCE_WINDOW_VALIDATION.md`** ⭐ **从这里开始**
- `README_SOURCE_WINDOW_VALIDATION.md` - 完整使用指南
- `VALIDATION_SCRIPT_READY.md` - 环境问题说明

### 工具
- `test_environment.py` - 环境诊断

## ⚠️  当前状态

**问题**：Python虚拟环境有二进制兼容性问题（pandas导入即崩溃）

**原因**：可能是ARM/x86架构不匹配或包损坏

**解决**：需要在真实Terminal中手动运行（见上方Quick Start）

**Codex限制**：无法修复系统级二进制问题

## 🎯 你会得到什么

运行后会生成矩阵，显示每个(store, window_days)组合的：
- 总候选数
- 有效候选数（满足30天完整性+窗口覆盖）
- K≥3是否可行

示例输出：
```
Store | 180天        | 210天        | 300天
------|-------------|-------------|-------------
166   | 45/38/✅    | 42/35/✅    | 38/30/✅
155   | 32/28/✅    | 30/26/✅    | 25/20/✅
240   | 28/24/✅    | 26/22/✅    | 22/18/✅
293   | 12/10/✅    | 10/8/✅     | 8/5/✅
```

## 📖 如何使用结果

1. **查看矩阵**：哪些配置满足K≥3？
2. **选择窗口长度**：
   - 优先选择对所有store都可行的窗口
   - 倾向于接近原协议的300天（如果可行）
   - 基于可行性选择，不是基于性能
3. **文档化**：在论文中说明选择理由
4. **消融实验**：将180/210/300的对比结果放入附录

## 🔑 关键原则

**窗口长度选择必须基于可行性（K≥3），而非哪个产生更好的模型表现。**

这是一个**设计的消融实验**，不是事后调参。

## 📚 详细文档

- 完整指南：`README_SOURCE_WINDOW_VALIDATION.md`
- 状态总结：`SUMMARY_SOURCE_WINDOW_VALIDATION.md`
- 环境问题：`VALIDATION_SCRIPT_READY.md`

## 🚀 下一步

1. ✅ 修复Python环境（5分钟）
2. ✅ 运行验证脚本（1-3分钟）
3. ✅ 查看结果矩阵
4. ✅ 基于可行性选择窗口长度
5. ✅ 更新D4实验配置
6. ✅ 在论文中文档化选择理由

---

**状态**：✅ 脚本就绪，⏳ 等待Terminal手动执行
**预计时间**：修复环境5分钟 + 运行脚本3分钟 = 8分钟
**价值**：系统性的窗口长度验证，可作为论文消融实验
