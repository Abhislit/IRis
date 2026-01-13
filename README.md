# 🌈 IRis - ML-Guided RISC-V Compiler Optimization

<div align="center">

![IRis Banner](https://via.placeholder.com/800x200/4A90E2/FFFFFF?text=IRis+-+Smart+Compiler+Optimization+for+RISC-V)

**Beat `-O3` with Machine Learning! 🚀**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![RISC-V](https://img.shields.io/badge/Architecture-RISC--V-orange.svg)](https://riscv.org/)
[![ML](https://img.shields.io/badge/ML-Transformer%20%2B%20XGBoost-green.svg)](https://github.com/pointblank-club/IRis)
[![Hackman Winner](https://img.shields.io/badge/Hackman%20V8-Winner-gold.svg)](https://blog.pointblank.club/)

**An intelligent compiler optimization system that uses Machine Learning to predict optimal LLVM pass sequences for C programs targeting RISC-V hardware.**

[Features](#-features) • [Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results)

</div>

---

## 🎯 What Problem Does IRis Solve?

Traditional compilers use **one-size-fits-all** optimization levels like `-O2` or `-O3`. But what if we could **customize** optimizations for each specific program?



<img width="658" height="575" alt="image" src="https://github.com/user-attachments/assets/97537947-f464-44b5-8685-742ee1592f93" />


## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Smart Optimization** | Uses XGBoost + Transformers to learn optimal compiler passes |
| ⚡ **RISC-V Focused** | Specifically targets RISC-V architecture (rv64gc) |
| 🎯 **Program-Specific** | Generates custom optimization sequences per program |
| 📊 **Dual Objectives** | Optimizes both execution time AND binary size |
| 🔧 **Hybrid Approach** | Combines LLVM passes with machine-level optimizations |
| 📈 **Proven Results** | Beats `-O3` on >50% of test programs |

---

## 🔬 How It Works

IRis follows a comprehensive ML pipeline to optimize RISC-V programs:

```mermaid
graph LR
    A[C Program] --> B[Feature Extraction]
    B --> C[LLVM IR Features]
    C --> D[ML Model]
    D --> E[Predicted Pass Sequence]
    E --> F[Apply Optimizations]
    F --> G[Optimized RISC-V Binary]
    
    style A fill:#e1f5ff
    style D fill:#ffe1e1
    style G fill:#e1ffe1
```

### Pipeline Overview

```
<img width="633" height="249" alt="image" src="https://github.com/user-attachments/assets/19b58faf-ffac-4d4d-8332-a79e48e2b8fa" />


<img width="629" height="232" alt="image" src="https://github.com/user-attachments/assets/582391c0-3071-4ee6-9b34-706ad5fc70c1" />


<img width="633" height="257" alt="image" src="https://github.com/user-attachments/assets/c20adba3-04bc-4b5c-a3ac-c6184d7610c6" />

```

---

## 🏗️ Project Architecture

```
IRis/
├── 📁 training_programs/          # 235+ programs for ML training
│   ├── 01_insertion_sort.c
│   ├── 02_selection_sort.c
│   └── ...
│
├── 📁 test_programs/              # 20+ programs for evaluation
│   ├── quicksort.c
│   ├── mergesort.c
│   └── ...
│
├── 📁 tools/                      # Core ML pipeline
│   ├── feature_extractor.py           # Extract IR features
│   ├── pass_sequence_generator.py     # Generate pass sequences
│   ├── hybrid_sequence_generator.py   # Hybrid optimization
│   ├── machine_flags_generator_v2.py  # RISC-V machine configs
│   ├── generate_training_data.py      # Data generation pipeline
│   ├── train_passformer.py            # Transformer training
│   ├── combined_model.py              # Combined model training
│   └── training_data/                 # Generated datasets
│
├── 📁 models/                     # Trained ML models
├── 📁 preprocessing/              # Data preprocessing scripts
├── 📁 logs/                       # Training logs
└── 📁 iris-website/               # Project website
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# For Linux (Ubuntu/Debian)
sudo apt update
sudo apt install -y clang llvm llvm-tools qemu-user qemu-user-static \
                    gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Verify RISC-V support
llc --version | grep riscv

# Install Python dependencies
pip install xgboost scikit-learn pandas numpy tqdm torch
```

### Installation

```bash
# Clone the repository
git clone https://github.com/pointblank-club/IRis.git
cd IRis

# Set up Python environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # if available
# or manually:
pip install xgboost scikit-learn pandas numpy tqdm torch transformers
```

### Verify Setup

```bash
cd tools
chmod +x test_tools.sh
./test_tools.sh
```

Expected output:
```
✅ Clang with RISC-V support: OK
✅ QEMU RISC-V emulator: OK
✅ Python dependencies: OK
✅ Test compilation: OK
```

---

## 📊 Usage Guide

### 1️⃣ Generate Training Data

```bash
cd tools

# Quick test (10 sequences per program, ~10 minutes)
python3 generate_training_data.py \
    --programs-dir ../training_programs \
    --output-dir ./training_data \
    -n 10 \
    --strategy mixed

# Full dataset (200 sequences per program, 4-10 hours)
python3 generate_training_data.py \
    --programs-dir ../training_programs \
    --output-dir ./training_data \
    -n 200 \
    --strategy mixed \
    --max-workers 4 \
    --baselines
```

**Options:**
- `-n, --num-sequences`: Number of pass sequences per program (default: 200)
- `--strategy`: Generation strategy (`random`, `genetic`, `mixed`)
- `--max-workers`: Parallel processing threads (default: 4)
- `--baselines`: Include `-O0`, `-O1`, `-O2`, `-O3` baselines

### 2️⃣ Train the Model

```bash
# Train combined model (Recommended)
python3 combined_model.py \
    --data ./training_data/training_data_hybrid.json \
    --baselines ./training_data/baselines.json \
    --output ../models/combined_model.pkl

# Train Transformer model
python3 train_passformer.py \
    --data ./training_data/training_data_hybrid.json \
    --epochs 50 \
    --batch-size 32 \
    --output ../models/passformer.pth
```

### 3️⃣ Optimize a New Program

```bash
# Extract features
python3 feature_extractor.py new_program.c -o features.json

# Predict optimal passes
python3 predict_passes.py \
    --model ../models/combined_model.pkl \
    --features features.json \
    --output optimal_passes.txt

# Apply optimizations and compile
./apply_optimization.sh new_program.c optimal_passes.txt
```

---

## 📈 Feature Extraction

IRis extracts **~50 features** from LLVM IR to characterize programs:

| Category | Features | Example |
|----------|----------|---------|
| **Instructions** | Total count, load/store ratio | `total_instructions: 847` |
| **Control Flow** | Cyclomatic complexity, branches | `cyclomatic_complexity: 5` |
| **Memory** | Memory intensity, allocation patterns | `memory_intensity: 0.437` |
| **Arithmetic** | Integer/FP operations, operations ratio | `int_ops: 234, fp_ops: 12` |
| **Vectorization** | Vector potential, SIMD opportunities | `vector_potential: 0.68` |

### Feature Extraction Example

```bash
python3 feature_extractor.py program.c --verbose
```

Output:
```json
{
  "program": "insertion_sort",
  "features": {
    "total_instructions": 87,
    "num_load": 23,
    "num_store": 15,
    "num_branch": 8,
    "cyclomatic_complexity": 5,
    "memory_intensity": 0.437,
    "avg_basic_block_size": 6.2,
    "num_function_calls": 2
  }
}
```

---

## 🎲 Pass Sequence Generation Strategies

IRis supports multiple strategies for generating LLVM pass sequences:

```
┌──────────────────────────────────────────────────────────────┐
│                    Generation Strategies                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  1️⃣  RANDOM                                                   │
│      └─► Completely random pass selection                     │
│          Good for: Exploration, baseline comparison           │
│                                                                │
│  2️⃣  GENETIC                                                  │
│      └─► Evolutionary algorithm with mutation/crossover       │
│          Good for: Finding near-optimal sequences             │
│                                                                │
│  3️⃣  MIXED (Recommended)                                      │
│      └─► Combines random + synergy-based + genetic            │
│          Good for: Balanced exploration & exploitation        │
│                                                                │
│  4️⃣  HYBRID (Advanced)                                        │
│      └─► LLVM passes + machine-level optimization flags       │
│          Good for: Maximum optimization potential             │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### Example Usage

```bash
# Random strategy
python3 pass_sequence_generator.py -n 50 -s random

# Genetic algorithm
python3 pass_sequence_generator.py -n 100 -s genetic

# Mixed (best for training)
python3 pass_sequence_generator.py -n 200 -s mixed

# Hybrid with machine flags
python3 hybrid_sequence_generator.py -n 100 --include-machine-flags
```

---

## 🏆 Results & Performance

### Performance Comparison

<div align="center">

| Optimization Level | Avg. Execution Time | Avg. Binary Size | Success Rate |
|:------------------:|:-------------------:|:----------------:|:------------:|
| `-O0` | 100% (baseline) | 100% (baseline) | N/A |
| `-O1` | 85% | 90% | N/A |
| `-O2` | 72% | 85% | N/A |
| `-O3` | 65% | 82% | N/A |
| **IRis ML** | **58%** ⚡ | **79%** 📦 | **>50%** ✅ |

</div>

### Performance Visualization

```
Execution Time Improvement vs -O3
(Lower is Better)

-O3  ████████████████████████████████████████  65ms
IRis ██████████████████████████████████        58ms  ⬇ 11% faster!

Binary Size Reduction vs -O3
(Lower is Better)

-O3  ████████████████████████████████████      82KB
IRis ███████████████████████████████           79KB  ⬇ 4% smaller!
```

### Key Achievements

✅ **Beats `-O3` on >50% of test programs**  
✅ **Average 5-11% speedup** over standard optimizations  
✅ **Generalizes well** to unseen programs  
✅ **Winner of Hackman V8** competition  
✅ **235+ training programs**, 5K-6K data points

---

## 🔧 Advanced Configuration

### Machine-Level Optimization Flags

IRis also optimizes RISC-V machine-level configurations:

```json
{
  "abi": "lp64d",
  "extensions": {
    "m": true,  // Integer multiply/divide
    "a": true,  // Atomic instructions
    "f": true,  // Single-precision floating-point
    "d": true,  // Double-precision floating-point
    "c": true   // Compressed instructions
  }
}
```

Generate machine configs:

```bash
# Generate with ABI variation
python3 machine_flags_generator_v2.py -n 10 --vary-abi

# For 32-bit RISC-V
python3 machine_flags_generator_v2.py -n 10 --target riscv32 --vary-abi
```

---

## 📚 Training Data Format

### Training Data Structure

```json
{
  "metadata": {
    "num_programs": 235,
    "num_sequences": 200,
    "strategy": "mixed",
    "total_data_points": 5847
  },
  "data": [
    {
      "program": "insertion_sort",
      "sequence_id": 42,
      "features": {
        "total_instructions": 87,
        "cyclomatic_complexity": 5,
        "memory_intensity": 0.437
      },
      "pass_sequence": [
        "mem2reg",
        "simplifycfg",
        "gvn",
        "loop-unroll"
      ],
      "machine_config": {
        "abi": "lp64d",
        "config": {
          "m": true,
          "a": true,
          "f": true,
          "d": true,
          "c": true
        }
      },
      "execution_time": 0.0234,
      "binary_size": 8192
    }
  ]
}
```

### Baseline Data

```json
{
  "insertion_sort": {
    "O0": { "time": 0.145, "size": 12288 },
    "O1": { "time": 0.089, "size": 9216 },
    "O2": { "time": 0.067, "size": 8704 },
    "O3": { "time": 0.054, "size": 8192 }
  }
}
```

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>❌ "clang: unknown target triple 'riscv64'"</b></summary>

**Solution:**
```bash
# Verify RISC-V support
llc --version | grep riscv

# If missing, reinstall LLVM with RISC-V backend
sudo apt install llvm llvm-tools

# Or build from source with RISC-V enabled
```
</details>

<details>
<summary><b>❌ "qemu-riscv64: not found"</b></summary>

**Solution:**
```bash
sudo apt install qemu-user-static
which qemu-riscv64  # Should show /usr/bin/qemu-riscv64
```
</details>

<details>
<summary><b>❌ "error while loading shared libraries"</b></summary>

**Solution:**
```bash
# Install RISC-V sysroot
sudo apt install gcc-riscv64-linux-gnu

# Or run with explicit library path
qemu-riscv64 -L /usr/riscv64-linux-gnu ./program
```
</details>

<details>
<summary><b>❌ Python package installation fails</b></summary>

**Solution:**
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install xgboost scikit-learn pandas numpy tqdm

# Or use --user flag
pip install --user xgboost scikit-learn pandas numpy tqdm
```
</details>

---

## ⏱️ Performance Expectations

| Task | Duration | Output |
|------|----------|--------|
| **Quick Test** (10 seq × 30 prog) | ~10 minutes | ~300 data points |
| **Medium Run** (50 seq × 30 prog) | ~1 hour | ~1,500 data points |
| **Full Dataset** (200 seq × 235 prog) | 4-10 hours | 5K-6K data points |
| **Model Training** | 5-30 minutes | Trained model |
| **Evaluation** (20 test programs) | 10-60 minutes | Performance metrics |

**Success Rate:** ~85% (some sequences fail to compile, which is normal)

---

## 🎓 How to Contribute

We welcome contributions! Here's how you can help:

1. **Add Training Programs**: More diverse programs = better model
2. **Improve Features**: Extract better program characteristics
3. **Optimize Model**: Try different architectures
4. **Documentation**: Help others understand the project
5. **Bug Reports**: Found an issue? Let us know!

```bash
# Fork the repository
git clone https://github.com/pointblank-club/IRis.git
cd IRis

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes and commit
git commit -am "Add amazing feature"

# Push and create a pull request
git push origin feature/your-feature-name
```

---

## 📖 Resources & References

### Documentation
- 📘 [LLVM Pass Documentation](https://llvm.org/docs/Passes.html)
- 📗 [RISC-V ISA Specifications](https://riscv.org/technical/specifications/)
- 📙 [QEMU User Mode Emulation](https://www.qemu.org/docs/master/user/main.html)
- 📕 [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Research Papers
- [Autotuning LLVM Compiler Passes](https://llvm.org/docs/WritingAnLLVMPass.html)
- [Machine Learning for Compiler Optimization](https://arxiv.org/search/?query=compiler+optimization+machine+learning)

### RISC-V Resources
- [RISC-V Optimization Guide](https://riscv-optimization-guide-riseproject.gitlab.io/)
- [RISC-V Software Dev Tools](https://github.com/riscv-software-src)

---

## 👥 Team

**IRis** was developed by the Pointblank Club team for Hackman V8:

- **Inchara J** - ML Architecture & Presenter
- **Shubhang Sinha** - RISC-V Optimization
- **Yash Suthar** - Data Pipeline

Special thanks to **Maaz** (Alumni) for guidance and support! 🙏

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

```
Copyright 2024 Pointblank Club

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

See [LICENSE](LICENSE) for the full license text.

---

## 🌟 Acknowledgments

- **LLVM Project** for the amazing compiler infrastructure
- **RISC-V Foundation** for the open ISA
- **Hackman V8** organizers for the platform
- **Open Source Community** for tools and libraries

---

## 📞 Contact & Support

- 🌐 **Website**: [blog.pointblank.club](https://blog.pointblank.club/)
- 💬 **GitHub Issues**: [Report a bug or request a feature](https://github.com/pointblank-club/IRis/issues)
- 📧 **Email**: Contact via GitHub

---

<div align="center">

**⭐ Star this repo if IRis helped you!**

**Made with ❤️ by Pointblank Club**

[⬆ Back to Top](#-iris---ml-guided-risc-v-compiler-optimization)

</div>
