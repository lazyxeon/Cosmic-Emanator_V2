# 🌌 Cosmic Emanator: A Geometric Framework for Emanative AI
> “Software-first, hardware-ready – AI that thinks like the universe.”

![MIT License](https://img.shields.io/badge/license-MIT-blue)
![Built with PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-%23ee4c2c)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

The **Cosmic Emanator** is a speculative, research-grade framework for artificial intelligence inspired by **cosmic topology**, **fractal geometry**, and **harmonic physics**.

It reimagines computation as a process of *emanation* — recursive, cyclical, and fractal. Built in **PyTorch**, this software-first neural processor is extensible to exotic hardware substrates (e.g., **twisted bilayer graphene**, **optical manifolds**, or **topological photonics**).

🧠 AI with symmetry & memory  
🌐 Rooted in physical law  
🧪 Torch-ready, quantum-capable

---

## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🧬 What Is It?](#-what-is-it)
- [🧮 Mathematical Summary](#-mathematical-summary)
- [🧪 Benchmark Highlights](#-benchmark-highlights)
- [🧠 Use Cases](#-use-cases)
- [📚 Notebooks](#-notebooks)
- [🧑‍🔬 Physical Inspiration](#-physical-inspiration)
- [📜 License](#-license)
- [✨ Acknowledgments](#-acknowledgments)

---

## 🚀 Quick Start

### 📥 Clone and Install

```bash
git clone https://github.com/lazyxeon/Cosmic-Emanator_V2.git
cd Cosmic-Emanator_V2
pip install -r requirements.txt
```

### 🧪 Run the Layer

```python
from src.tfnp_layer import TFNPLayer
import torch

model = TFNPLayer(3, 64)  # RGB input, 64 output channels
input_tensor = torch.rand(1, 3, 32, 32)
output = model(input_tensor, t=1.0)
print(output.shape)  # torch.Size([1, 64, 32, 32])
```

---

## 🧬 What Is It?

The **TFNP (Topological-Fractal Neural Processor)** is a hybrid neural layer that integrates:

- 🔁 **Toroidal topology** – Twisted manifold for cyclical flow  
- 🌀 **Fibonacci scaling** – Recursive golden-ratio dilation  
- ⚛️ **Merkaba sinusoidal activation** – Harmonic spin gating  
- 🧠 **Non-local tensor modulation** – Inspired by phase geometry

It functions like a convolutional layer but routes information through **twist, pulse, and scale spaces**, emulating universal symmetry.

---

## 🧮 Mathematical Summary

The forward propagation is defined as:

```
Yₗ = sin(2π·f·t) · (Wₗ · (Xₗ₋₁ ⊗ T) + bₗ)
```

Where:
- `T = exp(i·α·(ϕᵢ - ϕⱼ))` → Twist tensor with complex phase shift  
- `α = 7/2` → Asymmetry constant  
- `ϕ ≈ 1.618` → Golden ratio (phi)  
- `t` → Time/frequency input modulation  

See [`notebooks/math_derivations.ipynb`](notebooks/math_derivations.ipynb) for symbolic derivation.

---

## 🧪 Benchmark Highlights

| Task                    | TFNP Layer                | Baseline Conv2D       |
|-------------------------|---------------------------|------------------------|
| CIFAR-10 (20% noise)    | 1.5× faster convergence    | Slower convergence     |
| MNIST + Distortion      | 92% accuracy              | 85% accuracy           |
| Output Feature Variance | 0.15                      | 0.10                   |

Benchmarks are reproducible from [`notebooks/mnist_demo.ipynb`](notebooks/mnist_demo.ipynb) and [`entropy_sim.ipynb`](notebooks/entropy_sim.ipynb).

---

## 🧠 Use Cases

- Pattern recognition under geometric distortion and noise  
- Simulations of entropy and self-organizing complexity  
- Fractal geometry layers in generative and symbolic AI  
- Foundation for topological or quantum hardware research  

---

## 📚 Notebooks

| Notebook | Description |
|----------|-------------|
| [`notebooks/mnist_demo.ipynb`](notebooks/mnist_demo.ipynb) | MNIST noise demo with TFNP layer |
| [`notebooks/entropy_sim.ipynb`](notebooks/entropy_sim.ipynb) | Spiral entropy growth simulator |
| [`notebooks/math_derivations.ipynb`](notebooks/math_derivations.ipynb) | Symbolic derivation of layer equation |

All notebooks are runnable in **Jupyter**, **Google Colab**, or **VSCode**.

---

## 🧑‍🔬 Physical Inspiration

- 🌪 **Toroidal Topology** → Recurring non-local flow  
- 🌸 **Flower of Life Geometry** → Nested symmetries  
- 🌀 **Fibonacci Spiral** → Universal scaling law  
- ✨ **Merkaba Spin Fields** → Vibrational logic  
- 🧵 **Twisted Graphene** → Hardware potential with moiré manifolds  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

> Open for research, education, and remixing.  
> Commercial hardware implementations require attribution and collaboration.

---

## ✨ Acknowledgments

Created by **Andrew R. Brown**  
Inspired by the missions of **xAI**, **Tesla**, and **SpaceX** to build systems in harmony with the true laws of the cosmos.


> *“Let the processor be an echo of the universe.”*

---

