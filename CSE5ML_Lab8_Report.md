# 🧠 CSE5ML – Lab 8 Report: From Traditional ML to Deep Learning (CNN)

## 1️⃣ Pre-Lab 8 – Traditional Machine Learning (HOG + SVM)

### 🎯 Objective
To demonstrate how classical machine learning methods handle image classification **without deep learning**, using handcrafted feature extraction and a simple classifier.  
This Pre-Lab serves as a **baseline model** to compare with later CNN results.

### ⚙️ Workflow and Explanation

| Step | Description & ML Meaning |
|------|---------------------------|
| **1. Load CIFAR-10 Dataset** | 60,000 32×32 colour images, 10 object categories (airplane, car, cat, etc.). Standard benchmark for vision models. |
| **2. Convert to Grayscale** | Simplifies data and enables the HOG descriptor (which does not support colour). |
| **3. Feature Extraction with HOG** | Histogram of Oriented Gradients describes local edge patterns and texture. Converts image into numeric descriptors representing structure. |
| **4. Flatten to 324-Dimensional Vectors** | Each image represented as a fixed-length numerical vector for ML algorithms. |
| **5. Train Linear SVM** | Learns decision boundaries separating classes. Linear kernel chosen for simplicity and computational speed. |
| **6. Evaluate Performance** | Achieved ≈ **53.6% accuracy** on CIFAR-10 test set. |

### 📊 Interpretation as Baseline
- Acts as a **benchmark** for later CNN experiments.  
- Demonstrates how far feature-engineered methods can go before deep learning.  
- Useful for illustrating differences in how models "see" data.

### ✅ Benefits
- Simple and explainable pipeline.  
- Low computational cost.  
- Helps students understand feature extraction process explicitly.

### ⚠️ Limitations
- **Manual feature design** → cannot generalize to unseen data well.  
- Loses colour and spatial structure.  
- Limited capacity for complex, nonlinear decision boundaries.

---

## 2️⃣ Lab 8 – Deep Learning with Convolutional Neural Network (CNN)

### 🎯 Objective
To build a **Convolutional Neural Network (CNN)** that learns features directly from raw pixel values.  
This experiment extends beyond Pre-Lab 8 by allowing the model to **automatically learn hierarchical representations**.

### ⚙️ Workflow and Explanation

| Step | Concept & ML Meaning |
|------|----------------------|
| **1. Load and Normalize CIFAR-10** | Scale pixel range `[0,255]` → `[0,1]` for stable gradient descent. |
| **2. One-Hot Encode Labels** | Convert class indices (0–9) to binary matrix suitable for Softmax output. |
| **3. Build CNN Architecture** | Layers: `Conv2D → Conv2D → MaxPooling → Flatten → Dense → Dropout → Softmax`. Each convolutional layer learns progressively abstract features. |
| **4. ReLU Activation** | Introduces non-linearity, enabling the network to model complex relationships. |
| **5. Dropout (0.2)** | Randomly disables 20% neurons per batch to reduce overfitting. |
| **6. Optimizer (SGD, lr=0.002, momentum=0.7)** | Gradually updates weights in direction of lowest loss while maintaining smoothness. |
| **7. Training (5 epochs)** | Basic CNN achieves ~**51%** accuracy — comparable to baseline but trainable end-to-end. |
| **8. Deeper CNN (more Conv layers, 20 epochs)** | Accuracy improves to **≈ 60–70%**, confirming deeper networks learn richer patterns. |

### 📊 Comparative Summary

| Aspect | Pre-Lab 8 (HOG + SVM) | Lab 8 (CNN) |
|--------|------------------------|--------------|
| Feature Source | Handcrafted (HOG) | Automatically learned |
| Non-Linearity | Low | High (ReLU + deep layers) |
| Spatial Awareness | None | Strong (Convolution & Pooling) |
| Accuracy | ~53% | ~60–70% |
| Compute Cost | Low (CPU) | High (GPU recommended) |
| Interpretability | Easy | Lower |
| Scalability | Limited | Excellent |

### ✅ Benefits of CNN
- Learns features automatically (no manual extraction).  
- Captures spatial and colour relationships in images.  
- Improves accuracy and generalization.  

### ⚠️ Limitations
- Requires more data and GPU computation.  
- Sensitive to hyperparameters.  
- Can overfit if regularization is weak.

---

## 3️⃣ Why These Numbers? Understanding the Hyperparameters

These values are not arbitrary — they are **hyperparameters** that control *how* learning happens.

| Parameter | Role | Value Used | Why This Value | Typical Range |
|------------|------|-------------|----------------|----------------|
| **Learning Rate (lr)** | Step size for weight updates | 0.001–0.002 | Stable gradient descent speed for SGD on CIFAR-10 | 1e-4 – 1e-2 |
| **Momentum (β)** | Adds "inertia" to gradient updates | 0.7–0.9 | Smooths oscillation and accelerates convergence | 0.7 – 0.9 |
| **Dropout Rate** | Randomly deactivates neurons to prevent overfitting | 0.2 | Light regularization suitable for small CNN | 0.2 – 0.5 |
| **Batch Size** | Number of samples per update | 60–64 | Balances memory and training stability | 32 – 128 |
| **Epochs** | Full passes through data | 5 → 20 | Increases learning time for better convergence | 10 – 100 |

### 🧩 How They Improve Learning

- **Learning Rate:** controls update step. Too high → unstable; too low → slow learning.  
- **Momentum:** keeps gradient direction consistent → faster, smoother convergence.  
- **Dropout:** regularization by forcing the model to learn redundant representations.  
- **Batch Size:** smaller = noisier but faster updates; larger = smoother but heavier memory use.  
- **Epochs:** more epochs allow deeper convergence but risk overfitting.

---

## 4️⃣ Best Practices in Modern CNN Training

| Area | Recommended Practice | Rationale |
|------|----------------------|-----------|
| **Normalization** | Always normalize or standardize pixel values. | Prevents exploding gradients and speeds up training. |
| **Optimizer Choice** | Use **Adam** or **SGD + Momentum 0.9**. | Adam adapts learning rate automatically; SGD often generalizes better. |
| **Learning Rate Scheduling** | Apply decay or `ReduceLROnPlateau`. | Adapts LR dynamically when validation loss stalls. |
| **Regularization** | Combine Dropout (0.3–0.5) with L2 weight decay. | Improves generalization and reduces overfitting. |
| **Batch Normalization** | Add after Conv/Dense layers. | Stabilizes training and accelerates convergence. |
| **Early Stopping** | Stop when validation loss stops improving. | Avoids wasting compute and overfitting. |
| **Data Augmentation** | Random flips, rotations, crops on CIFAR-10. | Increases diversity and robustness of features. |
| **Seed Control** | Fix `tf.random.set_seed()`. | Ensures reproducible results. |

---

## 5️⃣ Key Takeaways

- **Pre-Lab 8** serves as a *baseline*, showing traditional ML’s limits with handcrafted features.  
- **Lab 8 (CNN)** automates feature learning → better performance and flexibility.  
- **Hyperparameters** such as learning rate, momentum, and dropout define how effectively models learn.  
- Combining **deep architecture + good regularization + tuned hyperparameters** leads to efficient, stable, and high-performing models.

> 💡 Deep learning replaces manual feature design with **representation learning** — the model learns what matters, while hyperparameters control how it learns.

---
*Report generated on {datetime.now().strftime("%d %B %Y")}*
