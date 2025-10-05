# Neural Networks with Keras & TensorFlow — Handy Cheat Sheet

> A compact, **GitHub‑ready** summary tying together: TensorFlow vs Keras vs Python API, forward vs backward, activations (ReLU, LeakyReLU), automatic differentiation with `tf.GradientTape`, training loops, early stopping, and a small regression template.

---

## 0) Quick Mental Model (Stack)
- **TensorFlow (runtime / engine)**: the **compute engine** that does matmul, autodiff, and optimizer steps on **CPU/GPU/TPU**.  
- **Python API**: your **language** to issue instructions.  
- **Keras (`tf.keras`)**: **high‑level API** that *packages* model definition and training (`Sequential`, `Dense`, `compile`, `fit`).

```
You (idea) → Python (code)
           → Keras (high-level training loop & layers)
           → TensorFlow (autodiff + math ops)
           → Hardware (CPU/GPU/TPU)
```

---

## 1) Core Notation
- **Data / features**: \(X\in\mathbb{R}^{N\times d_{\text{in}}}\), target \(y\).
- **Parameters**: weights \(W\), bias \(b\) (learnable → `tf.Variable`).
- **Pre‑activation (affine)**: \(\boxed{z = h_{\text{prev}} W + b}\).
- **Activation (nonlinear)**: \(\boxed{h = \phi(z)}\).  
  - Without \(\phi\): stacking many affine layers still collapses to **one affine**, can’t learn nonlinear relations.

> Convention: \(h^{[0]} = X\); for layer \(l\): \(z^{[l]} = h^{[l-1]} W^{[l]} + b^{[l]}\), \(h^{[l]} = \phi^{[l]}(z^{[l]})\).  
> Regression head is usually **linear**: \(\hat y = z^{[L]}\).

---

## 2) Forward vs Backward (Clean Separation)
### Forward (compute predictions)
1. Compute \(z^{[l]} = h^{[l-1]} W^{[l]} + b^{[l]}\)  
2. Apply activation \(h^{[l]} = \phi^{[l]}(z^{[l]})\)  
3. Final prediction \(\hat y\) → compute **loss** \(\mathcal L(\hat y, y)\) (e.g., MSE/MAE)

### Backward (compute gradients & update)
- Use **Automatic Differentiation (reverse‑mode)** to obtain \(\nabla_\theta \mathcal L\) for all parameters \(\theta\).
- Optimizer update: \(\theta \leftarrow \theta - \eta \nabla_\theta \mathcal L\) (SGD/Adam, etc.).

---

## 3) Activations in Forward *and* Backward
### Why activations?
- Add **nonlinearity** so the network can approximate curved functions (e.g., \(x^2\), complex mappings).

### Gradient flow differences
- **Sigmoid**: \(\sigma'(z) = \sigma(z)(1-\sigma(z)) \le 0.25\) → gradients **shrink** across layers (vanish).  
- **tanh**: \(1 - \tanh^2(z) \le 1\), near 0 OK, but **saturates** when \(|z|\) large → **vanish**.  
- **ReLU**: \(\phi(z)=\max(0,z)\), \(\phi'(z)=1\) if \(z>0\), else 0.  
  - In the **active region** \(z>0\): activation factor in backprop is **1** → **does not shrink** the gradient.
  - Risk: **dying ReLU** when many \(z\le 0\) (zero gradients).
- **Leaky ReLU**: slope \(\alpha \approx 0.01\) for \(z<0\) so gradients don’t die completely.

### Practical helpers
- **He initialization** (for ReLU/Leaky): keeps activations/gradients well‑scaled.  
- **(Batch/Layer) Normalization**, **residual connections**, good **LR/optimizer** also stabilize training.

---

## 4) Automatic Differentiation (Autodiff) with `tf.GradientTape`
> Autodiff = the system **records** your forward ops and **applies the chain rule** backward to compute exact gradients.

Minimal scalar example:
```python
import tensorflow as tf

x = tf.Variable(3.0)     # trackable (learnable) variable
with tf.GradientTape() as tape:
    y = x**2 + 2*x + 5   # forward
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())     # 8.0  (since d/dx = 2x+2 at x=3)
```

Mini training step for a 1‑hidden‑layer ReLU regressor:
```python
import tensorflow as tf

W1 = tf.Variable(tf.random.normal([d_in, 10], stddev=0.1))
b1 = tf.Variable(tf.zeros([10]))
W2 = tf.Variable(tf.random.normal([10, 1], stddev=0.1))
b2 = tf.Variable(tf.zeros([1]))
opt = tf.keras.optimizers.Adam(1e-2)

@tf.function  # optional speedup
def train_step(X_batch, y_batch):
    with tf.GradientTape() as tape:
        h = tf.nn.relu(tf.matmul(X_batch, W1) + b1)
        y_hat = tf.matmul(h, W2) + b2
        loss = tf.reduce_mean(tf.abs(y_batch - y_hat))  # MAE
    grads = tape.gradient(loss, [W1, b1, W2, b2])
    opt.apply_gradients(zip(grads, [W1, b1, W2, b2]))
    return loss
```

Notes:
- `tf.Variable` are watched automatically; use `tape.watch(tensor)` for plain tensors.
- `GradientTape` is one‑shot by default; use `persistent=True` if you need multiple `.gradient()` calls.

---

## 5) Keras: Where is Autodiff?
Keras wraps the loop for you in `model.fit()`. Internally, it uses `tf.GradientTape()` in `Model.train_step()`:
```python
class MyModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
```

### Typical high‑level usage
```python
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential([
    Input(shape=(d_in,)),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1)  # linear head for regression
])
model.compile(optimizer="adam", loss="mae", metrics=["mse","mae"])

callbacks=[
    EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_delta=1e-4)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=200, batch_size=32,
    verbose=1, callbacks=callbacks
)
```

`verbose` options: `0` (silent), `1` (progress bar), `2` (one line per epoch), `"auto"` (smart default).

---

## 6) Data Prep & Scaling (Why it matters)
- Fit scalers **on train only**, then transform train/val/test → avoid leakage.
- Normalized inputs make \(z = xW+b\) well‑scaled → **smoother loss surface** and **healthier gradients**.

Common choices:
- **StandardScaler** (zero mean, unit variance)  
- **MinMaxScaler** (\([0,1]\) range)  
- One‑hot for categoricals; avoid encoding nominal categories as integers with order.

---

## 7) Early Stopping & “Don’t Waste Compute”
We can’t know the “best step” in advance. Use **validation‑based early stopping**:
- Track **`val_loss`**. If it doesn’t improve by `min_delta` for `patience` checks → stop and **restore best weights**.
- Combine with **`ReduceLROnPlateau`** to drop LR when plateau, potentially unlocking another improvement; stop if still stagnant.
- Optionally monitor gradient norms or parameter deltas for convergence heuristics in custom loops.

---

## 8) Tiny Regression Template (Auto MPG‑style)
```python
# X: (N, d_in) after preprocessing; y: (N, 1)
model = Sequential([
    Input(shape=(d_in,)),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1)  # linear output
])
model.compile(optimizer="adam", loss="mae")

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, min_delta=1e-4, restore_best_weights=True)
hist = model.fit(X_train, y_train, validation_split=0.1,
                 epochs=200, batch_size=32, callbacks=[es], verbose=1)

test_mae = model.evaluate(X_test, y_test, verbose=1)
y_pred = model.predict(X_test)
```

- **Loss choice**: MAE (less outlier‑sensitive) vs MSE (smooth gradients).  
- **Output layer**: linear for regression (no activation).  
- **Initialization**: Keras uses He by default for ReLU Dense (via Glorot/He variants).

---

## 9) Common Pitfalls Checklist
- ❌ Mixing NumPy ops inside `call()` → breaks autodiff graph. Use `tf.*` ops.
- ❌ Computing loss **outside** the tape context → gradients `None`.
- ❌ Forgetting to scale/one‑hot features → poor convergence.
- ❌ Too high LR → divergence; too low → crawling. Use `ReduceLROnPlateau`.
- ❌ ReLU dying (all negative z) → consider LeakyReLU/BatchNorm/He init.

---

## 10) Glossary (one‑liners)
- **Affine**: \(xW+b\) (linear + bias).  
- **ReLU**: \(\max(0,z)\); **Leaky**: slope \(\alpha\) for \(z<0\).  
- **Autodiff**: program‑level exact derivatives; TF uses reverse‑mode (backprop).  
- **`tf.GradientTape`**: records forward ops, returns gradients via chain rule.  
- **EarlyStopping**: stop when **val** no longer improves; restore best.  
- **BatchNorm/LayerNorm**: normalize activations to stabilize training.

---

### References / Further Reading
- TensorFlow 2.x Guide — `tf.GradientTape`, Keras `Model.fit`, callbacks.  
- Classic Auto MPG regression example (data cleaning, scaling, simple MLP).

> Tip: Keep this sheet in your repo `/docs/nn_keras_tf_cheatsheet.md`, and link it from your README.
