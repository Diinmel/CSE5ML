# Tóm tắt kiến thức NN (Bản “cuốn chả giò”) — Keras & TensorFlow

> File tổng hợp nhanh những gì mình vừa trao đổi: ví von **cuốn chả giò**, định nghĩa **z/h/affine**, tách bạch **forward vs backward**, **autodiff** với `tf.GradientTape`, **ReLU & Leaky ReLU** (và vì sao giúp giảm vanishing), vòng lặp huấn luyện, **early stopping**, `verbose`, và vài snippet mẫu.

---

## 1) Bức tranh tổng thể: TensorFlow – Python API – Keras
- **TensorFlow (runtime/engine)**: “bếp & lửa” thực thi toán nặng (nhân ma trận, autodiff, tối ưu) trên **CPU/GPU/TPU**.
- **Python API**: “ngôn ngữ gọi món” — em viết lệnh Python để mô tả mô hình, training.
- **Keras (`tf.keras`)**: “bánh tráng gói” quy trình cao cấp: `Sequential`, `Dense`, `compile`, `fit`… giúp **thử nghiệm nhanh**.

```
Ý tưởng → Python code → Keras (fit/compile) → TensorFlow (autodiff + toán) → Phần cứng
```

---

## 2) Ví von “cuốn chả giò”
- **Bánh tráng** = Keras (khung gói quy trình).
- **Nhân** = các lớp & tham số (W, b) — TF tính toán.
- **Nước mắm** = loss + metrics.
- **Gia vị** = hyperparams (LR, batch size, epochs).
- **Chiên** = `fit()` (Keras điều phối); **nhiệt/lửa** = TF trên GPU/CPU.
- **Lật mặt/canh dầu** = backprop + optimizer (SGD/Adam).

Nếu **không dùng Keras**, em vẫn nấu được (dùng `tf.Variable`, `tf.GradientTape`, `apply_gradients`…), nhưng mất công “tự gói” mọi thứ.

---

## 3) Ký hiệu cốt lõi: *z*, *h*, **affine**
- **Affine (biến đổi tuyến tính + tịnh tiến)**:  
  \[ \boxed{z = h_{\text{prev}} W + b} \quad (\text{pre-activation}) \]
- **Kích hoạt (phi tuyến)**:  
  \[ \boxed{h = \phi(z)} \quad (\text{post-activation}) \]
- Quy ước: \(h^{[0]}=X\). Ở lớp \(l\):  
  \(z^{[l]} = h^{[l-1]} W^{[l]} + b^{[l]}, \; h^{[l]} = \phi^{[l]}(z^{[l]})\).
- **Regression head** thường **linear**: \(\hat y = z^{[L]}\) (không activation).

> Nếu **không có \(\phi\)**: xếp chồng bao nhiêu affine vẫn gộp thành **một affine** ⇒ không học được quan hệ **phi tuyến**.

---

## 4) Forward vs Backward — tách bạch
**Forward**: tính dự đoán \(\hat y\) và **loss**  
1) \(z^{[l]} = h^{[l-1]} W^{[l]} + b^{[l]}\)  
2) \(h^{[l]} = \phi^{[l]}(z^{[l]})\)  
3) \(\hat y \rightarrow \mathcal L(\hat y, y)\)

**Backward** (*backprop/autodiff*): tính \(\nabla_\theta \mathcal L\) rồi **cập nhật** tham số \(\theta \leftarrow \theta - \eta \nabla_\theta \mathcal L\).

---

## 5) Activations: **ReLU** & **Leaky ReLU** (vs sigmoid/tanh)
- **Sigmoid**: \(\sigma(z)=1/(1+e^{-z})\), \(\sigma'(z)\le 0.25\) ⇒ tích nhiều lớp **teo nhanh** (*vanish*).  
- **tanh**: \(1-\tanh^2(z)\le 1\), bão hoà khi \(|z|\) lớn ⇒ cũng **vanish**.
- **ReLU**: \(\phi(z)=\max(0,z)\),  
  \[ \phi'(z) = \begin{cases} 1, & z>0 \\ 0, & z\le 0 \end{cases} \]
  - **Vùng dương** (\(z>0\)): phần “do activation” **không co** gradient (nhân với 1) ⇒ **đỡ vanish** hơn sigmoid/tanh.
  - **Rủi ro**: *dying ReLU* khi nhiều \(z\le 0\) ⇒ gradient = 0.
- **Leaky ReLU**: thêm độ dốc \(\alpha \approx 0.01\) cho \(z<0\): \(\phi'(z)=\alpha>0\) ⇒ gradient **không tắt hẳn**.

**Thực hành hỗ trợ gradient**:  
- **He initialization** (cho ReLU/Leaky): giữ biên độ tín hiệu/gradient hợp lý.  
- **(Batch/Layer) Normalization**, **Residual/Skip**, **LR/optimizer** phù hợp.

> ReLU giúp **giảm vanishing do activation** ở vùng \(z>0\), nhưng vanish/explode còn phụ thuộc **trọng số** ⇒ cần khởi tạo/chuẩn hoá tốt.

---

## 6) Autodiff: `tf.GradientTape()` — máy **quay forward** rồi **tua ngược**
- **Autodiff**: hệ thống **ghi lại** các phép toán trong forward rồi **áp chain rule** để ra gradient **chính xác (đến sai số máy)**.
- **Reverse-mode** phù hợp NN: 1 loss vô hướng, **rất nhiều tham số** ⇒ chi phí xấp xỉ **1–2 lần forward**.

Ví dụ cực nhỏ (scalar):
```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2 + 2*x + 5
dy_dx = tape.gradient(y, x)   # 8.0  (vì d/dx = 2x+2 tại x=3)
```

Mini training step:
```python
W1 = tf.Variable(tf.random.normal([d_in, 10], stddev=0.1))
b1 = tf.Variable(tf.zeros([10]))
W2 = tf.Variable(tf.random.normal([10, 1], stddev=0.1))
b2 = tf.Variable(tf.zeros([1]))
opt = tf.keras.optimizers.Adam(1e-2)

@tf.function
def train_step(Xb, yb):
    with tf.GradientTape() as tape:
        h = tf.nn.relu(tf.matmul(Xb, W1) + b1)
        y_hat = tf.matmul(h, W2) + b2
        loss = tf.reduce_mean(tf.abs(yb - y_hat))  # MAE
    grads = tape.gradient(loss, [W1, b1, W2, b2])
    opt.apply_gradients(zip(grads, [W1, b1, W2, b2]))
    return loss
```

Ghi nhớ: `tf.Variable` được tape theo dõi tự động; `tf.Tensor` thường cần `tape.watch(tensor)` nếu muốn gradient theo nó.

---

## 7) Keras dùng autodiff ở đâu?
- Trong `model.fit()`: Keras gọi **`Model.train_step()`**, bên trong dùng `tf.GradientTape()` để tính gradient và `optimizer.apply_gradients(...)` để cập nhật.  
- Có thể **tùy biến** bằng cách **override `train_step`** nếu cần điều khiển chi tiết.

---

## 8) Training loop & dừng sớm
- **EarlyStopping** (theo `val_loss`): dừng khi **không cải thiện** thêm (với `patience`, `min_delta`) và **khôi phục best weights**.
- **ReduceLROnPlateau**: hạ LR khi chững để thử “mở khoá” thêm cải thiện.
- **Giới hạn cứng**: `max_epochs/max_steps`, `max_time`.
- **Chuẩn hoá đầu vào** (Standard/MinMax) để bề mặt loss mượt hơn, gradient khoẻ hơn.

---

## 9) `verbose` trong `model.fit()`
- `0`: im lặng (không in progress; `History` vẫn có dữ liệu).  
- `1`: progress bar (mặc định, hợp notebook).  
- `2`: mỗi epoch 1 dòng (gọn trong console).  
- `"auto"`: Keras tự chọn theo môi trường.

---

## 10) Đếm “tập tham số” theo số hidden layers
- Mỗi **biến đổi affine** cần **một cặp** \((W,b)\).  
- **1 hidden layer** ⇒ 2 affine (In→H, H→Out) ⇒ **2 cặp (W,b)** ⇒ “**4 tập tham số**”.  
- **2 hidden layers** ⇒ 3 affine ⇒ **3 cặp (W,b)** ⇒ “**6 tập tham số**”.

Ví dụ (In=1, H=10, Out=1, 1 hidden):
- \(W_1:(1,10), b_1:(10)\); \(W_2:(10,1), b_2:(1)\) ⇒ **31 tham số**.

---

## 11) Mẹo ổn định gradient (tóm nhanh)
- **He init**, **Batch/LayerNorm**, **LR/optimizer** hợp lý (Adam là baseline tốt).
- **Residual connections** khi mô hình sâu.
- Tránh **NumPy** trong `call()` (làm đứt đồ thị), dùng `tf.*`.
- Theo dõi **val_loss**; overfit ⇒ regularization (L2/Dropout nhỏ).

---

## 12) Mẫu Keras cực gọn cho hồi quy tabular
```python
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential([
    Input(shape=(d_in,)),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1)  # linear head
])
model.compile(optimizer="adam", loss="mae", metrics=["mse","mae"])

cbs = [
    EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_delta=1e-4),
]
hist = model.fit(X_train, y_train, validation_split=0.1, epochs=200, batch_size=32,
                 callbacks=cbs, verbose=1)
test_mae = model.evaluate(X_test, y_test, verbose=1)
y_pred = model.predict(X_test)
```

---

## 13) Checklist lỗi hay gặp
- Loss tính **ngoài** `with tape:` → gradient `None`.
- Dùng **NumPy** bên trong `call()` → autodiff **không** theo dõi.
- Không **scale**/one-hot đúng → hội tụ kém.
- LR quá lớn/nhỏ; quên **EarlyStopping** & **ReduceLROnPlateau**.
- ReLU chết hàng loạt (nhiều \(z\le 0\)) → thử **Leaky ReLU** + He init/chuẩn hoá.

---

## 14) 5 dòng ghi nhớ
1) \(z=xW+b\) (affine), \(h=\phi(z)\) (phi tuyến).  
2) **ReLU/Leaky** giúp gradient **đỡ vanish** (ở \(z>0\) hoặc \(z<0\) có \(\alpha\)).  
3) **Autodiff** (reverse-mode) = **ghi forward**, **chain rule backward**.  
4) `model.fit()` của Keras dùng **`tf.GradientTape`** ở bên dưới.  
5) Dùng **EarlyStopping** + **ReduceLROnPlateau** để không phí compute.
