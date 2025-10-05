# 🧠 Lecture 10: Deep Reinforcement Learning (DRL)

---

## 🧩 1. Giới thiệu tổng quan
**Deep Reinforcement Learning (DRL)** là sự kết hợp giữa **Reinforcement Learning (RL)** và **Deep Learning (DL)**.  
Nó cho phép máy tính **tự học cách ra quyết định** thông qua tương tác với môi trường và phần thưởng (reward), mà không cần nhãn sẵn.

**Mục tiêu:** Tối đa hóa **tổng phần thưởng kỳ vọng** trong tương lai.

---

## 🎲 2. Các khái niệm nền tảng

### 🧮 Random Variable (Biến ngẫu nhiên)
- Biểu diễn kết quả không chắc chắn trong quá trình học.  
- Trong RL, phần thưởng, trạng thái và hành động đều có tính ngẫu nhiên.

### 📊 PMF & PDF
- **PMF (Probability Mass Function):** Xác suất của từng giá trị cụ thể (rời rạc).
- **PDF (Probability Density Function):** Xác suất của một khoảng giá trị (liên tục).

### 🎯 Random Sampling
- Là quá trình chọn ngẫu nhiên một hành động hoặc trạng thái.  
- Góp phần **exploration** (khám phá) thay vì chỉ **exploitation** (khai thác).

---

## ⚙️ 3. Cấu trúc Reinforcement Learning

### 🧠 Agent – Environment Interaction
Agent tương tác với Environment theo chu trình:
1. Agent chọn **Action (hành động)** theo **Policy (chiến lược)**.
2. Environment trả về **Reward (phần thưởng)** và **Next State (trạng thái mới)**.
3. Agent cập nhật chiến lược để học tốt hơn.

### 🧭 Policy (π)
- Xác định cách agent chọn hành động trong mỗi trạng thái.
- **Deterministic:** luôn chọn cùng một hành động.  
- **Stochastic:** chọn ngẫu nhiên dựa trên xác suất.

### 💰 Reward
- Tín hiệu đánh giá mức độ "tốt" của hành động.
- Giúp agent học **hành vi tối ưu** để nhận được phần thưởng cao nhất.

### 🔁 State Transition
- Mô tả cách mà trạng thái thay đổi sau mỗi hành động.  
- Cơ sở lý thuyết: **Markov Decision Process (MDP)** – trạng thái kế tiếp chỉ phụ thuộc vào trạng thái hiện tại.

---

## 💰 4. Value Function

### 🔹 Action-Value Function (Q(s,a))
- Đánh giá mức độ tốt khi agent chọn hành động *a* tại trạng thái *s*.
- Giúp agent chọn hành động có giá trị kỳ vọng cao nhất.

### 🔹 State-Value Function (V(s))
- Đo giá trị trung bình của trạng thái *s* dưới chiến lược hiện tại.

### 💡 Ý nghĩa
- **Q(s,a):** giúp ra quyết định.  
- **V(s):** giúp đánh giá mục độ tốt của trạng thái.

---

## 🧠 5. Deep Q-Network (DQN)

### 🔧 Vấn đề
- Q-table không hoạt động hiệu quả khi số trạng thái rất lớn.

### 💡 Giải pháp: DQN
- Dùng **Deep Neural Network** để xấp xỉ hàm Q(s,a) thay vì bảng giá trị.
- Học cách ước lượng giá trị tương lai của từng hành động.

### 📘 Temporal Difference (TD) Learning
\[ Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)] \]
- **α:** learning rate  
- **γ:** discount factor  
- **r:** reward

### ✅ Lợi ích
- Học được trong môi trường phức tạp (game, robot).  
- Tự điều chỉnh theo trải nghiệm.

### ⚠️ Hạn chế
- Cần nhiều tài nguyên tính toán.  
- Dễ bị overfitting hoặc không ổn định khi training.

---

## 🚀 6. Ứng dụng thực tế

| Lĩnh vực | Ứng dụng | Ví dụ |
|-----------|-----------|--------|
| 🎮 Game | AI chơi game | AlphaGo, Atari DQN |
| 🚗 Xe tự lái | Điều khiển, tránh vật cản | Tesla Autopilot |
| 🤖 Robotics | Robot tự học hành động | Amazon Picking Robot |
| 💬 NLP | RLHF trong ChatGPT | GPT-4 fine-tuning |
| 💹 Finance | Giao dịch tự động | Stock trading agent |

---

## 📘 7. Takeaway & Cách ghi nhớ

### 🎯 Takeaway
1. RL = học qua **thử và sai (trial & error)**.  
2. Mục tiêu: **tối đa hóa phần thưởng tương lai**.  
3. DQN = deep network xấp xỉ Q-value.  
4. TD-learning cập nhật học hiệu quả.  
5. ChatGPT và AlphaGo là minh chứng cho DRL.

### 🧠 Gợi nhớ nhanh
| Ký hiệu | Nghĩa |
|----------|----------|
| **S** | State |
| **A** | Action |
| **R** | Reward |
| **P** | Policy |
| **Q** | Action-Value Function |
| **TD** | Temporal Difference |

📎 Mẹo nhớ: *"Agent học Policy tối đa Q-value bằng TD update"*

---

## 🍰 8. *Tiramisu of Reinforcement Learning*

Ẩn dụ giúp ghi nhớ cấu trúc Reinforcement Learning như một chiếc **tiramisu chuẩn Ý**:

| Lớp bánh | Khái niệm RL | Ý nghĩa học sâu |
|-----------|--------------------|----------------|
| 🧁 **Ladyfinger (lớp nền)** | **Environment** | Nền tảng mà agent tương tác; nơi mọ i hành động và phần thưởng bắt nguồn. |
| ☕ **Coffee syrup (thấm vào ladyfinger)** | **Reward signal** | Năng lượng tương tác giúp agent biết hành động nào “ngon” hơn. |
| 🍶 **Mascarpone (lớp kem đầu)** | **Policy (π)** | Công thức pha vị, định hướng agent ra quyết định. |
| 🧁 **Ladyfinger (lần hai)** | **State–Action–Transition (MDP)** | Cấu trúc giữ cho quá trình học có thứ tự và tiếp nối. |
| 🍶 **Mascarpone (lớp kem cuối)** | **Value Function (V/Q)** | Đánh giá độ ngon trung bình – giá trị tích luũy tương lai. |
| 🍫 **Cocoa dust (lớp phủ)** | **Exploration & Randomness** | Vị đắng nhẹ cân bằng vị ngọt – tương tự như sự khám phá trong RL. |

🎯 **Kết quả cuối cùng:** Chiếc **tiramisu hoàn chỉnh** ít đặng và hài hòa giữa exploration & exploitation — chính là **Deep Q-Network** trong RL.

---

> 📚 *Tóm lại:* Reinforcement Learning giống như việc làm một chiếc tiramisu – mỗi lớp hơi đồng v

