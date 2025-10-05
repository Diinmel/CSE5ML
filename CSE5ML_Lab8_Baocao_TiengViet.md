# 🧠 CSE5ML – Báo cáo Lab 8: Từ Học máy truyền thống đến Deep Learning (CNN)

## 1️⃣ Pre-Lab 8 – Học máy truyền thống (HOG + SVM)

### 🎯 Mục tiêu
Giới thiệu cách tiếp cận cổ điển trong phân loại ảnh **khi chưa có deep learning**, bằng cách trích xuất đặc trưng thủ công và sử dụng bộ phân loại đơn giản (SVM).  
Phần Pre-Lab này đóng vai trò là **mốc so sánh (baseline)** để đối chiếu với kết quả của CNN sau này.

### ⚙️ Quy trình và ý nghĩa

| Bước | Giải thích & Ý nghĩa trong học máy |
|------|------------------------------------|
| **1. Tải bộ dữ liệu CIFAR-10** | 60.000 ảnh màu 32×32 thuộc 10 loại (máy bay, xe hơi, mèo,...). Là bộ dữ liệu chuẩn trong thị giác máy tính. |
| **2. Chuyển ảnh sang grayscale** | Giúp đơn giản dữ liệu và tương thích với HOG (chỉ xử lý ảnh xám). |
| **3. Trích đặc trưng bằng HOG** | HOG (Histogram of Oriented Gradients) mô tả hướng cạnh và kết cấu của ảnh. Biến ảnh thành vector đặc trưng thể hiện hình dạng tổng quát. |
| **4. Chuyển thành vector 324 chiều** | Mỗi ảnh được biểu diễn bằng vector số học cố định, dùng làm đầu vào cho mô hình ML. |
| **5. Huấn luyện mô hình SVM (linear)** | Học siêu phẳng phân tách các lớp dữ liệu. Kernel tuyến tính được chọn để giảm độ phức tạp tính toán. |
| **6. Đánh giá mô hình** | Đạt độ chính xác khoảng **53.6%** trên tập test CIFAR-10. |

### 📊 Vai trò baseline
- Là **điểm chuẩn** để so sánh với CNN sau này.  
- Giúp hiểu rõ giới hạn của phương pháp trích đặc trưng thủ công.  
- Tạo nền tảng để thấy ưu điểm của deep learning.

### ✅ Lợi ích
- Đơn giản, dễ giải thích.  
- Tốc độ huấn luyện nhanh, không cần GPU.  
- Giúp sinh viên hiểu rõ quy trình trích đặc trưng thủ công.

### ⚠️ Hạn chế
- Phụ thuộc vào đặc trưng do con người thiết kế (handcrafted).  
- Mất thông tin không gian và màu sắc.  
- Không mô hình hóa được quan hệ phi tuyến giữa các lớp.

---

## 2️⃣ Lab 8 – Deep Learning với Convolutional Neural Network (CNN)

### 🎯 Mục tiêu
Xây dựng mô hình **Convolutional Neural Network (CNN)** để học đặc trưng trực tiếp từ giá trị pixel.  
Thí nghiệm này mở rộng Pre-Lab 8 bằng việc cho phép mô hình **tự động học các đặc trưng đa cấp độ**.

### ⚙️ Quy trình và ý nghĩa

| Bước | Giải thích & Ý nghĩa |
|------|----------------------|
| **1. Chuẩn hóa dữ liệu** | Biến đổi pixel từ `[0,255]` → `[0,1]` để gradient ổn định hơn. |
| **2. One-hot encoding nhãn** | Chuyển nhãn 0–9 thành vector nhị phân phục vụ đầu ra Softmax. |
| **3. Xây mô hình CNN** | `Conv2D → Conv2D → MaxPooling → Flatten → Dense → Dropout → Softmax`. Mỗi lớp học đặc trưng ngày càng phức tạp. |
| **4. Hàm kích hoạt ReLU** | Đưa tính phi tuyến vào mô hình, cho phép học các mẫu phức tạp. |
| **5. Dropout(0.2)** | Tắt ngẫu nhiên 20% neuron để tránh overfitting. |
| **6. Tối ưu hóa bằng SGD (lr=0.002, momentum=0.7)** | Cập nhật trọng số theo hướng giảm loss, momentum giúp giảm dao động. |
| **7. Huấn luyện 5 epochs** | Mạng CNN cơ bản đạt ~**51%** accuracy — gần bằng baseline nhưng có khả năng mở rộng. |
| **8. Tăng độ sâu (20 epochs)** | Mạng sâu hơn đạt **~60–70%**, thể hiện khả năng học tốt hơn. |

### 📊 So sánh tổng quan

| Khía cạnh | Pre-Lab 8 (HOG + SVM) | Lab 8 (CNN) |
|-----------|------------------------|-------------|
| Cách lấy đặc trưng | Thủ công (HOG) | Tự học từ dữ liệu |
| Phi tuyến | Thấp | Cao (ReLU + nhiều lớp) |
| Hiểu không gian ảnh | Không có | Có (Convolution + Pooling) |
| Độ chính xác | ~53% | ~60–70% |
| Tốc độ tính toán | Nhanh | Chậm, cần GPU |
| Dễ giải thích | Cao | Khó hơn |
| Khả năng mở rộng | Hạn chế | Rất tốt |

### ✅ Ưu điểm của CNN
- Học đặc trưng tự động.  
- Bắt được mối quan hệ không gian và màu sắc.  
- Tăng hiệu suất và khả năng khái quát hóa.

### ⚠️ Nhược điểm
- Cần tài nguyên lớn (GPU, thời gian).  
- Nhạy cảm với tham số huấn luyện.  
- Có thể overfit nếu thiếu regularization.

---

## 3️⃣ Vì sao chọn các giá trị này? (Giải thích Hyperparameters)

Các con số như learning rate, momentum, dropout... là **siêu tham số** điều khiển *cách mô hình học*.

| Tham số | Vai trò | Giá trị dùng | Lý do chọn | Khoảng phổ biến |
|----------|----------|---------------|-------------|----------------|
| **Learning Rate (lr)** | Bước nhảy trong gradient descent | 0.001–0.002 | Giúp gradient ổn định, phù hợp cho SGD với ảnh nhỏ | 1e-4 – 1e-2 |
| **Momentum (β)** | Giảm dao động, tăng tốc hội tụ | 0.7–0.9 | Giúp cập nhật mượt mà và ổn định | 0.7 – 0.9 |
| **Dropout Rate** | Giảm overfitting bằng cách tắt neuron ngẫu nhiên | 0.2 | Hợp lý với mạng nhỏ, chưa quá phức tạp | 0.2 – 0.5 |
| **Batch Size** | Số mẫu cập nhật mỗi lần | 60–64 | Cân bằng giữa tốc độ và ổn định | 32 – 128 |
| **Epochs** | Số vòng học toàn bộ dữ liệu | 5 → 20 | Cho phép mạng học đủ lâu để hội tụ | 10 – 100 |

### 🧩 Vai trò trong việc giúp mô hình học tốt hơn
- **Learning rate:** kiểm soát tốc độ học, tránh nhảy quá xa hoặc quá chậm.  
- **Momentum:** giữ hướng di chuyển của gradient ổn định, giảm dao động.  
- **Dropout:** giúp mạng học khái quát hơn, tránh nhớ máy.  
- **Batch size:** ảnh hưởng đến tốc độ và độ nhiễu của quá trình học.  
- **Epochs:** càng nhiều vòng lặp → mô hình học sâu hơn nhưng cần theo dõi overfitting.

---

## 4️⃣ Thực hành tốt nhất (Best Practices) cho CNN hiện nay

| Mảng | Thực hành khuyến nghị | Lý do |
|------|------------------------|-------|
| **Chuẩn hóa dữ liệu** | Luôn scale pixel về [0,1] hoặc chuẩn hóa Z-score | Giúp gradient ổn định |
| **Tối ưu hóa** | Dùng **Adam** hoặc **SGD + Momentum 0.9** | Adam hội tụ nhanh, SGD tổng quát tốt hơn |
| **Điều chỉnh learning rate** | Dùng decay hoặc `ReduceLROnPlateau` | Giúp mô hình thích nghi tốc độ học |
| **Regularization** | Kết hợp Dropout (0.3–0.5) và L2 weight decay | Giảm overfitting |
| **Batch Normalization** | Thêm sau Conv hoặc Dense | Ổn định đầu ra và tăng tốc hội tụ |
| **Early Stopping** | Dừng khi val_loss không giảm | Tránh lãng phí tài nguyên |
| **Data Augmentation** | Lật, xoay, crop ảnh CIFAR-10 | Tăng đa dạng và khả năng khái quát |
| **Random Seed** | Cố định `tf.random.set_seed()` | Tái tạo kết quả dễ dàng |

---

## 5️⃣ Kết luận

- **Pre-Lab 8** đóng vai trò **baseline**, cho thấy giới hạn của học máy truyền thống.  
- **Lab 8 (CNN)** chứng minh khả năng học đặc trưng tự động và hiệu quả vượt trội.  
- Các **hyperparameter** như learning rate, momentum, dropout giúp mô hình học nhanh, ổn định, và tránh overfitting.  
- Kết hợp giữa **kiến trúc hợp lý + điều chỉnh tham số tốt + regularization** là chìa khóa cho mô hình hiệu quả và bền vững.

> 💡 Deep Learning thay thế việc thiết kế thủ công bằng **representation learning** – mô hình tự tìm ra điều quan trọng nhất để nhận dạng.

---
*Tạo ngày 05/10/2025*
