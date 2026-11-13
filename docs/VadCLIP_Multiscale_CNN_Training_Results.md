# Kết quả huấn luyện Temporal CNN đa kích thước kernel

## 1. Phương pháp

```text
CLIP visual features → LGT Adapter → Conv1D đa kích thước → GELU
       → Ghép response → Max pooling theo thời gian → FC 2 logits
       → Softmax → Nhãn video Normal / Abnormal
```

- Dùng feature CLIP 512 chiều đã trích xuất; giữ cấu trúc LGT của VadCLIP. LGT, CNN và FC được train từ đầu, không nạp checkpoint đã train.
- Mỗi nhánh CNN dùng 32 kernel, mỗi kernel có kích thước `512 × k`. Các nhánh có stride 1, same padding để giữ cùng chiều dài response, sau đó ghép theo chiều channel.
- Max pooling giữ một response cực đại theo thời gian cho mỗi kernel. FC kết hợp các giá trị này để phân loại trực tiếp video.
- Loss duy nhất là **Cross-Entropy trên hai logits video và nhãn Normal/Abnormal**. Không sử dụng nhánh alignment, text hoặc loss frame.
- Khi đánh giá frame-level, áp dụng cùng FC lên response tại từng vị trí trước pooling, lấy xác suất Abnormal sau softmax. Đây là score suy ra sau training, **chưa phải cách truy ngược vị trí argmax từ max pooling**.

## 2. Hyperparameter training

| Hyperparameter | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-5` |
| Weight decay | `0.01` |
| Scheduler | Giảm LR × `0.1` sau epoch 4 và 8 |
| Batch size | 64 Normal + 64 Abnormal |
| Số epoch | 10 |
| Seed | 234 |
| Gradient clipping | Max norm `1.0` |

## 3. Cách tính frame score để so sánh với VadCLIP

Model được train để dự đoán nhãn của cả video, trong khi VadCLIP baseline được đánh giá bằng AUC trên từng frame. Vì vậy, xây dựng thêm cách suy ra score cho từng vị trí thời gian từ response của CNN, nhằm **tính AUC frame-level và so sánh với VadCLIP baseline trên cùng tập test**. Cách tính này chỉ dùng khi đánh giá, không thay đổi loss training.

### 3.1. Lấy score từ response của các kernel

1. Lấy response **trước max pooling**: response sau GELU với cấu hình có GELU, hoặc đầu ra trực tiếp của convolution với cấu hình không GELU. Tại mỗi vị trí thời gian, có 128 hoặc 256 giá trị response, mỗi kernel đóng góp một giá trị.
2. Gom các giá trị tại cùng vị trí thành một vector. Ví dụ, với 256 kernel, mỗi vị trí có một vector 256 chiều.
3. Đưa vector đó qua **chính lớp FC đã học để phân loại video**, thu được hai logits: Normal và Abnormal.
4. Áp dụng softmax lên hai logits, lấy xác suất Abnormal làm score tại vị trí đang xét. Lặp lại cho mọi vị trí trong video.

```text
Tại một vị trí thời gian:
256 response từ 256 kernel → FC đã học → Normal logit / Abnormal logit
                                      → Softmax → Abnormal score
```

Khác biệt nằm ở đầu vào FC: khi training, FC nhận các response cực đại trên toàn video; khi tính frame score, FC nhận response của các kernel tại cùng một vị trí. Đây là cách suy ra score để đánh giá, không phải truy ngược vị trí argmax đã được max pooling chọn.

### 3.2. Công thức tính score

```text
Normal logit   = tổng (response mỗi kernel × trọng số Normal của kernel đó)
                + bias Normal

Abnormal logit = tổng (response mỗi kernel × trọng số Abnormal của kernel đó)
                + bias Abnormal

Frame score = softmax(hai logits)[Abnormal]
            = sigmoid(Abnormal logit − Normal logit)
```

Các trọng số và bias được lấy từ FC đã train, không học thêm lớp mới. Công thức dùng toàn bộ kernel tại vị trí đang xét, không chọn top-k kernel. Score nằm trong khoảng 0–1; càng cao thì model càng nghiêng về Abnormal tại vị trí đó. Tuy nhiên, đây là score suy ra từ model học nhãn video, không phải xác suất đã được học trực tiếp bằng nhãn frame.

## 4. Kết quả các lần chạy

Cả ba lượt đánh giá trên **290 video, tổng cộng 1.109.888 frame** của UCF-Crime test set. Các độ đo trong bảng tính theo %, chênh lệch tính theo điểm phần trăm.

Đánh giá tại epoch 5, 7 và 10; **epoch 10 là kết quả chính**, không chọn checkpoint theo test AUC. Các dòng ghi “bỏ GELU” chỉ bỏ GELU sau CNN, không thay đổi LGT.

- **Frame AUC/AP:** đánh giá score Abnormal trên toàn bộ frame test; score mỗi vị trí feature được lặp cho 16 frame tương ứng.
- **Ano-AUC:** frame-level AUC chỉ trên các video abnormal, vẫn gồm frame normal và anomaly trong những video đó.
- **Video AUC/AP/Accuracy:** đánh giá dự đoán Normal/Abnormal cho toàn video sau max pooling và FC.

| Lượt chạy | Epoch | Frame AUC | Frame AP | Ano-AUC | Video AUC | Video AP | Video Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **VadCLIP baseline tự train** | - | **87,72** | **33,13** | **69,94** | - | - | - |
| 128 kernel, k = 1/3/5/7, GELU | 5 | 85,64 | 26,76 | 64,62 | 96,97 | 97,22 | 91,38 |
| 128 kernel, k = 1/3/5/7, GELU | 7 | 85,65 | 26,50 | 64,59 | 96,97 | 97,23 | 91,72 |
| **128 kernel, k = 1/3/5/7, GELU** | **10** | **85,59** | **26,25** | **64,45** | **96,99** | **97,22** | **91,38** |
| 256 kernel, k = 2–9, GELU | 5 | 86,21 | 27,95 | 66,00 | 96,98 | 97,21 | 91,03 |
| 256 kernel, k = 2–9, GELU | 7 | 86,13 | 27,72 | 65,83 | 96,94 | 97,21 | 91,03 |
| **256 kernel, k = 2–9, GELU** | **10** | **86,09** | **27,70** | **65,79** | **96,93** | **97,19** | **91,38** |
| 256 kernel, k = 2–9, bỏ GELU | 5 | 86,49 | 29,21 | 66,85 | 96,88 | 97,10 | 90,34 |
| 256 kernel, k = 2–9, bỏ GELU | 7 | 86,39 | 28,73 | 66,60 | 96,92 | 97,17 | 91,38 |
| **256 kernel, k = 2–9, bỏ GELU** | **10** | **86,29** | **28,23** | **66,38** | **96,91** | **97,16** | **90,69** |

Tại epoch 10, khi cùng dùng GELU, lượt 256 kernel tăng **0,50 điểm Frame AUC**, **1,45 điểm Frame AP** và **1,34 điểm Ano-AUC** so với lượt 128 kernel. Hai lượt thay đổi đồng thời tổng số kernel và tập kernel size, nên chưa tách riêng tác dụng của từng thay đổi.

Với cùng 256 kernel và k = 2–9, bỏ GELU tăng **0,21 điểm Frame AUC**, **0,54 điểm Frame AP** và **0,59 điểm Ano-AUC** tại epoch 10. Chênh lệch được tính từ số liệu chưa làm tròn. Đây là so sánh các lượt chạy hiện có, chưa xác nhận độ ổn định qua nhiều lần chạy.

Mốc baseline tự train trong bảng lấy từ lượt đánh giá ở thí nghiệm CNN trước: **Frame AUC 87,72%, Frame AP 33,13% và Ano-AUC 69,94%**. Ba lượt lần lượt thấp hơn mốc AUC này **2,13**, **1,63** và **1,43 điểm** tại epoch 10. Các log multi-scale được cung cấp chưa có kết quả evaluate lại baseline trong cùng lượt; các chỉ số video-level của baseline chưa được ghi nhận.
