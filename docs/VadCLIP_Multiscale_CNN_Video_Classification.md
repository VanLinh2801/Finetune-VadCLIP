# Temporal CNN đa kích thước kernel cho phân loại video

## 1. Pipeline

```text
CLIP visual features
        ↓
    LGT Adapter
        ↓
Feature [512, T]: 512 input channels, T vị trí thời gian
        │
        ├─ Conv1D: 32 kernel, mỗi kernel [512, 1] → GELU → [32, T]
        ├─ Conv1D: 32 kernel, mỗi kernel [512, 3] → GELU → [32, T]
        ├─ Conv1D: 32 kernel, mỗi kernel [512, 5] → GELU → [32, T]
        └─ Conv1D: 32 kernel, mỗi kernel [512, 7] → GELU → [32, T]
           Các nhánh cùng nhận feature [512, T], stride 1, same padding
                                ↓
Ghép 4 nhóm response theo chiều channel: 32 + 32 + 32 + 32 = 128
(xếp các hàng response lại; không cộng response, không nối trục thời gian)
                                ↓
Response [128, T]: 128 output channels = response của 128 kernel
                                ↓
Max pooling theo thời gian: lấy max trên T giá trị của từng hàng
                                ↓
Vector [128]: mỗi kernel đóng góp một response cực đại
        ↓
FC: 128 → 2
        ↓
Hai logits video: Normal / Abnormal
        ↓
Softmax → Xác suất Normal / Abnormal
        ↓
Argmax → Nhãn video: Normal (0) hoặc Abnormal (1)
```

- Ví dụ: 128 kernel chia thành 4 nhóm, mỗi nhóm 32 kernel, kích thước `512×1`, `512×3`, `512×5`, `512×7`. Các con số này là cấu hình minh họa, chưa chốt cho thí nghiệm.
- Channel ở đây là chiều feature, không phải channel màu của ảnh. Mỗi vị trí có vector 512 chiều nên Conv1D nhận 512 input channels; mỗi kernel tạo một output channel.
- Mỗi kernel phủ toàn bộ 512 chiều feature và k vị trí thời gian. Tại mỗi lần trượt, phép convolution nhân trọng số với vùng `[512, k]`, cộng gộp rồi thêm bias để tạo một giá trị response. Do đó, 32 kernel tạo `[32, T]`, không phải `[32, 512, T]`.
- Các nhánh dùng stride 1 và same padding để giữ cùng chiều dài thời gian.
- Cụ thể, kernel size 1, 3, 5, 7 lần lượt thêm 0, 1, 2, 3 vị trí zero-padding ở mỗi phía trên trục thời gian, không padding chiều feature 512. Nhờ vậy, mỗi nhánh đều trả response `[32, T]`, có thể ghép thành `[128, T]` và tính score tại cùng vị trí thời gian.
- Nếu không padding, chiều dài response là `T − k + 1`, nên các nhánh có chiều dài khác nhau. Vẫn có thể phân loại video bằng cách max pooling riêng từng nhánh rồi ghép các vector; tuy nhiên, khi tính score frame phải căn lại response theo tâm kernel và xử lý các vị trí sát biên không có đủ response từ mọi nhánh.
- Max pooling giữ response lớn nhất theo thời gian của từng kernel, chỉ xét các vị trí hợp lệ và loại padding. Các kernel có thể đạt cực đại tại những thời điểm khác nhau.
- FC kết hợp 128 giá trị này để phân loại trực tiếp video. Model không tạo logit frame rồi aggregate thành logit video trong quá trình training.
- Đầu ra dự đoán cuối cùng là nhãn nhị phân Normal (0) hoặc Abnormal (1). Softmax chuyển hai logits thành xác suất; argmax chọn nhãn có xác suất lớn hơn.
- Khi training, dùng **Cross-Entropy trực tiếp trên hai logits và nhãn video**, không dùng nhãn frame. Không đưa nhãn đã argmax vào loss; Cross-Entropy đã tích hợp bước log-softmax.

`T` là số vị trí feature theo thời gian; mỗi vị trí tương ứng một nhóm frame gốc.

## 2. Hai cách thử nghiệm

### Cách 1: Dùng feature từ VadCLIP đã train

```text
CLIP features
      ↓
LGT từ checkpoint baseline tự train, frozen
      ↓
Temporal CNN đa kernel + Max pooling + FC
      ↓
Softmax → Argmax → Nhãn video Normal (0) / Abnormal (1)
```

- Nạp LGT từ checkpoint baseline tự train, không dùng checkpoint tác giả.
- Đóng băng LGT; chỉ train CNN và FC mới.
- Nhánh mới độc lập với A và C, không sử dụng score hoặc loss của hai nhánh này.
- Có thể trích xuất feature sau LGT trước khi training.

### Cách 2: Train model độc lập

```text
CLIP features
      ↓
LGT khởi tạo mới, được train
      ↓
Temporal CNN đa kernel + Max pooling + FC
      ↓
Softmax → Argmax → Nhãn video Normal (0) / Abnormal (1)
```

- Không nạp checkpoint VadCLIP đã train.
- Train đồng thời LGT, CNN và FC bằng Cross-Entropy cấp video.
- Không sử dụng nhánh A, C hoặc loss liên quan đến text.
- CLIP pretrained vẫn frozen; from scratch áp dụng cho LGT và các lớp mới.

## 3. Truy ngược vị trí evidence

Khi max pooling, lưu **giá trị max và vị trí argmax** của từng kernel.

Với kernel j, tính đóng góp cho quyết định Abnormal so với Normal:

```text
Đóng góp[j] = MaxResponse[j] × (WeightAbnormal[j] − WeightNormal[j])
```

- `MaxResponse[j]`: response cực đại của kernel j trên video.
- `WeightAbnormal[j]`, `WeightNormal[j]`: trọng số FC nối kernel đó đến nhãn Abnormal và Normal.
- Chọn những kernel có đóng góp dương lớn nhất, sau đó truy lại vị trí argmax của chúng trên response theo thời gian.

Ví dụ:

```text
Kernel 12 → đóng góp Abnormal cao → cực đại tại vị trí 40
Kernel 37 → đóng góp Abnormal cao → cực đại tại vị trí 42
```

Đây là các vị trí ứng viên abnormal, chưa phải nhãn chắc chắn. Không chỉ chọn kernel có response lớn nhất, vì response đó có thể hỗ trợ nhãn Normal.

## 4. Score liên tục cho frame và AUC

Argmax chỉ xác định một số vị trí. Để chấm điểm **mọi vị trí thời gian**, sử dụng toàn bộ response trước pooling và chính trọng số FC đã học.

```text
Response [128, T] trước pooling
              ↓
Áp dụng cùng FC cho từng vị trí
              ↓
Hai logits Normal / Abnormal tại mỗi vị trí
              ↓
Softmax → Abnormal score [T]
```

Tại vị trí t, tính hai logits:

```text
LogitAbnormal[t] = tổng_j (WeightAbnormal[j] × Response[j, t]) + BiasAbnormal
LogitNormal[t]   = tổng_j (WeightNormal[j]   × Response[j, t]) + BiasNormal
```

- `Response[j, t]`: response của kernel j tại vị trí t, sau GELU.
- Các weight và bias lấy từ cùng FC dùng để phân loại video, không thêm classifier mới.
- Tổng được tính trên toàn bộ kernel, không chỉ các kernel được chọn để minh họa evidence.

Abnormal score là xác suất Abnormal sau softmax hai logits, tương đương:

```text
Score[t] = sigmoid(LogitAbnormal[t] − LogitNormal[t])
```

- Kernel hỗ trợ Abnormal làm tăng chênh lệch logit; kernel hỗ trợ Normal làm giảm chênh lệch đó.
- Khi đánh giá, lặp score của mỗi vị trí feature cho 16 frame tương ứng, bỏ padding và kiểm tra khớp độ dài ground truth theo protocol UCF-Crime đang sử dụng.
- Tính score trên mọi video test, không chỉ các video được dự đoán Abnormal.
- Ghép score của toàn bộ frame test rồi tính **ROC-AUC/AP frame-level** với nhãn anomaly/normal của frame. Không lấy trung bình AUC từng video.

**Training vẫn chỉ dùng pooling → FC → loss video.** Việc áp dụng FC tại từng vị trí là cách suy ra score frame sau train. Vì FC được học trên các cực đại có thể nằm ở nhiều thời điểm khác nhau, chất lượng định vị cần được đánh giá riêng, không mặc định tốt theo kết quả phân loại video.
