# Kết quả Temporal CNN với L1 Regularization

## 1. Phương pháp

Model phân loại trực tiếp video Normal/Abnormal bằng Temporal CNN đa kích thước:

```text
CLIP visual features → LGT Adapter → Temporal CNN đa kích thước
        → Ghép response → Max pooling theo thời gian → FC → Hai logits video
        → Softmax → Nhãn Normal / Abnormal
```

- Dùng feature CLIP 512 chiều đã trích xuất; giữ LGT của VadCLIP. LGT, CNN và FC được train từ đầu, không nạp checkpoint đã train.
- CNN có 8 nhánh với kernel size `2, 3, 4, 5, 6, 7, 8, 9`; mỗi nhánh có 32 kernel kích thước `512 × k`, tổng cộng 256 kernel. Không dùng GELU sau CNN; các activation trong LGT giữ nguyên.
- Response của mỗi kernel được max pooling theo thời gian; 256 giá trị cực đại được đưa vào FC `256 → 2` để phân loại video.
- Các lượt L1 bổ sung penalty trên weight của FC, không phạt bias:

```text
Loss = Cross-Entropy video + λ × tổng trị tuyệt đối của FC.weight
```

`FC.weight` có kích thước `[2, 256]`. Lượt `λ = 0` chỉ dùng Cross-Entropy; ba lượt còn lại dùng `λ = 1e-5`, `3e-5` và `1e-4`. Không sử dụng nhãn frame khi training.

Để tính frame-level AUC/AP và so sánh với VadCLIP, cùng FC được áp dụng lên response **tại từng vị trí trước max pooling**; softmax của hai logits cho score Abnormal. Đây là phép suy ra score sau training, không phải truy ngược argmax của max pooling.

## 2. Cấu hình training

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

- Tập train gồm 8.000 Normal views và 8.100 Abnormal views. Mỗi epoch có 125 bước và sử dụng 16.000 views do batch cân bằng và bỏ batch thiếu.
- Evaluate tại epoch 5, 7 và 10. **Epoch 10 là kết quả chính**, không chọn checkpoint theo test AUC.
- Tập test gồm 290 video, tổng 1.109.888 frame. Frame AUC/AP tính trên toàn bộ frame test; Ano-AUC tính trên frame thuộc các video abnormal. Mỗi score vị trí feature được lặp cho 16 frame tương ứng.
- Video AUC/AP/accuracy đánh giá nhãn video sau max pooling và FC. Không có AUC2/mAP vì model không sử dụng nhánh alignment phân loại hành động.

## 3. Kết quả training

Các giá trị trong bảng là **phần trăm**. Các lượt dùng toàn bộ 256 kernel để tính frame score; chưa áp dụng kernel selection.

| Cấu hình | Epoch | Frame AUC | Frame AP | Ano-AUC | Video AUC | Video AP | Video Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| VadCLIP baseline tự train | — | 87,72 | 33,13 | 69,94 | — | — | — |
| Không L1 | 5 | 86,49 | 29,21 | 66,85 | 96,88 | 97,10 | 90,34 |
| Không L1 | 7 | 86,39 | 28,73 | 66,60 | 96,92 | 97,17 | 91,38 |
| **Không L1** | **10** | **86,29** | **28,23** | **66,38** | **96,91** | **97,16** | **90,69** |
| L1 `1e-5` | 5 | 86,41 | 29,21 | 66,84 | 96,75 | 96,96 | 90,00 |
| L1 `1e-5` | 7 | 86,33 | 28,54 | 66,56 | 96,82 | 97,06 | 91,03 |
| **L1 `1e-5`** | **10** | **86,27** | **28,35** | **66,44** | **96,79** | **97,04** | **90,34** |
| L1 `3e-5` | 5 | 86,31 | 29,02 | 66,35 | 96,89 | 97,09 | 90,34 |
| L1 `3e-5` | 7 | 86,20 | 28,23 | 66,11 | 96,90 | 97,12 | 91,38 |
| **L1 `3e-5`** | **10** | **86,11** | **27,75** | **65,94** | **96,86** | **97,10** | **90,34** |
| L1 `1e-4` | 5 | 86,37 | 29,02 | 66,63 | 96,84 | 97,03 | 89,66 |
| L1 `1e-4` | 7 | 86,31 | 28,46 | 66,43 | 96,86 | 97,09 | 90,34 |
| **L1 `1e-4`** | **10** | **86,22** | **28,11** | **66,27** | **96,86** | **97,09** | **90,34** |

Baseline được đánh giá lại trên cùng tập test. Các chỉ số video-level của baseline không có trong kết quả evaluate này.

## 4. Thống kê L1 và weight FC

### 4.1. Loss và gradient

| Lambda | Epoch | CE trung bình | L1 chưa nhân λ | `λ × L1` trung bình | `λ × L1 / CE` | Norm gradient L1 / CE trên FC |
|---|---:|---:|---:|---:|---:|---:|
| `1e-5` | 5 | 0,002472 | 15,0911 | 0,000151 | 6,11% | — |
| `1e-5` | 10 | 0,001149 | 15,1040 | 0,000151 | 13,14% | — |
| `3e-5` | 5 | 0,002611 | 15,0869 | 0,000453 | 17,33% | 1,62% |
| `3e-5` | 10 | 0,001209 | 15,0991 | 0,000453 | 37,46% | 2,82% |
| `1e-4` | 5 | 0,002671 | 15,0770 | 0,001508 | 56,44% | 4,39% |
| `1e-4` | 10 | 0,001227 | 15,0858 | 0,001509 | 122,96% | 8,60% |

CE và L1 trong bảng là trung bình theo epoch. Gradient dùng L2 norm của gradient trên 512 weight FC, đo **trước gradient clipping trên batch đầu tiên của epoch**, không phải trung bình toàn epoch. Lượt `1e-5` chưa ghi nhận gradient này. Tỷ lệ giá trị loss và tỷ lệ gradient được tính trên các phạm vi khác nhau.

### 4.2. Phân bố weight tại epoch 10

Đặt `Δw = weight Abnormal − weight Normal` cho mỗi kernel:

| Thống kê | Không L1 | L1 `1e-5` | L1 `3e-5` | L1 `1e-4` |
|---|---:|---:|---:|---:|
| Tổng `|FC.weight|` | 15,10355 | 15,10416 | 15,09926 | 15,08585 |
| Trung bình `|Δw|` | 0,036654 | 0,036671 | 0,036650 | 0,036623 |
| Median `|Δw|` | 0,029663 | 0,029630 | 0,029637 | 0,029659 |
| Kernel có `|Δw| < 0,001` | 5/256 | 4/256 | 4/256 | 4/256 |

L1 được áp dụng lên hai hàng weight FC riêng, không áp dụng trực tiếp lên `Δw`. Các thống kê trong bảng so sánh phân bố tổng thể; kernel cùng index giữa các model train độc lập không mặc định biểu diễn cùng pattern.

## 5. Kernel selection sau training

Sau khi train, thử chọn kernel theo `|Δw|` rồi tính lại frame score bằng response trước pooling và weight FC của các kernel được chọn. Bias FC được giữ nguyên. Không cập nhật weight, không train lại model và không thay đổi video classifier. Việc chọn kernel dựa trên weight, không dùng nhãn test; tuy nhiên các kết quả chọn số lượng kernel trên test chỉ mang tính **thăm dò**.

- Với **Top-K theo `|Δw|`**, xếp kernel theo độ lớn tuyệt đối của chênh lệch weight, không phân biệt dấu. Ví dụ, kernel có `Δw = −0,09` được xếp trước kernel có `Δw = +0,08`. Vì vậy, Top-K có thể giữ cả kernel có `Δw` dương và kernel có `Δw` âm; đây không phải phép chỉ chọn kernel Abnormal.
- Với **`Δw > 0`**, giữ tất cả kernel có weight Abnormal lớn hơn weight Normal. Đây là phép thử riêng, không phải Top-K và không cố định số kernel được giữ.
- Dấu của `Δw` chỉ cho biết hướng tác động khi response dương. Đóng góp thực tế tại một vị trí là `Δw × response`; response có thể âm vì cấu hình này không dùng GELU sau CNN. Do đó, kernel có `Δw > 0` không nhất thiết làm tăng score Abnormal tại mọi vị trí.

| Checkpoint epoch 10 | Kernel dùng tính frame score | Frame AUC | Frame AP | Ano-AUC |
|---|---|---:|---:|---:|
| Không L1 | Toàn bộ 256 | 86,293 | 28,235 | 66,383 |
| Không L1 | Top-128 | 86,319 | 28,643 | 66,481 |
| Không L1 | Top-64 | 86,011 | 27,805 | 65,855 |
| Không L1 | Top-32 | 85,579 | 26,960 | 65,507 |
| Không L1 | `Δw > 0`: 134 kernel | 86,184 | 27,555 | 65,844 |
| L1 `1e-5` | Toàn bộ 256 | 86,272 | 28,349 | 66,445 |
| L1 `1e-5` | Top-128 | 86,292 | 28,752 | 66,522 |
| L1 `1e-5` | Top-64 | 85,986 | 28,008 | 65,907 |
| L1 `1e-5` | Top-32 | 85,556 | 27,089 | 65,597 |
| L1 `1e-5` | `Δw > 0`: 134 kernel | 86,185 | 27,668 | 65,930 |
| L1 `3e-5` | Toàn bộ 256 | 86,111 | 27,751 | 65,942 |
| L1 `3e-5` | Top-128 | 86,158 | 28,360 | 66,100 |
| L1 `3e-5` | Top-64 | 85,837 | 27,541 | 65,493 |
| L1 `3e-5` | Top-32 | 85,416 | 26,627 | 65,169 |
| L1 `3e-5` | `Δw > 0`: 134 kernel | 86,043 | 27,299 | 65,477 |
| L1 `1e-4` | Toàn bộ 256 | 86,217 | 28,106 | 66,268 |
| L1 `1e-4` | Top-128 | 86,241 | 28,466 | 66,358 |
| L1 `1e-4` | Top-64 | 85,956 | 27,704 | 65,815 |
| L1 `1e-4` | Top-32 | 85,539 | 26,760 | 65,499 |
| L1 `1e-4` | `Δw > 0`: 134 kernel | 86,131 | 27,495 | 65,754 |

Top-128 tăng Frame AP nhẹ ở các checkpoint đã thử, bao gồm cả model không L1; vì vậy chưa thể quy cải thiện này cho L1. Kết quả Top-64 và Top-32 thấp hơn dùng toàn bộ kernel trong các lượt hiện có. Không nên chọn số lượng kernel trên test rồi báo chính kết quả đó như đánh giá cuối cùng.
