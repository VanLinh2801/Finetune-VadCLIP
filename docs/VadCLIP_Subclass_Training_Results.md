# Nhật ký thí nghiệm: Huấn luyện VadCLIP với subclass

## 1. Mục tiêu

VadCLIP gốc sử dụng một tên duy nhất để biểu diễn mỗi class, ví dụ `Explosion`, `Fighting` hoặc `Robbery`. Tuy nhiên, cùng một class có thể xuất hiện dưới nhiều hình thức khác nhau. Một vụ nổ có thể xảy ra ở xe, trong tòa nhà hoặc ngoài đường; một vụ đánh nhau có thể gồm hai người hoặc nhiều người.

Thí nghiệm subclass thay một tên class tổng quát bằng một nhóm tên cụ thể hơn. Mục tiêu là giúp model nhận biết được nhiều biểu hiện khác nhau của cùng một class mà không cần bổ sung nhãn subclass cho từng video.

## 2. Cách xây dựng subclass

Subclass được viết theo các nguyên tắc sau:

- Mỗi subclass vẫn phải thuộc rõ ràng về một class gốc.
- Tên ngắn, tập trung vào hành động, đối tượng hoặc bối cảnh có thể quan sát được.
- Các subclass trong cùng class nên mô tả những trường hợp khác nhau, tránh chỉ thay đổi vài từ nhưng giữ nguyên ý nghĩa.
- Không sử dụng description của từng video test và không gán nhãn subclass cho video.
- Mỗi subclass được đưa riêng qua learnable prompt và CLIP Text Encoder để tạo một text embedding riêng.

Ba bộ subclass đã được sử dụng trong các thí nghiệm.

### 2.1. Bộ 1: Năm mô tả ngắn cho mỗi class

Bộ đầu tiên sử dụng 5 subclass cho mỗi class. Tên subclass gồm tên class gốc và một hành động hoặc tình huống cụ thể.

| Class | Ví dụ subclass |
|---|---|
| Arrest | `Arrest with handcuffs`, `Arrest with police escort` |
| Arson | `Arson with a lighter`, `Arson with poured fuel` |
| Explosion | `Explosion of a vehicle`, `Explosion with flying debris` |
| Fighting | `Fighting between groups`, `Fighting with mutual grappling` |
| Shoplifting | `Shoplifting into a bag`, `Shoplifting inside clothing` |

Toàn bộ 14 class tạo thành 70 subclass.

### 2.2. Bộ 2: Mười mô tả trực quan cho mỗi class

Bộ thứ hai tăng lên 10 subclass cho mỗi class. Tên subclass vẫn giữ class gốc nhưng mô tả chi tiết hơn dấu hiệu có thể quan sát trong video.

| Class | Ví dụ subclass |
|---|---|
| Abuse | `Abuse with a person being dragged`, `Abuse with a person being pinned down` |
| Arrest | `Arrest beside a police patrol car`, `Arrest during a police pat-down search` |
| Explosion | `Explosion with a bright blast flash`, `Explosion with an outward dust cloud` |
| RoadAccidents | `Road accident involving a struck pedestrian`, `Road accident with an overturned vehicle` |
| Robbery | `Robbery with a gun pointed at a cashier`, `Robbery with an armed person demanding money` |

Toàn bộ 14 class tạo thành 140 subclass.

### 2.3. Bộ 3: Mười nhãn ngữ nghĩa cho mỗi class

Bộ thứ ba cũng có 10 subclass cho mỗi class, nhưng sử dụng tên dạng nhãn ngữ nghĩa ngắn thay vì câu mô tả. Mỗi nhãn biểu diễn một dạng sự kiện hoặc bối cảnh đặc trưng.

| Class | Ví dụ subclass |
|---|---|
| Normal | `Pedestrian Activity`, `Store Browsing`, `Office Activity` |
| Arson | `Vehicle Arson`, `Ignition Arson`, `Fire-Spreading Arson` |
| Explosion | `Vehicle Explosion`, `Fireball Explosion`, `Multiple-Blast Explosion` |
| Fighting | `One-on-One Fight`, `Grappling Fight`, `Group Fight` |
| RoadAccidents | `Vehicle Collision`, `Car-Motorcycle Collision`, `Vehicle Rollover` |
| Shoplifting | `Pocket Shoplifting`, `Bag Shoplifting`, `Checkout-Avoidance Shoplifting` |

Toàn bộ 14 class tạo thành 140 subclass.

## 3. Cách đưa subclass vào VadCLIP

Kiến trúc visual của VadCLIP và nhánh classifier phát hiện abnormal được giữ nguyên. Thay đổi nằm ở nhánh alignment giữa visual feature và text.

Quy trình thực hiện:

1. Mỗi subclass được đặt giữa các learnable prompt token dùng chung của VadCLIP.
2. CLIP Text Encoder đã đóng băng biến từng subclass thành một embedding riêng.
3. Model tính mức tương đồng giữa từng snippet và từng subclass.
4. Với mỗi subclass, Temporal Top-K chọn các snippet có alignment cao nhất và lấy trung bình để tạo điểm ở cấp video.
5. Các điểm subclass thuộc cùng một class được gộp lại thành điểm của class gốc.
6. Model tiếp tục học bằng nhãn class cấp video như VadCLIP; không cần nhãn subclass.

Hai cách gộp được thử nghiệm:

- **Max:** lấy điểm của subclass mạnh nhất làm điểm của class. Cách này chọn một biểu hiện phù hợp nhất với video.
- **Soft:** cho tất cả subclass cùng đóng góp, nhưng subclass có điểm cao được ưu tiên hơn. Temperature được đặt bằng `1.0`.

Hàm loss tổng vẫn gồm ba thành phần:

```text
L_total = L_classifier + L_MIL_class + L_text_regularization
```

- `L_classifier`: phân biệt video normal và abnormal từ nhánh classifier.
- `L_MIL_class`: dự đoán class gốc sau khi gộp các điểm subclass.
- `L_text_regularization`: hạn chế biểu diễn text của Normal quá gần các class abnormal.

Không sử dụng loss subclass riêng, description video, normal suppression hoặc abnormal suppression trong các lượt chạy dưới đây.

## 4. Cấu hình huấn luyện

Hai cách khởi tạo model được sử dụng:

| Cách khởi tạo | Số epoch | Learning rate | Lịch giảm learning rate |
|---|---:|---:|---|
| Train từ đầu | 10 | `2e-5` | Giảm 10 lần sau epoch 4 và 8 |
| Train từ checkpoint | 3 | `5e-6` hoặc `2e-5` tùy lượt | Giảm 10 lần sau epoch 2 |

Các cấu hình chung:

- Checkpoint dùng để fine-tune là **baseline tự train**, không phải checkpoint do tác giả VadCLIP công bố.
- Batch gồm 64 normal chunk và 64 abnormal chunk.
- Visual feature có 512 chiều; mỗi mẫu được chuẩn hóa về tối đa 256 snippet.
- CLIP pretrained được đóng băng; learnable prompt và các module VadCLIP còn lại được cập nhật.
- Optimizer AdamW, seed `234`, huấn luyện FP32 và không dùng AMP.
- Model được đánh giá trên đủ 290 video test.
- Kết quả train từ đầu lấy epoch 10; kết quả từ checkpoint lấy epoch 3. Test set không được dùng để chọn epoch tốt nhất.

## 5. Mốc baseline tự train

Mốc so sánh được lấy từ báo cáo baseline huấn luyện lại bằng class name gốc:

| Model | AUC1 | AP1 | AUC2 | AP2 | AVG mAP |
|---|---:|---:|---:|---:|---:|
| Baseline tự train | **87.44** | 32.07 | 85.64 | 26.15 | 6.92 |

Trong đó:

- AUC1 và AP1 thuộc nhánh classifier.
- AUC2 và AP2 thuộc nhánh alignment text-visual.
- AVG mAP là trung bình mAP tại IoU từ 0.1 đến 0.5.
- Báo cáo baseline này không ghi Ano-AUC, vì vậy Ano-AUC của các thí nghiệm chỉ được báo lại, không tính chênh lệch với baseline.

## 6. Kết quả với Bộ 1

### 6.1. Cấu hình các lượt chạy

| Lượt | Khởi tạo | Gộp subclass | Epoch | Learning rate |
|---|---|---|---:|---:|
| B1-Scratch-Max | Train từ đầu | Max | 10 | `2e-5` |
| B1-Scratch-Soft | Train từ đầu | Soft | 10 | `2e-5` |
| B1-Checkpoint-Max-LR thấp | Baseline tự train | Max | 3 | `5e-6` |
| B1-Checkpoint-Soft-LR thấp | Baseline tự train | Soft | 3 | `5e-6` |
| B1-Checkpoint-Max | Baseline tự train | Max | 3 | `2e-5` |
| B1-Checkpoint-Soft | Baseline tự train | Soft | 3 | `2e-5` |

### 6.2. Kết quả

| Lượt | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | AVG mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline tự train | **87.44** | 32.07 | 85.64 | 26.15 | Không ghi nhận | 6.92 |
| B1-Scratch-Max | 87.18 | **34.66** | 84.75 | 25.51 | 68.35 | 7.11 |
| B1-Scratch-Soft | 86.97 | 34.36 | 84.12 | 25.28 | 68.00 | 5.60 |
| B1-Checkpoint-Max-LR thấp | 87.26 | 32.13 | 85.73 | 26.89 | **69.00** | 6.43 |
| B1-Checkpoint-Soft-LR thấp | **87.27** | 32.11 | **85.74** | **27.26** | 68.87 | 5.62 |
| B1-Checkpoint-Max | 87.14 | 33.92 | 85.68 | 25.75 | 68.46 | 6.94 |
| B1-Checkpoint-Soft | 86.88 | 33.34 | 85.42 | 25.81 | 68.27 | **7.34** |

Nhận xét:

- Learning rate thấp giúp hai lượt checkpoint giữ AUC1 gần baseline nhất, nhưng AVG mAP đều thấp hơn baseline.
- Khi dùng LR `2e-5`, `soft` nâng AVG mAP lên `7.34`, nhưng AUC1 giảm xuống `86.88`.
- `Scratch-Max` cho AP1 cao nhất (`34.66`) và AVG mAP `7.11`, nhưng AUC1 vẫn thấp hơn baseline `0.26` điểm.

## 7. Kết quả với Bộ 2

### 7.1. Cấu hình

Cả bốn lượt đều dùng LR `2e-5`. Lượt train từ đầu chạy 10 epoch; lượt từ checkpoint chạy 3 epoch.

### 7.2. Kết quả

| Lượt | Khởi tạo | Gộp | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | AVG mAP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline tự train | Class name gốc | Không dùng subclass | **87.44** | 32.07 | **85.64** | 26.15 | Không ghi nhận | 6.92 |
| B2-Scratch-Max | Train từ đầu | Max | 86.82 | 29.85 | 85.48 | 27.56 | 67.75 | 5.49 |
| B2-Scratch-Soft | Train từ đầu | Soft | 86.27 | 32.27 | 84.89 | 25.92 | 66.72 | 8.14 |
| B2-Checkpoint-Max | Baseline tự train | Max | 86.97 | 32.49 | 85.59 | 26.90 | **68.47** | 8.24 |
| B2-Checkpoint-Soft | Baseline tự train | Soft | 86.80 | **33.61** | 84.95 | 25.04 | 67.88 | **8.37** |

Nhận xét:

- Hai lượt từ checkpoint đạt AVG mAP cao hơn rõ rệt so với baseline: `8.24` với Max và `8.37` với Soft.
- Đổi lại, AUC1 của cả bốn cấu hình đều thấp hơn baseline. Lượt giữ AUC1 tốt nhất là Checkpoint-Max với `86.97`, thấp hơn baseline `0.47` điểm.
- Soft aggregation không ổn định khi train từ đầu: AVG mAP cao (`8.14`) nhưng AUC1 và Ano-AUC giảm mạnh hơn.
- Phần tăng mAP không đồng đều ở mọi ngưỡng IoU. Ví dụ, Checkpoint-Soft đạt AVG mAP cao nhất nhưng mAP@0.5 chỉ `2.94`, thấp hơn baseline `3.10`.

## 8. Kết quả với Bộ 3

### 8.1. Cấu hình

Cả bốn lượt dùng LR `2e-5`. Lượt train từ đầu chạy 10 epoch; lượt từ checkpoint chạy 3 epoch. Các subclass được viết dưới dạng nhãn ngữ nghĩa ngắn.

### 8.2. Kết quả

| Lượt | Khởi tạo | Gộp | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | AVG mAP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline tự train | Class name gốc | Không dùng subclass | **87.44** | 32.07 | 85.64 | 26.15 | Không ghi nhận | 6.92 |
| B3-Scratch-Max | Train từ đầu | Max | **87.22** | 32.89 | **86.10** | 27.74 | **68.63** | **7.75** |
| B3-Scratch-Soft | Train từ đầu | Soft | 87.14 | **33.65** | 85.73 | 27.55 | 68.59 | 5.87 |
| B3-Checkpoint-Max | Baseline tự train | Max | 86.49 | 33.03 | 83.79 | 24.37 | 67.60 | 5.81 |
| B3-Checkpoint-Soft | Baseline tự train | Soft | 86.98 | 33.23 | 85.70 | **30.34** | 68.51 | 7.32 |

Nhận xét:

- `B3-Scratch-Max` là cấu hình cân bằng nhất của Bộ 3: AUC1 chỉ thấp hơn baseline `0.22` điểm, AUC2 cao hơn baseline `0.46` điểm và AVG mAP tăng từ `6.92` lên `7.75`.
- `B3-Checkpoint-Soft` đạt AP2 cao nhất (`30.34`) nhưng AVG mAP chỉ gần baseline. AP2 đo phát hiện abnormal từ alignment, trong khi mAP còn yêu cầu phân loại đúng class và định vị đúng sự kiện.
- `B3-Checkpoint-Max` giảm mạnh cả AUC2 và AVG mAP. Nhãn ngữ nghĩa ngắn kết hợp Max không phù hợp khi bắt đầu từ checkpoint class-name đã hội tụ.
- `B3-Scratch-Soft` giữ AUC tương đối tốt nhưng AVG mAP giảm xuống `5.87`.

## 9. So sánh tổng hợp

| Tiêu chí | Cấu hình tốt nhất | Kết quả |
|---|---|---:|
| AUC1 cao nhất trong các model subclass | B1-Checkpoint-Soft-LR thấp | 87.27 |
| AP1 cao nhất | B1-Scratch-Max | 34.66 |
| AUC2 cao nhất | B3-Scratch-Max | 86.10 |
| AP2 cao nhất | B3-Checkpoint-Soft | 30.34 |
| Ano-AUC cao nhất | B1-Checkpoint-Max-LR thấp | 69.00 |
| AVG mAP cao nhất | B2-Checkpoint-Soft | 8.37 |
| Cân bằng AUC và mAP tốt nhất | B3-Scratch-Max | AUC1 87.22; AVG mAP 7.75 |

Các kết quả cho thấy chưa có một cấu hình subclass tốt nhất ở mọi metric:

- Bộ 1 giữ AUC tốt nhất, đặc biệt khi fine-tune checkpoint với learning rate thấp, nhưng cải thiện mAP còn hạn chế.
- Bộ 2 tạo mAP cao nhất, nhưng AUC1 và Ano-AUC giảm.
- Bộ 3 kết hợp với train từ đầu và Max cho sự cân bằng tốt nhất giữa AUC và mAP.
- Max thường khiến một subclass chiếm ưu thế trong mỗi class. Soft cho nhiều subclass đóng góp hơn nhưng có thể làm loãng tín hiệu và không phải lúc nào cũng cải thiện kết quả.

## 10. Kết luận

Subclass giúp mở rộng cách biểu diễn class và có thể cải thiện khả năng phân loại, định vị sự kiện ở một số cấu hình. Tuy nhiên, tăng số lượng subclass không tự động làm model tốt hơn. Chất lượng cách đặt tên, cách gộp và cách khởi tạo model đều ảnh hưởng mạnh đến kết quả.

Kết quả đáng chú ý nhất là:

- Bộ 2 với Checkpoint-Soft đạt AVG mAP cao nhất (`8.37`).
- Bộ 3 với Scratch-Max giữ AUC1 gần baseline nhất trong các cấu hình đồng thời cải thiện mAP (`87.22` và `7.75`).
- Chưa có cấu hình nào vượt baseline tự train `87.44` về AUC1.

Mỗi cấu hình mới chỉ được chạy với một seed. Vì vậy, các chênh lệch nhỏ cần được xem là tín hiệu thử nghiệm, chưa đủ để kết luận chắc chắn về độ ổn định thống kê.
