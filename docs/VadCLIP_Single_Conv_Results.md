# Kết quả thí nghiệm VadCLIP với một lớp Temporal CNN

## 1. Mô hình và phương pháp tổng quan

### 1.1. Sơ đồ tổng quát

Thí nghiệm thay classifier của VadCLIP bằng Temporal CNN. Sơ đồ tổng quát:

```text
CLIP visual features → LGT-Adapter → Temporal CNN → logits1 → sigmoid → Score classifier
```

**Nhánh text không đổi.** Logits1 mới tiếp tục tham gia visual-text adjustment trước khi tạo logits2 của nhánh alignment.

### 1.2. Cấu trúc Temporal CNN

Thí nghiệm thay classifier của VadCLIP bằng **một lớp Temporal CNN**, sau đó thử các cách tạo frame logit và video logit.

```text
Visual features của VadCLIP (512 chiều/frame)
                       ↓
           Conv1D: 512 → 128 channels
           kernel size 5, stride 1, padding 2
                       ↓
              128 responses/frame
                       ↓
                     GELU (*)
                       ↓
     Bước 1: Tổng hợp theo channel → frame logit
                       ↓
     Bước 2: Tổng hợp theo thời gian → video logit
                       ↓
                Video-level BCE
```

CNN đọc 5 vị trí feature liên tiếp để tạo response tại mỗi frame. Classifier mới không có projection hoặc residual.

**(*) GELU theo cấu hình của từng lượt, được đánh dấu trong bảng kết quả.**

### 1.3. Bước 1: Tạo logit cho từng frame

Tại mỗi frame, tổng hợp 128 channel responses thành **một frame logit**. Các cách được thử gồm:

- **Kernel response:** qua GELU, Max pooling theo channel, rồi nhân scale và cộng bias. **Frame logit = scale × response lớn nhất + bias.**
- **Linear readout:** qua GELU, dùng Linear(128 → 1) kết hợp cả 128 responses bằng trọng số học được. Đây là phép kết hợp có trọng số, không phải Max pooling.
- **Competitive response:** chia responses thành hai nhóm 64 channels và Max pooling riêng từng nhóm. **Frame logit = max nhóm abnormal − max nhóm normal + bias.** Không dùng scale.

Sigmoid của frame logit là abnormal score dùng để tính AUC1/AP1. Các trọng số tạo logit được học trong training, không chỉ dùng khi evaluate.

### 1.4. Bước 2: Tạo logit cho video

Tổng hợp các frame logits theo **trục thời gian** để học từ nhãn video:

- **Max:** lấy frame logit cao nhất trong video làm video logit.
- **Top-K:** lấy trung bình K frame logits cao nhất làm video logit. K bằng số vị trí feature hợp lệ chia 16, làm tròn lên, tối thiểu là 1. Với 256 vị trí, chọn 16.

**Hai bước độc lập:** Max ở bước 1 chọn response giữa các channels của cùng frame; Max/Top-K ở bước 2 chọn giữa các frame trong video. Padding không tham gia pooling.

### 1.5. Hàm loss

**Loss tổng = video BCE + alignment MIL + text regularization.**

- **Video BCE:** đưa video logit ở bước 2 vào BCEWithLogitsLoss với nhãn Normal = 0, abnormal = 1. Không sigmoid trước loss. Đây là phần thay classifier loss gốc.
- **Alignment MIL:** giữ loss phân loại 14 class ở nhánh alignment.
- **Text regularization:** giữ loss hạn chế similarity giữa text embedding Normal và các abnormal class, với hệ số 0,1 như trước.

Training chỉ dùng nhãn video, không dùng nhãn frame.

### 1.6. Tham số huấn luyện chung

| Tham số | Cấu hình chung; thay đổi của từng lượt ghi tại phần kết quả |
|---|---|
| Khởi tạo | From scratch, không nạp checkpoint VadCLIP đã train |
| CLIP | Giữ pretrained weights và frozen; learnable prompt được cập nhật |
| Optimizer | AdamW, weight decay 0,01 |
| Learning rate ban đầu | `1e-5` |
| Scheduler | Giảm LR còn 0,1 lần sau epoch 4 và 8 |
| Số epoch | 10; báo kết quả checkpoint epoch 10 |
| Batch size | 64 Normal + 64 abnormal |
| Seed | 234 |
| AMP | Không dùng |
| Gradient clipping | Chưa bật |
| Deterministic | Chưa bật chế độ deterministic nghiêm ngặt |

Train dùng 16.100 crop/flip feature sequences; test dùng 290 video UCF-Crime. Các lượt CNN train độc lập, không chọn checkpoint theo test AUC. Các bảng dưới đây ghi số liệu tại epoch 10; lịch training dài hơn được ghi riêng ở phần 4.

### 1.7. Đánh giá

- **AUC1 / AP1:** ROC-AUC / Average Precision của classifier; abnormal score = sigmoid của frame logit.
- **AUC2 / AP2:** ROC-AUC / Average Precision của alignment; abnormal score = 1 − xác suất class Normal sau softmax trên 14 class.
- **Ano-AUC:** AUC trên các video abnormal; dùng nhánh có AUC toàn test cao hơn.
- **mAP:** đánh giá phát hiện và định vị sự kiện theo class từ alignment, trung bình tại các ngưỡng temporal IoU 0,1–0,5.
- **Top-15% anomaly:** chọn 15% vị trí feature có classifier score cao nhất trong mỗi abnormal video, tính tỷ lệ có anomaly, rồi lấy trung bình theo video (macro). Một vị trí được tính có anomaly nếu ít nhất một trong 16 frame video gốc tương ứng có nhãn anomaly.

AUC/AP dùng score của **toàn bộ frame**, không dùng video logit đã pooling. Score tại mỗi vị trí feature được lặp cho 16 frame video gốc theo protocol đánh giá. Top-15% chỉ là thống kê sau train, khác Top-K trong loss.

Các bảng ghi AUC/AP/mAP theo thang phần trăm. Dòng baseline là VadCLIP tự train, được đánh giá lại trong đợt thí nghiệm CNN, có AUC1 **87,72**. Các kết quả CNN dưới đây được tổng hợp từ log đánh giá đã ghi nhận.

## 2. Kernel response: so sánh Max và Top-K theo thời gian

### Phương pháp

- **Frame logit:** Conv1D → GELU → Max theo 128 channels → nhân scale dương và cộng bias. Scale và bias được học.
- **Video logit:** chạy hai lượt, một lượt dùng Max, một lượt dùng Top-K theo thời gian. Chỉ thay đổi bước này; cấu hình còn lại theo phần 1.

### Kết quả

| Phương pháp | Video aggregation | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP | Top-15% anomaly |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VadCLIP tự train | Gốc | 87,72 | 33,13 | 86,52 | 28,23 | 69,94 | 7,57 | 35,29% |
| Kernel response | Max | 82,81 | 27,64 | 86,10 | 25,86 | 65,72 | 6,77 | 35,37% |
| Kernel response | Top-K | 82,92 | 28,52 | 86,32 | 26,47 | 66,18 | 8,79 | 35,04% |

## 3. So sánh kernel response với Linear readout

### Phương pháp

- **Frame logit:** so sánh Kernel response với **Linear(128 → 1)** sau GELU. Kernel response lấy response lớn nhất; Linear học trọng số để kết hợp cả 128 responses.
- **Video logit:** cả hai dùng **Top-K theo thời gian**. Các tham số còn lại theo phần 1.

Mỗi model được train riêng. Kernel response trong bảng là lượt đối chứng readout, khác lượt Top-K ban đầu ở phần 2.

### Kết quả

| Phương pháp | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP |
|---|---:|---:|---:|---:|---:|---:|
| VadCLIP tự train | 87,72 | 33,13 | 86,52 | 28,23 | 69,94 | 7,57 |
| Kernel response, đối chứng readout | 83,09 | 28,68 | 86,25 | 26,40 | 66,04 | 8,17 |
| Linear readout | 86,57 | 31,07 | 85,72 | 26,15 | 66,64 | 9,37 |

## 4. Competitive response: hiệu số hai nhóm kernel

### Phương pháp

- **Frame logit:** chia responses thành hai nhóm 64 channels, Max theo channel trong từng nhóm, rồi tính **max nhóm abnormal − max nhóm normal + bias**. Không dùng scale.
- **Video logit:** dùng **Top-K theo thời gian**.

Hai nhóm kernel học từ nhãn video, không có nhãn frame riêng. Các lượt đều dùng seed 234 và train độc lập.

### Cấu hình các lượt chạy

- **Lượt 1 và 2:** LR `1e-5`, 10 epoch, giảm LR sau epoch 4/8; chưa dùng gradient clipping hoặc deterministic nghiêm ngặt.
- **Lượt 3:** LR `5e-6`, 20 epoch, giảm LR sau epoch 8/16; gradient clipping norm 1,0; deterministic và seed DataLoader riêng 234; evaluate sau mỗi epoch. Batch size và loss như phần 1.

### Kết quả

| Phương pháp | GELU | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP | Top-15% anomaly |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VadCLIP tự train | - | 87,72 | 33,13 | 86,52 | 28,23 | 69,94 | 7,57 | 35,29% |
| Competitive response, lượt 1 | Không | 87,01 | 33,24 | 85,31 | 25,80 | 67,87 | 6,84 | 35,78% |
| Competitive response, lượt 2 | Không | 85,86 | 33,69 | 84,98 | 25,79 | 65,79 | 7,44 | 37,20% |
| Competitive response, lượt 3 | Có | 86,10 | 26,94 | 84,91 | 23,82 | 65,50 | 7,84 | 36,86% |

Số liệu lượt 3 lấy tại epoch 10; các epoch sau gần như không thay đổi theo ghi nhận của lượt chạy. Lịch training của lượt 3 khác lượt 1–2, nên bảng không tách riêng tác động của GELU.

### Thống kê classifier score

Phân nhóm theo classifier Top-15% và ground truth, chỉ dùng sau train. Số lượng đếm theo vị trí feature; thành viên mỗi nhóm có thể khác giữa các lượt.

| Nhóm frame | Cách xác định | Số lượng lượt 1 / 2 / 3 | Mean score lượt 1 / 2 / 3 | Median score lượt 1 / 2 / 3 |
|---|---|---:|---:|---:|
| True-high trong abnormal video | Có anomaly và nằm trong Top-15% | 1.507 / 1.482 / 1.522 | 0,913 / 0,900 / 0,912 | 0,996 / 0,996 / 0,994 |
| False-high trong abnormal video | Không có anomaly nhưng nằm trong Top-15% | 2.889 / 2.914 / 2.874 | 0,888 / 0,876 / 0,879 | 0,993 / 0,992 / 0,987 |
| High-score trong normal video | Các frame thuộc Top-15% của normal video | 6.141 / 6.141 / 6.141 | 0,142 / 0,142 / 0,095 | 0,0081 / 0,0074 / 0,0116 |
| Anomaly ngoài Top-15% | Có anomaly nhưng không được chọn vào Top-15% | 3.893 / 3.918 / 3.878 | 0,807 / 0,785 / 0,783 | 0,975 / 0,967 / 0,946 |

Score nằm trong khoảng 0–1. Các nhóm được chọn theo thứ hạng Top-15%, không theo ngưỡng score 0,5.

Tỷ lệ anomaly tính trên tổng số vị trí được chọn (micro), lần lượt cho lượt 1 / 2 / 3: **34,28% / 33,71% / 34,62%**. Bảng kết quả dùng trung bình theo video (macro).

### Phân tích response của hai nhóm kernel

Phân tích bổ sung cho **lượt 3, có GELU**, trên 290 video test và các checkpoint epoch 1, 4, 8, 10. Phần này chưa có đối chứng baseline hoặc Linear. Ground truth chỉ dùng để phân nhóm sau train.

Mục đích là xác định false-high có score cao do nhóm abnormal phản hồi mạnh hay nhóm normal chưa đủ mạnh để giảm logit. Tại mỗi frame, lấy response lớn nhất của từng nhóm, sau đó thống kê trung bình:

| Nhóm frame tại epoch 10 | Max response nhóm abnormal | Max response nhóm normal | Hiệu hai response | Score trung bình |
|---|---:|---:|---:|---:|
| True-high trong abnormal video | 10,30 | 5,41 | 4,89 | 0,912 |
| False-high trong abnormal video | 9,89 | 5,52 | 4,36 | 0,879 |
| Anomaly ngoài Top-15% | 8,93 | 6,11 | 2,82 | 0,783 |
| Top-15% trong normal video | 4,87 | 9,45 | −4,57 | 0,095 |

- Trên normal video, response nhóm normal cao hơn nhóm abnormal, tạo hiệu âm và score thấp.
- Trong abnormal video, true-high và false-high có response gần nhau ở cả hai nhóm. Nhóm normal chỉ phản hồi nhỉnh hơn trên false-high: **5,52 so với 5,41**, chưa tạo khoảng cách lớn.
- Anomaly ngoài Top-15% vừa có response abnormal thấp hơn, vừa có response normal cao hơn true-high. Cả hai thành phần đều làm logit thấp hơn.

Response là giá trị trước sigmoid, không phải xác suất. Hiệu trong bảng chưa cộng bias; bias tại epoch 10 khoảng **−0,0004**. Score trung bình được tính riêng trên các frame, không phải sigmoid của hiệu trung bình.

### Mức tập trung vào các kernel

Đếm số lần mỗi kernel tạo response lớn nhất để kiểm tra model có sử dụng đa dạng kernel hay không. Tại epoch 10, hai kernel số 15 và 38 trong **nhóm abnormal** chiếm phần lớn lượt thắng:

| Nhóm frame | Kernel 15 thắng | Kernel 38 thắng | Tổng hai kernel |
|---|---:|---:|---:|
| True-high | 65,57% | 32,19% | 97,77% |
| False-high trong abnormal video | 50,45% | 44,78% | 95,23% |
| Anomaly ngoài Top-15% | 64,88% | 26,33% | 91,21% |

- Cùng hai kernel tạo phần lớn response thắng trên cả true-high lẫn false-high. Tên nhóm abnormal không đồng nghĩa các kernel đã nhận diện đúng anomaly.
- Trên toàn test set, 5 kernel thường thắng nhất chiếm **90,37%** lượt thắng của nhóm abnormal và **86,79%** của nhóm normal.
- Mức tập trung đã cao từ epoch 1: hai tỷ lệ Top-5 tương ứng là **92,83%** và **95,93%**. Không phải hiện tượng chỉ xuất hiện về cuối training.

Đây là thống kê lượt thắng trên test, không phải thống kê gradient trong training; chưa thể kết luận các kernel ít thắng hoàn toàn không được học.

### Thay đổi trên cùng tập frame qua epoch

Cố định các nhóm frame theo Top-15% của epoch 10, sau đó đo lại response tại những epoch trước. Cách này tránh so sánh các nhóm có thành viên khác nhau.

| Nhóm frame cố định | Hiệu response epoch 1 | Epoch 4 | Epoch 8 | Epoch 10 |
|---|---:|---:|---:|---:|
| True-high | 3,30 | 4,79 | 5,11 | 4,89 |
| False-high trong abnormal video | 2,60 | 3,95 | 4,81 | 4,36 |
| Anomaly ngoài Top-15% | 2,30 | 3,05 | 3,16 | 2,82 |

- Từ epoch 4 đến 10, hiệu response của true-high tăng **0,09**, false-high tăng **0,41**, còn anomaly ngoài Top-15% giảm **0,23**.
- Trên false-high, từ epoch 1 đến 10, response abnormal giảm **10,60 → 9,89**, nhưng response normal giảm mạnh hơn: **8,01 → 5,52**. Vì thế, hiệu hai nhóm vẫn tăng, làm score false-high cao hơn.
- Trong cùng khoảng epoch 4–10, AUC1 giảm **86,50 → 86,10**, AP1 giảm **30,52 → 26,94**. Các thống kê ghi nhận diễn biến cùng thời điểm, chưa chứng minh riêng thay đổi nào gây ra mức giảm metric.

### Mức thay đổi response khi qua GELU

Tại checkpoint epoch 10, đối chiếu Max response trước và sau GELU để kiểm tra activation thay đổi trực tiếp đầu vào phép lấy hiệu bao nhiêu:

| Nhóm frame | Thay đổi tuyệt đối trung bình, nhóm abnormal | Thay đổi tuyệt đối trung bình, nhóm normal |
|---|---:|---:|
| True-high | 0,000001 | 0,000247 |
| False-high trong abnormal video | 0,000121 | 0,000131 |

Các response thắng Max phần lớn dương và lớn, nên GELU gần như giữ nguyên chúng. Số liệu mô tả tác động trực tiếp lên response của checkpoint này, không thay thế phép so sánh hai lượt training có/không GELU.
