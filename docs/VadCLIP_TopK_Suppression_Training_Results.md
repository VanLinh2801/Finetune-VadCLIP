# Nhật Ký Thí Nghiệm: Top-k Suppression Cho VadCLIP

## Hướng Dẫn Đọc Nhanh

### Bài toán đang được giải quyết

VadCLIP dùng để phát hiện anomaly trong video. Mỗi video được chia thành nhiều `snippet`. Mô hình gán cho mỗi snippet một điểm từ thấp đến cao; điểm càng cao thì mô hình càng cho rằng snippet đó là abnormal.

Trong quy trình huấn luyện weakly supervised trên UCF-Crime, tập huấn luyện chỉ có nhãn **toàn bộ video**, tức video normal hoặc abnormal. Mô hình không được cho biết chính xác snippet nào chứa anomaly. Vì vậy, VadCLIP tự chọn một số snippet có điểm cao nhất làm bằng chứng. Nhóm được chọn này được gọi là **top-k**. Temporal ground truth của tập kiểm tra chỉ được dùng để đánh giá và phân tích sau khi huấn luyện.

Nếu top-k được chọn đúng, mô hình học được dấu hiệu bất thường. Nếu top-k chứa đoạn nền bình thường, mô hình có thể học nhầm đoạn nền đó như một dấu hiệu bất thường.

### Các thuật ngữ quan trọng

| Thuật ngữ | Cách hiểu đơn giản |
|---|---|
| Snippet | Một đoạn ngắn của video, tương ứng với 16 frame trong phần đánh giá |
| Anomaly score | Điểm bất thường mà mô hình gán cho một snippet |
| Top-k | Nhóm `k` snippet có điểm cao nhất |
| `true_high` | Abnormal snippet được chọn vào top-k |
| `false_high` | Snippet bình thường nhưng bị chọn nhầm vào top-k |
| `missed_anomaly` | Abnormal snippet không được chọn vào top-k |
| Classifier branch | Nhánh dự đoán trực tiếp một điểm bất thường cho mỗi snippet |
| Alignment branch | Nhánh so sánh hình ảnh với biểu diễn văn bản của các lớp |
| AUC | Khả năng xếp anomaly frame cao hơn normal frame trên toàn bộ tập kiểm tra |
| AP | Chất lượng phát hiện khi dữ liệu anomaly và normal mất cân bằng |
| mAP | Chất lượng xác định đúng loại và khoảng thời gian của sự kiện bất thường |

### Hai phương pháp được thử nghiệm

**Normal top-k suppression** sử dụng thông tin chắc chắn rằng mọi snippet trong video Normal đều bình thường. Các snippet có điểm cao nhất trong video Normal được kéo xuống.

**Abnormal high-low separation** không biết snippet nào thật sự bất thường. Phương pháp chỉ buộc nhóm điểm cao trong video bất thường phải cách xa nhóm điểm thấp. Vì nhóm điểm cao do chính mô hình chọn, phương pháp này có thể khuếch đại cả lựa chọn đúng lẫn lựa chọn sai.

### Kết quả quan trọng nhất

- Chưa có cấu hình nào vượt classifier AUC `88.02` của checkpoint do tác giả VadCLIP công bố.
- So với baseline tự huấn luyện `87.44`, cấu hình Normal tốt nhất gần như giữ nguyên classifier AUC (`87.43`) và tăng alignment AUC lên `86.53`.
- Cấu hình abnormal tốt nhất tăng avg mAP từ `6.92` lên `7.94`, nhưng classifier AUC giảm nhẹ còn `87.24`.
- Trong phân tích top-15%, phần lớn snippet được chọn là background. Vì vậy, tăng khoảng cách top-k chưa chắc cải thiện AUC.
- Visual feature vẫn chứa thông tin phân biệt tốt. Điểm nghẽn chính có nhiều khả năng nằm ở cách MIL chọn evidence và huấn luyện classifier từ nhãn video.

Phần còn lại của tài liệu trình bày chi tiết cách xây dựng loss, kết quả từng cấu hình và các phép phân tích dẫn tới những kết luận trên.

## 1. Mục Tiêu

VadCLIP được huấn luyện theo cơ chế weakly supervised learning: nhãn chỉ cho biết toàn bộ video là normal hay abnormal, nhưng không chỉ rõ snippet nào chứa anomaly. Trong quá trình huấn luyện, Multiple Instance Learning (MIL) tập trung vào một số snippet có điểm cao nhất.

Điều này dẫn đến hai vấn đề cần kiểm tra:

- Trong video bình thường, một số snippet nền có thể bị chấm điểm rất cao và trở thành dự đoán sai.
- Trong abnormal video, mô hình có thể chỉ tập trung vào một vài dấu hiệu nổi bật và bỏ qua các abnormal snippet khác.

Các thí nghiệm thêm ràng buộc top-k được thực hiện để trả lời hai câu hỏi:

> Việc hạ điểm các snippet cao nhất trong video Normal có làm giảm false positive và tăng AUC hay không?

> Việc tăng khoảng cách giữa nhóm điểm cao và nhóm điểm thấp trong video bất thường có giúp mô hình định vị sự kiện tốt hơn hay không?

Mọi thí nghiệm chỉ dùng baseline đều giữ nguyên kiến trúc, cách biểu diễn tên lớp và ba hàm mất mát gốc của VadCLIP. Thay đổi duy nhất là thêm một hàm mất mát phụ dựa trên top-k.

## 2. Cách Làm

### 2.1. Hai Nhánh Điểm Của VadCLIP

VadCLIP tạo ra hai loại anomaly score cho từng snippet:

```text
s_cls(t) = sigmoid(logits1(t))
```

Trong đó `s_cls(t)` là điểm bất thường của snippet thứ `t` do classifier branch dự đoán trực tiếp.

```text
s_align(t) = 1 - softmax(logits2(t))[Normal]
```

Trong đó `s_align(t)` là điểm bất thường của alignment branch. Nếu xác suất class Normal thấp thì điểm bất thường cao.

Ràng buộc bổ sung có thể áp dụng cho:

- `classifier`: chỉ nhánh `logits1`;
- `alignment`: chỉ nhánh `logits2`;
- `both`: áp dụng cho cả hai nhánh rồi lấy trung bình loss.

### 2.2. Phân Biệt Top-k Của MIL Và Top-k Của Regularization

VadCLIP gốc đã dùng top-k trong MIL:

```text
k_MIL = floor(T / 16) + 1
```

Với video được biểu diễn bằng `T = 256` snippet, MIL chọn khoảng `17` snippet, tương đương `6.6%` timeline.

Top-k dùng cho hàm mất mát bổ sung là một phép chọn riêng:

```text
k_reg = max(1, floor(T * r))
```

Trong đó `r` là tỷ lệ top-k, thường được đặt từ `0.10` đến `0.20`. Ví dụ, `r = 0.15` tương ứng khoảng `38` trong `256` snippet. Đây là nhóm dùng riêng cho hàm mất mát bổ sung; nó không thay thế `17` snippet mà MIL gốc lựa chọn.

### 2.3. Normal Top-k Suppression

Với mỗi video Normal, các snippet có score cao nhất chính là những dự đoán đáng ngờ nhất, vì toàn bộ video đã được biết là bình thường ở mức video-level.

Gọi `H` là tập `k_reg` snippet có score cao nhất. Loss suppression cho một nhánh score được tính như sau:

```text
L_normal_topk = mean(s(t)), với t thuộc H
```

Minimize loss này sẽ trực tiếp kéo các score cao nhất trong video Normal về gần `0`.

Nếu áp dụng cho cả hai nhánh:

```text
L_normal_topk =
    0.5 * L_normal_topk_classifier
  + 0.5 * L_normal_topk_alignment
```

Hàm loss tổng trở thành:

```text
L_total =
    L_cls
  + L_mil
  + L_text_reg
  + lambda_normal_topk(epoch) * L_normal_topk
```

Trong đó:

- `L_cls`: binary classification loss của nhánh `logits1`;
- `L_mil`: MIL classification loss của nhánh video-text alignment `logits2`;
- `L_text_reg`: giữ các text embedding của class tách biệt;
- `L_normal_topk`: hạ các false-high score trong video Normal;
- `lambda_normal_topk`: trọng số của regularization.

### 2.4. Abnormal High-Low Separation

Tên gọi ngắn trong quá trình thử nghiệm là `abnormal top-k suppression`, nhưng phương pháp này không trực tiếp hạ toàn bộ top-k của video abnormal. Bản chất của nó là tạo khoảng cách giữa nhóm score cao và nhóm score thấp trong cùng video bất thường.

Với mỗi video abnormal:

- `H`: nhóm `k_reg` snippet có score cao nhất;
- `L`: nhóm `k_reg` snippet có score thấp nhất;
- `m`: khoảng cách tối thiểu mong muốn giữa hai nhóm.

Loss được tính như sau:

```text
L_abnormal_bg = max(0, max(s(L)) - min(s(H)) + m)
```

Loss bằng `0` khi snippet yếu nhất trong nhóm high vẫn cao hơn snippet mạnh nhất trong nhóm low ít nhất một khoảng `m`. Nếu điều kiện chưa đạt, gradient sẽ kéo nhóm high lên và nhóm low xuống.

Nếu áp dụng cho cả hai nhánh, loss classifier và alignment được lấy trung bình. Hàm loss tổng là:

```text
L_total =
    L_cls
  + L_mil
  + L_text_reg
  + lambda_abnormal_bg(epoch) * L_abnormal_bg
```

Phương pháp này chỉ biết nhãn của toàn video. Nó không biết snippet nào thật sự bất thường, nên có thể củng cố nhầm một đoạn nền đã được mô hình xếp vào nhóm điểm cao.

### 2.5. Warmup

Warmup là giai đoạn tạm chưa bật hàm mất mát bổ sung trong một số epoch đầu:

```text
lambda(epoch) = 0, nếu epoch <= warmup_epochs
lambda(epoch) = lambda_config, sau giai đoạn warmup
```

Mục đích là cho mô hình học tín hiệu cơ bản trước khi sử dụng chính dự đoán của nó để chọn top-k. Hai cấu hình chính đã được thử:

- `warmup = 0`: bật regularization ngay từ epoch đầu;
- `warmup = 1`: bắt đầu regularization sau epoch đầu tiên.

## 3. Cấu Hình Training Và Mốc So Sánh

### 3.1. Cấu Hình Chung

| Thành phần | Cấu hình |
|---|---|
| Dataset | UCF-Crime |
| Kiến trúc | VadCLIP gốc |
| Khởi tạo | Train from scratch |
| Class prompt | Class name và learnable prompt gốc của VadCLIP |
| Learning rate | 1e-5 cho baseline-only |
| Batch size | 64 |
| Số epoch tối đa | 10 |
| Seed | 234 |
| Evaluation | Sau mỗi epoch |
| Số video test | 290 |

### 3.2. Hai Mốc Baseline

| Mốc so sánh | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline gốc VadCLIP | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 6.68 |
| Baseline tự train cùng protocol của nhóm baseline-only | 87.44 | 32.07 | 85.64 | 26.15 | Không ghi nhận | 6.92 |

Mốc `87.44` được dùng để tính delta chính cho các thí nghiệm baseline-only vì được train lại bằng cùng train list, seed, learning rate và quy trình đánh giá. Một baseline tự train trước đó đạt `87.19`, nhưng được giữ như số tham khảo lịch sử thay vì dùng cho so sánh trực tiếp.

## 4. Kết Quả Baseline Kết Hợp Top-k Regularization

### 4.1. Cấu Hình Các Lượt Chạy

| Ký hiệu | Phương pháp | Nhánh | Lambda | Tỷ lệ top-k | Margin | Warmup |
|---|---|---|---:|---:|---:|---:|
| N1 | Normal suppression | Both | 0.05 | 0.15 | Không dùng | 1 |
| N2 | Normal suppression | Both | 0.05 | 0.15 | Không dùng | 0 |
| N3 | Normal suppression | Classifier | 0.10 | 0.15 | Không dùng | 0 |
| N4 | Normal suppression | Classifier | 0.05 | 0.10 | Không dùng | 0 |
| A1 | Abnormal high-low separation | Both | 0.05 | 0.15 | 0.15 | 0 |
| A2 | Abnormal high-low separation | Both | 0.05 | 0.20 | 0.25 | 0 |

### 4.2. Kết Quả Tổng Quan

| Mô hình | classifier AUC | classifier AP | alignment AUC | alignment AP | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | avg mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tự train cùng protocol | 87.44 | 32.07 | 85.64 | 26.15 | 12.02 | 9.98 | 5.47 | 4.04 | 3.10 | 6.92 |
| N1: Normal, both, lambda 0.05, warmup 1 | 87.05 | 33.03 | 85.32 | 24.94 | 9.54 | 7.68 | 4.62 | 3.26 | 2.40 | 5.50 |
| N2: Normal, both, lambda 0.05, warmup 0 | **87.43** | 32.37 | **86.53** | **27.81** | 12.41 | 9.09 | 5.63 | 4.71 | 3.96 | 7.16 |
| N3: Normal, classifier, lambda 0.10, ratio 0.15 | 87.34 | **32.56** | 85.88 | 27.08 | 12.32 | 10.61 | 6.62 | 4.91 | 4.27 | 7.75 |
| N4: Normal, classifier, lambda 0.05, ratio 0.10 | 87.04 | 32.29 | 85.68 | 26.28 | 12.35 | 9.71 | 5.93 | 5.31 | 4.06 | 7.47 |
| A1: Abnormal, margin 0.15, ratio 0.15 | 87.23 | 30.86 | 86.34 | 26.87 | 12.26 | 9.66 | 6.08 | 4.27 | 3.83 | 7.22 |
| A2: Abnormal, margin 0.25, ratio 0.20 | 87.24 | 31.98 | 86.15 | 27.67 | **13.75** | **11.12** | **7.35** | 4.50 | 2.95 | **7.94** |

### 4.3. Delta So Với Baseline Tự Train Cùng Protocol

| Cấu hình | Delta classifier AUC | Delta classifier AP | Delta alignment AUC | Delta alignment AP | Delta avg mAP |
|---|---:|---:|---:|---:|---:|
| N1 | -0.39 | +0.96 | -0.32 | -1.21 | -1.42 |
| N2 | **-0.01** | +0.31 | **+0.89** | **+1.66** | +0.24 |
| N3 | -0.10 | +0.49 | +0.24 | +0.93 | +0.83 |
| N4 | -0.40 | +0.22 | +0.04 | +0.13 | +0.55 |
| A1 | -0.21 | -1.21 | +0.70 | +0.72 | +0.30 |
| A2 | -0.20 | -0.09 | +0.52 | +1.52 | **+1.02** |

### 4.4. Diễn Biến Đáng Chú Ý Theo Epoch

Với cấu hình N2, classifier AUC tăng từ `85.60` ở epoch 1 lên `87.43` tại epoch 7. Alignment AUC tăng từ `83.39` lên `86.53`. Mô hình tốt nhất theo classifier AUC xuất hiện ở epoch 7, nhưng avg mAP không đạt giá trị cao nhất tại cùng epoch.

Với cấu hình A2, classifier AUC tốt nhất đạt `87.24` ở epoch 6. Avg mAP cũng đạt mức tốt nhất `7.94` tại epoch này. Sau epoch 6, AUC và mAP không tiếp tục tăng ổn định.

Điều này cho thấy checkpoint tối ưu theo frame-level AUC không phải lúc nào cũng là checkpoint tối ưu theo temporal detection mAP.

## 5. Top-k Regularization Trong Pipeline Prompt

Các thí nghiệm dưới đây thay tên lớp đơn lẻ bằng nhiều cụm từ mô tả ngắn. Do cách biểu diễn văn bản đã thay đổi, kết quả được trình bày riêng với nhóm chỉ sử dụng VadCLIP gốc.

### 5.1. Năm Keyword Prompt Cho Mỗi Lớp Kết Hợp Abnormal Separation

#### Phương pháp

Tên lớp trong VadCLIP khá ngắn và chưa thể hiện đầy đủ các dấu hiệu thị giác. Trong khi đó, một câu mô tả dài có thể chứa bối cảnh và diễn biến không thể nhận biết từ một snippet. Vì vậy, mỗi lớp được bổ sung **năm cụm từ khóa ngắn**, tập trung vào đối tượng và hành động có thể quan sát trực tiếp.

Ví dụ:

| Lớp | Keyword prompt |
|---|---|
| Explosion | `blast shockwave debris`, `detonation fireball smoke` |
| Arson | `lighter accelerant ignition`, `person setting fire` |
| Fighting | `mutual punches brawl`, `opponents exchanging blows` |

Các prompt được thiết kế để làm rõ khác biệt giữa những lớp dễ nhầm. Chẳng hạn, `Arson` nhấn vào hành động cố ý châm lửa, còn `Explosion` nhấn vào vụ nổ, sóng xung kích và mảnh vỡ.

Mỗi prompt được CLIP mã hóa và chấm điểm dựa trên bốn yếu tố: gần đúng tên lớp, xa các lớp dễ nhầm, nhất quán với các prompt cùng lớp và không quá giống prompt của lớp khác. Điểm số được chuyển thành trọng số bằng softmax; prompt phù hợp hơn đóng góp nhiều hơn nhưng cả năm prompt vẫn được giữ lại.

Khi đưa vào VadCLIP, tên lớp được ghép với keyword và đặt giữa các token ngữ cảnh học được:

```text
[learnable prefix] + [class name + keyword] + [learnable postfix]
```

CLIP text encoder được giữ cố định. Mỗi snippet được so sánh riêng với năm prompt của từng lớp, sau đó năm kết quả được tổng hợp theo trọng số để tạo `logits2` cho 14 lớp. Các prompt không bị lấy trung bình thành một embedding duy nhất trước khi so sánh với hình ảnh.

Một consistency loss nhỏ được dùng để hạn chế các prompt cùng lớp tập trung vào những thời điểm hoàn toàn khác nhau. Hàm mất mát của mô hình prompt là:

```text
L_prompt = L_cls + L_mil + L_text_reg + 0.01 * L_consistency
```

Trong đó, `L_cls`, `L_mil` và `L_text_reg` là các loss của VadCLIP; `L_consistency` duy trì sự nhất quán giữa các prompt thuộc cùng lớp.

Khi kết hợp abnormal separation, mô hình còn được khuyến khích tạo khoảng cách giữa nhóm snippet có điểm cao và nhóm snippet có điểm thấp trong video bất thường:

```text
L_total = L_prompt + lambda_abnormal_bg(epoch) * L_abnormal_bg
```

#### Cấu hình và kết quả

Cấu hình abnormal separation sử dụng `lambda = 0.05`, margin `0.15`, warmup một epoch, tỷ lệ chọn `15%` và áp dụng trên cả hai nhánh điểm. Mô hình được huấn luyện từ đầu với learning rate `2e-5`.

| Cấu hình | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| Năm keyword prompt, chưa regularize | 87.15 | 30.03 | 85.34 | 24.98 | 68.19 | 9.14 |
| Năm keyword prompt + abnormal separation | **87.52** | **32.70** | **85.66** | **26.69** | Chưa lưu | Chưa lưu |
| Chênh lệch | **+0.37** | **+2.67** | **+0.32** | **+1.71** | - | - |

Abnormal separation cải thiện cả AUC và AP của hai nhánh trong thí nghiệm này. Tuy nhiên, Ano-AUC và mAP chưa được lưu đầy đủ nên không được suy diễn thêm.

### 5.2. Multi-prompt Kết Hợp Normal Suppression

Một bộ gồm năm prompt cho mỗi lớp cũng được thử nghiệm với normal top-k suppression. Phương pháp chọn các snippet có điểm bất thường cao nhất trong video bình thường và giảm các điểm này; kết quả cuối epoch 3 như sau:

| Cấu hình | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| Multi-prompt, không regularization | 86.41 | 28.87 | 85.60 | 26.43 | 67.06 | 9.67 |
| Multi-prompt + normal suppression | 86.33 | 28.41 | 85.29 | 25.76 | 66.64 | 9.27 |
| Chênh lệch | -0.08 | -0.46 | -0.32 | -0.67 | -0.42 | -0.40 |

Trong thí nghiệm này, normal suppression không cải thiện chỉ số nào. Điều đó cho thấy cùng một hàm mất mát phụ có thể cho kết quả khác nhau khi cách mô tả lớp làm thay đổi phân phối điểm của mô hình.

## 6. Nhận Xét Kết Quả

### 6.1. Normal Suppression Có Thể Giữ AUC, Nhưng Phụ Thuộc Cấu Hình

Cấu hình N2 gần như giữ nguyên classifier AUC so với baseline tự train: `87.43` so với `87.44`. Đồng thời, alignment AUC tăng `0.89` điểm và avg mAP tăng `0.24` điểm.

Tuy nhiên, N1 dùng cùng `lambda = 0.05`, cùng tỷ lệ `0.15` và cùng hai nhánh nhưng warmup một epoch lại giảm avg mAP từ `6.92` xuống `5.50`. Trong các lượt chạy hiện có, bật suppression ngay từ đầu tốt hơn warmup một epoch. Đây vẫn là quan sát từ một seed, chưa đủ để kết luận warmup luôn có hại.

### 6.2. Chỉ Regularize Classifier Cho mAP Tốt Nhưng Không Tăng AUC

N3 đạt avg mAP `7.75`, cao hơn baseline `0.83`, trong khi classifier AUC thấp hơn `0.10`. N4 cũng tăng avg mAP `0.55` nhưng giảm AUC `0.40`.

Điều này cho thấy hạ các normal snippet có score quá cao có thể giúp nhánh văn bản xác định khoảng thời gian sự kiện tốt hơn. Tuy nhiên, thay đổi đó chưa đủ để mọi anomaly frame trên tập kiểm tra được xếp cao hơn mọi normal frame, nên AUC không tăng.

### 6.3. Abnormal Separation Cải Thiện Alignment Và mAP Rõ Hơn Classifier AUC

A2 cho avg mAP tốt nhất trong nhóm baseline-only: `7.94`, tăng `1.02` điểm so với baseline cùng protocol. Alignment AP cũng tăng từ `26.15` lên `27.67`. Ngược lại, classifier AUC giảm từ `87.44` xuống `87.24`.

Kết quả phù hợp với bản chất của loss: nó tối ưu khoảng cách high-low trong từng video, không trực tiếp tối ưu thứ hạng toàn cục giữa anomaly frame của video này và normal frame của video khác.

### 6.4. Abnormal Separation Có Tác Dụng Khác Nhau Trên Baseline Và Mô Hình Prompt

Trên VadCLIP tự huấn luyện, abnormal separation làm classifier AUC giảm nhẹ khoảng `0.20-0.21` điểm. Khi mỗi lớp được biểu diễn bằng năm keyword prompt, cùng phương pháp lại tăng classifier AUC `0.37` điểm, từ `87.15` lên `87.52`.

Một cách giải thích hợp lý là prompt đã thay đổi thứ tự điểm của các snippet. Nếu nhóm điểm cao ban đầu chứa nhiều abnormal snippet, loss sẽ củng cố đúng bằng chứng. Ngược lại, nếu top-k chứa nhiều normal/background snippet, việc tiếp tục đẩy nhóm này lên sẽ làm sai lầm nghiêm trọng hơn.

### 6.5. mAP Tăng Không Đồng Nghĩa AUC Tăng

Frame-level AUC kiểm tra xem anomaly frame có được xếp cao hơn normal frame trên toàn bộ tập kiểm tra hay không. Detection mAP lại kiểm tra mô hình có xác định đúng **loại sự kiện** và **khoảng thời gian xảy ra sự kiện** hay không; chỉ số này được tính từ nhánh alignment.

Do đó, một mô hình có thể xác định đoạn sự kiện và loại sự kiện rõ hơn, làm mAP tăng, nhưng vẫn chấm một số normal frame quá cao hoặc bỏ sót anomaly frame ở video khác. A2 là ví dụ rõ nhất: avg mAP tăng nhưng classifier AUC vẫn thấp hơn baseline.

### 6.6. Rủi Ro Chính Của Việc Chọn Top-k Bằng Dự Đoán Hiện Tại

Các hàm mất mát bổ sung không có frame label trong lúc huấn luyện. Nhóm high và low đều được chọn từ điểm do chính mô hình tạo ra. Nếu thứ tự ban đầu sai:

- normal top-k có thể hạ một vùng thị giác khó nhưng hữu ích cho khả năng tổng quát hóa;
- abnormal separation có thể đẩy false-high lên và kéo missed anomaly xuống;
- tỷ lệ top-k lớn hơn làm tăng khả năng trộn anomaly và background trong cùng nhóm.

Vì vậy, loss top-k có thể cải thiện gap nội bộ mà không cải thiện global AUC.

## 7. Phân Tích Nguyên Nhân Top-k Suppression Chưa Hiệu Quả

Các bảng kết quả phía trên chỉ cho biết mô hình nào đạt điểm cao hơn, nhưng chưa trả lời được **vì sao** top-k suppression có lúc cải thiện và có lúc làm kết quả giảm. Phần này lần lượt kiểm tra ba vấn đề dễ hiểu:

1. Những snippet được mô hình chấm điểm cao có thực sự chứa anomaly frame không?
2. Các đoạn bị chọn đúng và chọn sai nằm ở đâu trong video, có đặc điểm gì giống nhau?
3. Sai sót bắt nguồn từ đặc trưng hình ảnh hay từ cách mô hình chọn bằng chứng để học?

Bốn mô hình được so sánh xuyên suốt gồm:

| Tên trong phần phân tích | Mô tả |
|---|---|
| Baseline tự train | VadCLIP huấn luyện từ đầu, không có hàm mất mát bổ sung |
| Baseline + abnormal top-k | VadCLIP train từ đầu và thêm abnormal high-low separation |
| Mô hình prompt | VadCLIP sử dụng nhiều câu mô tả cho mỗi lớp, không có hàm mất mát bổ sung |
| Mô hình prompt + abnormal top-k | Mô hình prompt kết hợp abnormal high-low separation |

Trong phần này, **snippet** là một đoạn ngắn gồm `16` frame liên tiếp; **anomaly score** càng gần `1` nghĩa là mô hình càng tin snippet đó là abnormal; **top-k** là nhóm `15%` snippet có điểm cao nhất trong một video.

Các kết quả dưới đây dùng **baseline tự huấn luyện**, không dùng checkpoint AUC `88.02` của tác giả làm mốc phân tích. Một số checkpoint được lưu ở các thời điểm khác nhau nên AUC có thể lệch nhẹ so với bảng kết quả huấn luyện. Tuy vậy, các xu hướng về điểm dự đoán, vị trí theo thời gian và visual feature vẫn nhất quán.

### 7.1. Cách Xác Định Mô Hình Chọn Đúng Hay Sai

#### Tại sao cần quy ước này

Trong quá trình huấn luyện, mô hình chỉ biết một video là normal hay abnormal, chứ không biết chính xác anomaly xảy ra ở frame nào. Sau khi huấn luyện xong, nhãn thời gian của tập kiểm tra mới được dùng để kiểm tra xem các snippet có score cao có trùng với anomaly thật hay không. Các nhãn này chỉ phục vụ phân tích, hoàn toàn không được dùng để huấn luyện lại mô hình.

Mỗi anomaly score đầu ra của VadCLIP tương ứng với `16` frame liên tiếp. Với snippet thứ `t`, tỷ lệ anomaly frame được tính như sau:

```text
anomaly_fraction(t) = số anomaly frame trong snippet t / tổng số frame của snippet t
```

Snippet được xem là abnormal nếu chứa ít nhất một anomaly frame. Trong mỗi video, `15%` snippet có score cao nhất được xem là nhóm mô hình tin tưởng nhất. Các snippet sau đó được chia thành bốn nhóm:

| Nhóm | Ground truth | Có thuộc top 15% score hay không | Ý nghĩa |
|---|---|---|---|
| `true_high` | Abnormal | Có | Abnormal snippet được mô hình chấm cao đúng |
| `false_high` | Normal/background | Có | Normal snippet bị mô hình chấm cao nhầm |
| `missed_anomaly` | Abnormal | Không | Abnormal snippet thật nhưng không được mô hình ưu tiên |
| `true_low` | Normal/background | Không | Normal snippet được mô hình chấm thấp đúng |

`Missed anomaly` không nhất thiết có score dưới `0.5`. Thuật ngữ này chỉ có nghĩa abnormal snippet đó không lọt vào nhóm `15%` score cao nhất của chính video.

### 7.2. Kiểm Tra Khả Năng Tách Abnormal Khỏi Normal

#### Tại sao thực hiện

Mục tiêu của mô hình là chấm anomaly score của abnormal frame cao hơn normal frame. Tuy nhiên, sự phân biệt này có thể được nhìn theo hai cách khác nhau:

- **Khoảng cách trong cùng video:** trong một abnormal video, nhóm snippet có score cao hơn nhóm snippet có score thấp bao nhiêu. Tên kỹ thuật của phép đo này là `intra-video gap`.
- **Khoảng cách trên toàn bộ tập kiểm tra:** score trung bình của tất cả anomaly frame cao hơn score trung bình của tất cả normal frame bao nhiêu. Tên kỹ thuật là `global gap`.

Phải tách hai phép đo vì một mô hình có thể tạo khoảng cách rất rõ bên trong từng video nhưng vẫn chấm một số normal frame ở video khác cao hơn anomaly frame. Khi đó khoảng cách trong video tăng nhưng AUC toàn cục vẫn giảm.

Global gap được tính bằng:

```text
global_gap = mean(score_anomaly) - mean(score_normal/background)
```

#### Điểm Trung Bình Của Hai Nhóm

| Mô hình | Nhánh | AUC | AP | Anomaly score TB | Normal score TB | Chênh lệch toàn cục |
|---|---|---:|---:|---:|---:|---:|
| Baseline tự train | Classifier | 87.44 | 32.07 | 0.843 | 0.267 | 0.576 |
| Baseline tự train | Alignment | 85.64 | 26.15 | 0.897 | 0.342 | 0.555 |
| Baseline + abnormal top-k | Classifier | 87.24 | 31.98 | 0.819 | 0.245 | 0.575 |
| Baseline + abnormal top-k | Alignment | 86.15 | 27.67 | 0.897 | 0.321 | 0.576 |
| Mô hình prompt | Classifier | 87.25 | 33.08 | 0.822 | 0.247 | 0.575 |
| Mô hình prompt | Alignment | 85.28 | 25.73 | 0.862 | 0.317 | 0.546 |
| Mô hình prompt + abnormal top-k | Classifier | 87.38 | 33.15 | 0.828 | 0.247 | **0.581** |
| Mô hình prompt + abnormal top-k | Alignment | 85.24 | 25.25 | 0.869 | 0.319 | 0.549 |

Mô hình prompt kết hợp abnormal top-k có khoảng cách trung bình lớn nhất (`0.581`) nhưng AUC vẫn thấp hơn baseline. Lý do là hai giá trị trung bình không phản ánh hết các trường hợp khó. AUC so sánh thứ tự giữa anomaly frame và normal frame trên toàn bộ tập kiểm tra. Chỉ cần vẫn còn một nhóm normal frame có score rất cao và một nhóm anomaly frame có score rất thấp, nhiều cặp sẽ bị xếp sai và AUC sẽ giảm.

#### Kiểm Tra Những Trường Hợp Bị Chấm Sai Nghiêm Trọng

Điểm trung bình có thể che khuất một nhóm nhỏ nhưng rất quan trọng. Vì vậy, ba mốc sau được dùng để quan sát các trường hợp khó nhất:

- `Positive p10`: ngưỡng score của nhóm `10%` anomaly frame bị chấm thấp nhất. Giá trị càng thấp, hiện tượng missed anomaly càng nghiêm trọng.
- `Negative p95`: chỉ `5%` normal frame có score cao hơn mốc này. Nó cho biết nhóm false-high đang nghiêm trọng đến đâu.
- `Negative p99`: chỉ `1%` normal frame có score cao hơn mốc này. Đây là nhóm normal frame bị mô hình chấm gần như chắc chắn là abnormal.

| Mô hình | Mốc 10% bất thường thấp nhất | Trung vị bất thường | Mốc 5% bình thường cao nhất | Mốc 1% bình thường cao nhất |
|---|---:|---:|---:|---:|
| Baseline tự train | **0.286** | 0.976 | **0.995** | **0.99970** |
| Baseline + abnormal top-k | 0.202 | 0.984 | 0.997 | 0.99980 |
| Mô hình prompt | 0.154 | 0.990 | 0.998 | 0.99996 |
| Mô hình prompt + abnormal top-k | 0.208 | **0.991** | 0.999 | 0.99994 |

Prompt làm trung vị điểm anomaly tăng từ `0.976` lên khoảng `0.990`, nghĩa là phần lớn anomaly được mô hình chấm rất cao. Tuy nhiên, `positive_p10` giảm từ `0.286` xuống `0.154`, nghĩa là nhóm anomaly khó lại bị đẩy thấp hơn. Đồng thời, `negative_p95` và `negative_p99` đều tiến sát `1`, cho thấy vẫn tồn tại normal/background được mô hình chấm gần tuyệt đối.

Abnormal top-k nâng `positive_p10` của mô hình prompt từ `0.154` lên `0.208`, nhưng cũng làm `negative_p95` tăng từ `0.998` lên `0.999`. Vì vậy loss đã hỗ trợ một phần anomaly khó nhưng chưa kiểm soát được false-high ở đuôi trên.

#### Trong Top-k Có Bao Nhiêu Phần Trăm Anomaly Frame?

Đây là thống kê trực tiếp nhất để kiểm tra top-k có chọn đúng anomaly hay không. Với mỗi abnormal video, quy trình thực hiện như sau:

1. Chọn `15%` snippet có anomaly score cao nhất.
2. Mỗi snippet được mở rộng về `16` frame tương ứng.
3. Đếm tổng số anomaly frame nằm trong các snippet đã chọn.
4. Chia cho tổng số frame thuộc top-k của video đó.
5. Lấy trung bình tỷ lệ của tất cả abnormal video.

```text
anomaly_frame_ratio_in_topk(video)
    = số anomaly frame nằm trong top-k
    / tổng số frame nằm trong top-k

anomaly_frame_ratio_in_topk
    = trung bình tỷ lệ của tất cả abnormal video
```

Ví dụ, top-k của một video gồm `20` snippet, tương ứng `320` frame. Nếu `128` frame trong số đó là anomaly frame thì tỷ lệ anomaly frame trong top-k của video bằng `128 / 320 = 40%`. Mỗi video đóng góp như nhau vào kết quả cuối, bất kể video dài hay ngắn.

`Intra-video gap` trong cùng bảng cho biết, ở mỗi video, điểm trung bình của nhóm top-k cao hơn nhóm điểm thấp nhất bao nhiêu. Giá trị lớn chỉ cho thấy mô hình rất tự tin vào lựa chọn của mình; nó không đảm bảo lựa chọn đó đúng.

| Mô hình | Anomaly frame trong top-k | Chênh lệch cao-thấp trong video |
|---|---:|---:|
| Baseline tự train | **35.75%** | 0.408 |
| Baseline + abnormal top-k | 33.59% | 0.446 |
| Mô hình prompt | 34.48% | 0.436 |
| Mô hình prompt + abnormal top-k | 33.49% | **0.448** |

Ở baseline tự train, chỉ `35,75%` frame thuộc top-k là anomaly frame; khoảng `64,25%` còn lại là normal/background frame. Sau khi thêm abnormal top-k suppression, tỷ lệ anomaly frame còn giảm xuống `33,59%`. Mô hình prompt cũng chỉ đạt `34,48%`, và giảm xuống `33,49%` khi kết hợp abnormal top-k.

Trong khi đó, chênh lệch score giữa nhóm cao và thấp lại tăng. Nói cách khác, mô hình trở nên tự tin hơn vào top-k nhưng top-k không chứa nhiều anomaly frame hơn. Vì phần lớn frame trong top-k vẫn là normal/background, loss có thể đồng thời đẩy `true-high` và `false-high` lên cao hơn.

#### Hai Nhánh Của Mô Hình Có Chọn Cùng Một Đoạn Không?

VadCLIP tạo điểm theo hai cách: nhánh classifier dự đoán trực tiếp mức độ bất thường, còn nhánh alignment đo mức phù hợp với các lớp sự kiện. Hệ số tương quan (`correlation`) được dùng để kiểm tra hai nhánh có cùng tăng hoặc cùng giảm điểm trên một frame hay không. Giá trị gần `1` nghĩa là hai nhánh thường đồng ý; giá trị thấp hơn nghĩa là chúng có xu hướng chọn các vùng khác nhau.

| Mô hình | Toàn bộ frame | Anomaly frame | Normal frame |
|---|---:|---:|---:|
| Baseline tự train | **0.946** | **0.899** | **0.941** |
| Baseline + abnormal top-k | 0.921 | 0.859 | 0.914 |
| Mô hình prompt | 0.928 | 0.893 | 0.921 |
| Mô hình prompt + abnormal top-k | 0.924 | 0.862 | 0.918 |

Việc thêm prompt hoặc abnormal top-k làm mức đồng thuận giữa hai nhánh giảm, đặc biệt trên anomaly frame. Điều này không nhất thiết luôn xấu, nhưng trong bài toán chỉ có nhãn video, nó khiến các nhãn tạm do mô hình tự suy ra kém ổn định: hai nhánh có thể chọn các vùng khác nhau và củng cố các lỗi khác nhau.

#### Kết Quả Có Giống Nhau Ở Mọi Lớp Không?

| Lớp | Baseline AUC | Mô hình prompt AUC | Prompt + top-k AUC |
|---|---:|---:|---:|
| Arrest | 65.74 | 69.32 | 68.96 |
| Arson | 76.89 | 75.40 | 77.00 |
| Assault | 86.80 | **90.62** | 90.26 |
| Explosion | 48.10 | 50.92 | **53.14** |
| Fighting | **67.25** | 66.07 | 66.28 |
| Shoplifting | **66.37** | 63.82 | 63.04 |
| Vandalism | 75.27 | **77.64** | 77.46 |

Tác động của prompt và top-k không giống nhau giữa các lớp. `Assault`, `Arrest`, `Vandalism` và phần nào `Explosion` được cải thiện, trong khi `Shoplifting` và `Fighting` giảm. Do đó, một hàm mất mát hoặc một ngưỡng chung khó có thể phù hợp với tất cả loại sự kiện.

### 7.3. Các Snippet Bị Chọn Đúng Và Chọn Sai Có Nằm Liền Nhau Không?

#### Tại sao thực hiện

Một giả thuyết ban đầu là abnormal snippet thường kéo dài liên tục, còn normal snippet bị chấm cao nhầm chỉ xuất hiện riêng lẻ. Nếu giả thuyết đúng, có thể nhận biết và hạ các score cao đứng một mình. Vì vậy, vị trí của các snippet được kiểm tra theo thứ tự thời gian trước khi xây dựng loss mới.

#### Abnormal Snippet Kéo Dài Như Thế Nào?

Trên `140` abnormal test videos:

| Thống kê | Kết quả |
|---|---:|
| Tổng số abnormal snippet | 5.400 |
| Số abnormal snippet trung bình/video | 38.57 |
| Số chuỗi bất thường trung bình/video | 1.11 |
| Chuỗi dài nhất trung bình | 36.30 snippet |
| Độ dài chuỗi trung bình | 35.25 snippet |
| Abnormal snippet nằm cạnh abnormal snippet khác | 100% |

Kết quả cho thấy anomaly thường tạo thành một chuỗi dài, trung bình khoảng `35` abnormal snippet liên tiếp. Ranh giới snippet cũng không phải nguồn lỗi chính:

| Độ tinh khiết snippet | Kết quả |
|---|---:|
| Tỷ lệ anomaly frame trung bình trong một abnormal snippet | 97.41% |
| Snippet chứa đủ 16/16 anomaly frame | 95.02% |
| Snippet chứa lẫn anomaly frame và normal frame | 4.98% |

Như vậy, missed anomaly không chủ yếu xuất hiện vì một snippet bị trộn quá nhiều frame normal.

#### Mô Hình Đã Chọn Đúng Bao Nhiêu Snippet?

Các giá trị dưới đây là **tổng số snippet của 140 video bất thường**. Dấu chấm được dùng làm dấu phân cách hàng nghìn; ví dụ `1.613` nghĩa là một nghìn sáu trăm mười ba snippet.

| Mô hình | Abnormal được chấm cao đúng | Normal bị chấm cao nhầm | Abnormal bị bỏ sót | Normal được chấm thấp đúng |
|---|---:|---:|---:|---:|
| Baseline tự train | 1.613 | 8.924 | 3.787 | 55.044 |
| Baseline + abnormal top-k | 1.606 | 8.931 | 3.794 | 55.037 |
| Mô hình prompt | 1.597 | 8.940 | 3.803 | 55.028 |
| Mô hình prompt + abnormal top-k | 1.626 | 8.911 | 3.774 | 55.057 |

Với baseline, nhóm `15%` score cao nhất gồm `10.537` snippet, nhưng chỉ `1.613` snippet là abnormal và `8.924` snippet là normal/background bị chọn nhầm. Khi gộp toàn bộ snippet của tất cả video, tỷ lệ abnormal snippet trong top-k là `15,31%`; đồng thời mô hình chỉ chọn được `29,87%` tổng số abnormal snippet.

Con số `15,31%` không mâu thuẫn với tỷ lệ `35,75%` anomaly frame trong top-k ở phần trước. `15,31%` được tính bằng cách gộp toàn bộ **snippet** của tất cả video, còn `35,75%` được tính trên **frame** trong từng video rồi mới lấy trung bình. Hai phép đo dùng đơn vị và cách lấy trung bình khác nhau, nhưng cùng cho thấy phần lớn top-k vẫn là normal/background.

#### Đoạn Bình Thường Bị Chấm Cao Cũng Nằm Liền Nhau

Kết quả bác bỏ giả thuyết rằng dự đoán sai chỉ là các điểm cao xuất hiện riêng lẻ:

- Hơn `99%` snippet được chọn có ít nhất một snippet được chọn khác nằm ngay bên cạnh.
- Normal snippet bị chấm cao trong abnormal video tạo chuỗi dài trung bình `10,6` snippet, trung vị `7` và tối đa `76`.
- Các đoạn điểm cao trong video hoàn toàn bình thường tạo chuỗi dài trung bình `11,3` snippet, trung vị `7` và tối đa `168`.
- Khoảng `99,3%` normal snippet có score cao nằm cạnh một snippet có score cao khác.

Do đó, chỉ nhìn vào tính liên tục theo thời gian không đủ để phân biệt anomaly với hard-negative: cả hai đều có thể kéo dài, liền kề và ổn định.

### 7.4. Đặc Trưng Hình Ảnh Có Còn Phân Biệt Được Hai Nhóm Không?

#### Tại sao thực hiện

Sau khi không thể phân biệt hai nhóm bằng vị trí thời gian, câu hỏi tiếp theo là: **đặc trưng hình ảnh đã mất thông tin cần thiết, hay thông tin vẫn còn nhưng classifier chưa sử dụng tốt?** Ba vị trí trong mô hình được kiểm tra: đặc trưng CLIP ban đầu, đặc trưng sau khi xử lý quan hệ thời gian và đặc trưng cuối ngay trước phần dự đoán.

Để kiểm tra, một bộ phân loại đơn giản (`linear probe`) được huấn luyện trên các đặc trưng đã trích xuất bằng nhãn snippet thật. Nếu bộ phân loại đơn giản đạt AUC cao, nghĩa là đặc trưng vẫn chứa thông tin phân biệt; khi đó điểm yếu nằm ở cách VadCLIP lựa chọn và học từ các đoạn video. Phép thử này chỉ dùng để chẩn đoán sau huấn luyện, không cung cấp thêm nhãn cho VadCLIP.

| Mô hình | Đặc trưng CLIP ban đầu | Sau xử lý thời gian | Đặc trưng hình ảnh cuối |
|---|---:|---:|---:|
| Baseline tự train | 80.00 | 81.11 | 83.00 |
| Baseline + abnormal top-k | 81.53 | 82.00 | 83.22 |
| Mô hình prompt | 82.12 | 84.12 | **86.37** |
| Mô hình prompt + abnormal top-k | 82.95 | 83.93 | 83.64 |

Mô hình prompt không có loss bổ sung tạo ra đặc trưng hình ảnh cuối tốt nhất. Khi thêm abnormal top-k, AUC của bộ phân loại chẩn đoán giảm từ `86.37` xuống `83.64`. Như vậy, loss mới có thể cải thiện một số điểm hoặc mAP nhưng đồng thời làm thông tin phân biệt bất thường-bình thường trong đặc trưng kém rõ hơn.

Một phép đo thứ hai kiểm tra xem các snippet có đặc trưng gần nhau có thường cùng là bất thường hoặc cùng là bình thường hay không (`kNN purity`). Giá trị càng gần `1`, hai nhóm càng được sắp xếp rõ ràng trong không gian đặc trưng. Kết quả cũng cho cùng xu hướng:

| Mô hình | Đặc trưng CLIP ban đầu | Sau xử lý thời gian | Đặc trưng hình ảnh cuối |
|---|---:|---:|---:|
| Baseline tự train | 0.889 | 0.903 | 0.912 |
| Baseline + abnormal top-k | 0.884 | 0.902 | 0.914 |
| Mô hình prompt | 0.897 | 0.909 | **0.920** |
| Mô hình prompt + abnormal top-k | 0.890 | 0.906 | 0.916 |

Các tầng xử lý quan hệ thời gian và quan hệ giữa snippet nhìn chung làm hai nhóm bất thường-bình thường rõ hơn. Prompt không phá hỏng đặc trưng hình ảnh. Điểm nghẽn nằm nhiều hơn ở việc MIL chọn đoạn nào làm bằng chứng và dùng các đoạn đó để huấn luyện classifier.

#### Đặc Trưng Của Các Nhóm Đúng Và Sai Giống Nhau Đến Mức Nào?

| Mô hình | False-high ↔ true-high | Missed anomaly ↔ true-high | False-high ↔ true-low | Missed anomaly ↔ false-high |
|---|---:|---:|---:|---:|
| Baseline tự train | 0.961 | 0.976 | 0.948 | 0.943 |
| Baseline + abnormal top-k | 0.952 | 0.972 | 0.939 | 0.927 |
| Mô hình prompt | **0.922** | **0.954** | **0.892** | **0.877** |
| Mô hình prompt + abnormal top-k | 0.933 | 0.963 | 0.908 | 0.898 |

Các normal/background snippet bị chấm nhầm có đặc trưng khá giống abnormal snippet, nhất là khi cùng nằm trong một abnormal video. Tuy nhiên, mô hình prompt đã làm các nhóm tách nhau rõ hơn baseline. Điều này cho thấy AUC giảm không hoàn toàn do đặc trưng hình ảnh kém; classifier chưa sử dụng hết thông tin đã có.

#### Normal Snippet Bị Nhầm Thành Lớp Nào?

Trong mô hình prompt kết hợp abnormal top-k, normal snippet có score cao thường được nhánh alignment gán cho những lớp có nội dung thị giác rộng:

| Class được gán | Tỷ lệ false-high |
|---|---:|
| RoadAccidents | 24.1% |
| Robbery | 20.4% |
| Burglary | 14.2% |
| Shoplifting | 7.8% |
| Stealing | 7.7% |
| Fighting | 7.3% |
| Arson | 7.2% |

Các prompt đóng góp nhiều nhất gồm `robbery robber intimidation theft` (`18.8%`), `roadAccidents vehicle collision roadway` (`17.6%`) và `burglary burglar carrying property` (`11.3%`). Những mô tả này phản ứng rộng với người, phương tiện, chuyển động và vật thể, nên có thể làm background nhận semantic anomaly cao.

### 7.5. So Sánh Thông Tin Có Trong Đặc Trưng Với Khả Năng Của Classifier

#### Tại sao thực hiện

Phần trước cho thấy đặc trưng còn chứa thông tin phân biệt, nhưng chưa cho biết classifier của VadCLIP đã tận dụng được bao nhiêu. Vì vậy, cùng một đặc trưng ngay trước classifier được đưa vào một bộ phân loại đơn giản có nhãn snippet, rồi kết quả của hai bộ phân loại được so sánh.

Đầu vào classifier của VadCLIP là:

```text
h = z + MLP2(z)
```

Trong đó `z` là đặc trưng hình ảnh cuối cùng và `MLP2` là tầng biến đổi của mô hình. Bộ phân loại chẩn đoán được học trực tiếp từ nhãn snippet thật, còn classifier VadCLIP chỉ học từ nhãn toàn video. Khoảng cách giữa hai kết quả cho biết VadCLIP còn bỏ phí bao nhiêu thông tin phân biệt đã tồn tại trong đặc trưng.

#### Khi Xét Toàn Bộ Tập Kiểm Tra

| Mô hình | Bộ phân loại trên đặc trưng cuối | Bộ phân loại trên đầu vào classifier | Classifier của VadCLIP |
|---|---:|---:|---:|
| Baseline tự train | 83.00 | 82.68 | **84.00** |
| Baseline + abnormal top-k | 81.70 | 81.75 | **82.65** |
| Mô hình prompt | **86.83** | 86.82 | 85.05 |
| Mô hình prompt + abnormal top-k | 84.23 | **84.40** | 84.07 |

#### Khi Chỉ Xét Các Video Có Sự Kiện Bất Thường

| Mô hình | Bộ phân loại trên đặc trưng cuối | Bộ phân loại trên đầu vào classifier | Classifier của VadCLIP | Phần thông tin chưa được khai thác |
|---|---:|---:|---:|---:|
| Baseline tự train | 61.59 | 61.11 | **63.50** | -2.40 |
| Baseline + abnormal top-k | 59.19 | 59.57 | **62.09** | -2.51 |
| Mô hình prompt | 67.64 | **68.21** | 61.45 | **+6.76** |
| Mô hình prompt + abnormal top-k | 66.18 | **66.40** | 62.23 | +4.17 |

Ở baseline, classifier đã khai thác đặc trưng tương đối tốt. Với mô hình prompt, bộ phân loại chẩn đoán cao hơn classifier VadCLIP tới `6.76` điểm AUC khi chỉ xét video bất thường. Nói đơn giản, mô hình **đã tạo ra đặc trưng hữu ích nhưng chưa học được cách chấm điểm đúng từ đặc trưng đó**. Nguyên nhân phù hợp nhất với các thống kê trước là MIL tập trung gradient vào top-k chứa nhiều lựa chọn sai.

### 7.6. Có Thể Nhận Biết Đoạn Bình Thường Bằng Một Kho Đặc Trưng Mẫu Không?

#### Tại sao thực hiện

Một ý tưởng trực quan là lưu một tập đặc trưng đại diện cho hành vi bình thường, gọi là `normal memory bank`. Nếu một snippet rất giống các mẫu trong kho này, có thể xem nó là bình thường và hạ điểm. Trước khi đưa ý tưởng vào huấn luyện, phép phân tích kiểm tra hai câu hỏi:

1. Kho mẫu có phân biệt được bất thường và bình thường nói chung không?
2. Quan trọng hơn, nó có phân biệt được normal snippet bị chấm cao với abnormal snippet đang bị bỏ sót không?

Memory bank sử dụng `500` Normal feature rows từ `373` video, tối đa `50.000` feature mỗi tầng và `32` KMeans prototypes. Normal similarity được tính bằng cosine similarity lớn nhất với một normal prototype:

```text
normal_similarity(x) = max_j cosine(x, normal_prototype_j)
```

#### Kho Mẫu Có Tách Được Bất Thường Và Bình Thường Nói Chung Không?

| Mô hình | Đặc trưng CLIP ban đầu | Đặc trưng hình ảnh cuối | Đầu vào classifier |
|---|---:|---:|---:|
| Baseline tự train | 56.10 | 83.73 | 83.62 |
| Baseline + abnormal top-k | 56.10 | 84.90 | 84.40 |
| Mô hình prompt | 56.10 | **85.35** | **85.15** |
| Mô hình prompt + abnormal top-k | 56.10 | 84.46 | 84.43 |

Sau các tầng xử lý của VadCLIP, kho mẫu phân biệt abnormal và normal nói chung khá tốt. Tuy nhiên, đây chưa phải bài toán khó cần giải quyết: mục tiêu thực tế là phân biệt normal snippet bị chấm cao với abnormal snippet đang bị bỏ sót.

#### Kho Mẫu Có Phân Biệt Được Dự Đoán Sai Với Bất Thường Bị Bỏ Sót Không?

Trong phép đo này, kết quả trên `50%` mới cho thấy normal snippet bị chấm cao giống kho normal hơn abnormal snippet bị bỏ sót:

| Mô hình | Đặc trưng CLIP ban đầu | Đặc trưng hình ảnh cuối | Đầu vào classifier |
|---|---:|---:|---:|
| Baseline tự train | 45.66 | 47.31 | 46.97 |
| Baseline + abnormal top-k | 45.55 | 44.33 | 45.47 |
| Mô hình prompt | 45.92 | 41.52 | 40.10 |
| Mô hình prompt + abnormal top-k | 45.02 | 40.64 | 40.49 |

Tất cả kết quả đều dưới `50%`, tức là quy tắc hoạt động ngược với mong muốn. Missed anomaly thường còn giống kho normal hơn false-high. Nếu hạ score mọi snippet giống kho này, mô hình có nguy cơ hạ chính các anomaly khó trước khi sửa được false-high.

Hiệu quả còn phụ thuộc class: classifier-input memory có tín hiệu với `Stealing` (`78.20%`) và `Vandalism` (`67.39%`), nhưng đi ngược mạnh ở `Explosion` (`16.53%`), `Assault` (`7.18%`), `Fighting` (`28.31%`) và `Shooting` (`27.24%`). Do đó không nên áp dụng một normal-memory loss giống nhau cho mọi class.

### 7.7. Thông Tin Toàn Video Có Làm Lan Truyền Dự Đoán Sai Không?

#### Tại sao thực hiện

VadCLIP dùng điểm từ nhánh classifier để tóm tắt nội dung toàn video thành một vector ngữ cảnh. Vector này được bổ sung vào biểu diễn văn bản trước khi nhánh alignment tính điểm. Nếu classifier đang tập trung vào một đoạn nền bị chấm nhầm, ngữ cảnh thu được có thể mang thông tin sai và ảnh hưởng đến nhiều snippet khác trong video.

Để kiểm tra mà không cần huấn luyện lại, hệ số `gamma` được dùng để tăng hoặc giảm ảnh hưởng của ngữ cảnh:

```text
text_feature_gamma = text_feature + gamma * video_context
```

- `gamma = 0`: không sử dụng thông tin tóm tắt toàn video.
- `gamma = 1`: giữ nguyên cách mô hình đã được huấn luyện.

Sanity check xác nhận `gamma = 1` tái tạo logits gốc với sai số tuyệt đối bằng `0`.

| Mô hình | AUC2 gamma=1 | AUC2 tốt nhất | AP2 gamma=1 | AP2 gamma=0 |
|---|---:|---:|---:|---:|
| Baseline tự train | 85.64 | 85.76 tại 0.25 | 26.15 | 26.38 |
| Baseline + abnormal top-k | 85.62 | **86.08 tại 0** | 26.20 | **26.86** |
| Mô hình prompt | 85.28 | 85.28 tại 0.75 | 25.73 | **26.39** |
| Mô hình prompt + abnormal top-k | 85.29 | **85.46 tại 0** | 25.50 | **26.51** |

Giảm ảnh hưởng của ngữ cảnh làm AP của nhánh alignment tăng ở cả bốn mô hình. Điều này cho thấy ngữ cảnh toàn video đôi khi lan truyền dự đoán sai hoặc làm mô hình quá tự tin vào vùng đã chọn nhầm. Tuy nhiên, AUC chỉ cải thiện nhẹ và không đồng đều, nên ngữ cảnh không phải nguyên nhân duy nhất.

Ảnh hưởng cũng khác nhau giữa các class:

- Với `Arrest`, bỏ context cải thiện khoảng cách `missed anomaly - false high` từ `-0.345` lên `-0.179` ở mô hình prompt + top-k.
- `Vandalism` cải thiện nhẹ khi bỏ context.
- `Explosion` xấu hơn, khoảng cách giảm từ `-0.048` xuống `-0.080`.
- `Arson` đang có khoảng cách `+0.048` ở gamma 1 nhưng giảm thành `-0.035` ở gamma 0.

Ngữ cảnh cũng có mặt tích cực: nó giúp hạ điểm background trong video Normal. Với mô hình prompt + abnormal top-k, điểm false-high của nhánh alignment trong video Normal tăng từ `0.124` lên `0.304` khi bỏ ngữ cảnh. Vì vậy, không nên xóa toàn bộ cơ chế này; nếu điều chỉnh, cần kiểm soát riêng theo snippet hoặc theo lớp.

### 7.8. Những Kết Luận Chính Từ Phân Tích

1. **Một số ít dự đoán rất sai làm AUC giảm.** Mô hình vẫn để một nhóm anomaly frame có score rất thấp và một nhóm normal frame có score gần `1`, dù score trung bình của hai nhóm cách nhau khá xa.
2. **Nhóm top-k chưa đủ đáng tin để đóng vai trò nhãn tạm.** Trung bình chỉ khoảng một phần ba các đoạn điểm cao trùng với sự kiện thật. Loss vì vậy có thể củng cố cả lựa chọn đúng lẫn lựa chọn sai.
3. **Mô hình tự tin hơn không có nghĩa mô hình chính xác hơn.** Top-k suppression làm khoảng cách giữa nhóm cao và thấp trong cùng video tăng, nhưng không làm tỷ lệ chọn đúng tăng.
4. **Cách chia video thành snippet không phải nguyên nhân chính.** Khoảng `95%` abnormal snippet chứa đủ `16/16` anomaly frame và các snippet này thường nằm thành chuỗi dài.
5. **Không thể nhận biết dự đoán sai chỉ bằng tính liên tục.** Normal snippet bị chấm cao cũng thường nằm thành chuỗi dài giống anomaly thật.
6. **Đặc trưng hình ảnh vẫn chứa thông tin hữu ích.** Bộ phân loại chẩn đoán có thể khai thác đặc trưng tốt hơn classifier của VadCLIP, đặc biệt ở mô hình prompt. Điểm nghẽn nằm nhiều ở cách MIL chọn đoạn để học.
7. **Abnormal top-k tạo ra sự đánh đổi.** Nó có thể cải thiện AP hoặc mAP nhưng đồng thời làm thông tin phân biệt trong đặc trưng kém rõ hơn.
8. **Kho đặc trưng normal không giải quyết được trường hợp khó.** Missed anomaly còn có thể giống normal hơn false-high, nên hạ score theo độ giống normal sẽ dễ sửa nhầm đối tượng.
9. **Ngữ cảnh toàn video vừa có lợi vừa có hại.** Nó giúp hạ điểm nền ở một số trường hợp nhưng cũng có thể lan truyền thông tin từ vùng bị chọn sai. Vì tác động khác nhau theo lớp, không nên loại bỏ toàn bộ.
10. **Vấn đề trung tâm là cách MIL lựa chọn bằng chứng.** Khi mô hình chọn nhầm background vào top-k, gradient tiếp tục củng cố vùng sai và missed anomaly không nhận được đủ tín hiệu học.

Vòng lặp lỗi hiện tại có thể mô tả như sau:

```text
Classifier ban đầu chọn top-k chưa chính xác
→ nhiều background được xem như positive evidence
→ temporal, graph và video context lan truyền tín hiệu từ vùng được chọn sai
→ MIL tiếp tục tăng false-high và bỏ qua missed anomaly
→ thứ hạng top-k ở epoch sau tiếp tục bị khóa vào cùng evidence
```

## 8. Kết Luận

Kết quả hiện tại chưa cho thấy top-k suppression đơn lẻ có thể vượt classifier AUC của baseline gốc `88.02`. Tuy nhiên, một số cấu hình tạo được trade-off có giá trị:

- N2 là cấu hình cân bằng nhất cho baseline-only: classifier AUC gần như giữ nguyên, trong khi alignment AUC, alignment AP và avg mAP đều tăng.
- N3 cải thiện avg mAP tương đối tốt với mức giảm classifier AUC nhỏ.
- A2 đạt avg mAP tốt nhất trong nhóm baseline-only, nhưng không cải thiện classifier AUC.
- Năm keyword prompt kết hợp abnormal separation đạt classifier AUC `87.52`, cao hơn mô hình chưa regularize `0.37` điểm. Điều này cho thấy regularization có thể hữu ích khi mô hình ban đầu đã chọn tương đối đúng các đoạn nổi bật.
- Normal suppression trên một bộ prompt khác lại làm tất cả metric giảm, cho thấy hiệu quả phụ thuộc mạnh vào cách mô hình sắp xếp điểm của các snippet.

Kết luận quan trọng nhất là top-k regularization không chỉ phụ thuộc vào `lambda`, margin hay tỷ lệ top-k. Chất lượng của nhóm snippet được chọn mới là điều quyết định. Nếu top-k chứa nhiều false-high hoặc bỏ qua nhiều missed anomaly, regularization sẽ củng cố sai lầm thay vì sửa chúng. Các phân tích feature probe cho thấy vẫn còn khả năng cải thiện mà không nhất thiết phải thay visual encoder; trọng tâm cần chuyển từ việc tăng suppression đơn thuần sang cơ chế lựa chọn evidence đáng tin cậy và đa dạng hơn trong MIL.
