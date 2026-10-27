# Mô Tả Các Top-K Regularization Loss Cho VadCLIP

## 1. Mục Tiêu

VadCLIP là mô hình phát hiện bất thường trong video theo hướng weakly-supervised. Trong quá trình training, mô hình không có nhãn chính xác cho từng frame/snippet, mà chủ yếu dùng nhãn cấp video. Vì vậy, mô hình phải tự học đoạn nào trong video là quan trọng thông qua cơ chế Multiple Instance Learning (MIL).

Trong các thí nghiệm mở rộng, ba regularization loss dạng top-k đã được xây dựng để điều chỉnh phân phối anomaly score:

| Regularizer | Áp dụng trên | Mục tiêu chính |
|---|---|---|
| Normal Top-K Suppression | video Normal | kéo các score bất thường cao giả trong video Normal xuống |
| Abnormal Background Separation | video Abnormal | tách đoạn score cao khỏi đoạn nền score thấp trong cùng video bất thường |
| Ranking Margin | batch có Normal và Abnormal | ép score của video Abnormal cao hơn video Normal |

Phần cần đưa sang VadCLIP gốc hiện tại là **Abnormal Background Separation**, tức regularizer áp dụng trên **video Abnormal**, không phải Normal Top-K Suppression.

## 2. Loss Gốc Của VadCLIP

Loss gốc của VadCLIP gồm ba thành phần:

```text
L_original = L_cls + L_mil + L_text_reg
```

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| `L_cls` | loss phân loại normal/abnormal từ nhánh classifier `logits1` |
| `L_mil` | MIL loss theo class từ nhánh visual-text alignment `logits2` |
| `L_text_reg` | regularization trên text embedding, giúp embedding Normal tách khỏi các class bất thường |

### 2.1. Nhánh Classifier

Với mỗi snippet `t`, nhánh classifier sinh ra `logits1(t)`. Anomaly score từ nhánh này là:

```text
s_cls(t) = sigmoid(logits1(t))
```

Trong `L_cls`, VadCLIP chọn top-k snippet có score cao nhất trong video:

```text
k_mil = floor(T / 16) + 1
```

Sau đó lấy trung bình top-k score để tạo video-level prediction:

```text
p_video = mean(top-k(s_cls(t)))
```

Loss:

```text
L_cls = BCE(p_video, y)
```

Với:

```text
y = 0 nếu video là Normal
y = 1 nếu video là Abnormal
```

### 2.2. Nhánh Alignment

Nhánh alignment sinh ra `logits2(t)`, thể hiện độ tương đồng giữa visual feature của snippet và text embedding của các class.

Từ `logits2`, anomaly score theo nhánh alignment được định nghĩa là:

```text
s_align(t) = 1 - softmax(logits2(t))_Normal
```

Nếu snippet càng ít giống class Normal, `s_align(t)` càng cao.

### 2.3. Text Regularization

Gọi:

```text
e_0 = normalized text embedding của class Normal
e_j = normalized text embedding của class bất thường j
```

VadCLIP gốc dùng regularization:

```text
L_text_reg = 0.1 * (1 / 13) * sum_{j=1}^{13} | e_0^T e_j |
```

Ý nghĩa là làm embedding Normal bớt gần các embedding bất thường.

## 3. Anomaly Score Dùng Chung Cho Các Regularizer

Ba regularizer đều dùng anomaly score từ một hoặc cả hai nhánh:

```text
s_cls(t) = sigmoid(logits1(t))
s_align(t) = 1 - softmax(logits2(t))_Normal
```

Tham số:

```text
regularize_score_branch = classifier | alignment | both
```

Nếu dùng `both`, loss được tính riêng trên `s_cls` và `s_align`, sau đó lấy trung bình:

```text
L_regularizer =
  (1 / 2) * (L_regularizer^classifier + L_regularizer^alignment)
```

Trong các thí nghiệm gần đây, cấu hình thường dùng là:

```text
regularize_score_branch = both
top_k_ratio = 0.15
```

## 4. Regularizer 1: Normal Top-K Suppression

Regularizer này áp dụng trên **video Normal**.

Điều kiện chọn video:

```text
class(video) = Normal
```

Với video Normal `i` có `T_i` snippet hợp lệ:

```text
k_i = max(1, floor(top_k_ratio * T_i))
```

Chọn top-k snippet có anomaly score cao nhất:

```text
H_i = top-k snippets theo s(t)
```

Loss cho video `i`:

```text
L_normal_i =
  (1 / k_i) * sum_{t thuộc H_i} s(t)
```

Loss trên batch:

```text
L_normal_topk =
  (1 / |N_normal|) * sum_{i thuộc N_normal} L_normal_i
```

Ý nghĩa: nếu một video Normal có vài đoạn bị model gán anomaly score cao, loss này sẽ phạt trực tiếp các đoạn đó.

Loss tổng nếu chỉ bật regularizer này:

```text
L_total =
  L_original
+ lambda_normal_topk * L_normal_topk
```

Đây là logic **không phải** phần đang cần đưa sang VadCLIP gốc ở bước hiện tại, nhưng nó là một trong ba regularizer đã được xây trước đó.

## 5. Regularizer 2: Abnormal Background Separation

Đây là regularizer áp dụng trên **video Abnormal** và là phần cần đưa sang VadCLIP gốc hiện tại.

Điều kiện chọn video:

```text
class(video) != Normal
```

Ý tưởng chính:

```text
Trong cùng một video bất thường,
đoạn thật sự bất thường nên có score cao hơn đoạn nền hoặc đoạn ít liên quan.
```

Vì training không có frame label, không biết chính xác snippet nào là bất thường. Do đó, dùng chính anomaly score hiện tại của model để chia tương đối:

- nhóm score cao: xem như các snippet nghi ngờ bất thường;
- nhóm score thấp: xem như background hoặc normal-like snippet trong video bất thường.

Với video abnormal `i` có `T_i` snippet hợp lệ:

```text
k_i = max(1, floor(top_k_ratio * T_i))
```

Chọn hai nhóm:

```text
H_i = top-k snippets có anomaly score cao nhất
L_i = top-k snippets có anomaly score thấp nhất
```

Regularizer yêu cầu nhóm score cao phải tách khỏi nhóm score thấp ít nhất một khoảng `margin`:

```text
min_{t thuộc H_i} s(t) >= max_{t thuộc L_i} s(t) + margin
```

Viết thành hinge loss:

```text
L_abnormal_bg_i =
  max(0, max_{t thuộc L_i} s(t) - min_{t thuộc H_i} s(t) + margin)
```

Loss trên batch:

```text
L_abnormal_bg =
  (1 / |N_abnormal|) * sum_{i thuộc N_abnormal} L_abnormal_bg_i
```

Nếu dùng cả hai nhánh classifier và alignment:

```text
L_abnormal_bg =
  (1 / 2) * (L_abnormal_bg^classifier + L_abnormal_bg^alignment)
```

Loss tổng sau khi thêm vào VadCLIP gốc:

```text
L_total =
  L_cls
+ L_mil
+ L_text_reg
+ lambda_abnormal_bg(epoch) * L_abnormal_bg
```

Trong logic multi-prompt trước đó, regularizer này có warm-up:

```text
lambda_abnormal_bg(epoch) =
  0                       nếu epoch < abnormal_bg_warmup_epochs
  lambda_abnormal_bg      nếu epoch >= abnormal_bg_warmup_epochs
```

Ví dụ cấu hình đã dùng:

```text
lambda_abnormal_bg = 0.05
abnormal_bg_margin = 0.15
abnormal_bg_warmup_epochs = 1
top_k_ratio = 0.15
regularize_score_branch = both
```

### 5.1. Vì Sao Dùng `min(H)` Và `max(L)`?

Nếu chỉ yêu cầu trung bình nhóm cao lớn hơn trung bình nhóm thấp, một vài snippet rất cao có thể che mất nhiều snippet chưa tách rõ. Công thức:

```text
min(H) >= max(L) + margin
```

mạnh hơn vì nó yêu cầu toàn bộ nhóm high-score phải nằm cao hơn toàn bộ nhóm low-score một khoảng margin. Nói cách khác, nó tạo một khoảng cách rõ giữa hai vùng score trong cùng video abnormal.

### 5.2. Vì Sao Chọn Top-K Và Low-K Bằng Score `detach()`?

Trong code multi-prompt, chỉ số của nhóm `H_i` và `L_i` được chọn bằng score đã detach:

```text
H_i = top-k(s(t).detach())
L_i = bottom-k(s(t).detach())
```

Nhưng giá trị loss vẫn tính trên score gốc:

```text
score(H_i), score(L_i)
```

Ý nghĩa:

- việc chọn snippet high/low không tạo gradient qua thao tác chọn top-k;
- sau khi đã chọn được chỉ số snippet, loss vẫn truyền gradient qua score thật;
- cách này ổn định hơn vì tránh để gradient phụ thuộc trực tiếp vào thao tác ranking rời rạc.

## 6. Regularizer 3: Ranking Margin Loss

Regularizer này so sánh score giữa video Normal và video Abnormal trong cùng batch.

Với mỗi video, lấy trung bình top-k anomaly score:

```text
a_i = mean(top-k anomaly scores của video i)
```

Tính trung bình cho nhóm Normal và Abnormal:

```text
a_normal = mean(a_i với video i thuộc Normal)
a_abnormal = mean(a_i với video i thuộc Abnormal)
```

Yêu cầu:

```text
a_abnormal >= a_normal + margin
```

Viết thành hinge loss:

```text
L_ranking =
  max(0, margin - a_abnormal + a_normal)
```

Loss tổng nếu bật regularizer này:

```text
L_total =
  L_original
+ lambda_ranking_margin * L_ranking
```

Ý nghĩa: thay vì chỉ xử lý từng video riêng lẻ, loss này ép score top-k của video bất thường phải cao hơn score top-k của video bình thường ở mức batch.

## 7. So Sánh Loss Gốc Và Loss Sau Khi Thêm Regularizer

Loss gốc của VadCLIP:

```text
L_original =
  L_cls
+ L_mil
+ L_text_reg
```

Loss mở rộng tổng quát nếu bật cả ba regularizer:

```text
L_total =
  L_cls
+ L_mil
+ L_text_reg
+ lambda_normal_topk * L_normal_topk
+ lambda_abnormal_bg(epoch) * L_abnormal_bg
+ lambda_ranking_margin * L_ranking
```

Trong thí nghiệm đang muốn áp dụng cho VadCLIP gốc, chỉ cần bật abnormal background separation:

```text
L_total =
  L_cls
+ L_mil
+ L_text_reg
+ lambda_abnormal_bg(epoch) * L_abnormal_bg
```

Các thành phần còn lại có thể đặt bằng 0:

```text
lambda_normal_topk = 0
lambda_ranking_margin = 0
```

## 8. Bảng So Sánh Ba Regularizer

| Regularizer | Video được dùng | Chọn snippet | Công thức chính | Mục tiêu |
|---|---|---|---|---|
| Normal Top-K Suppression | Normal | top-k score cao nhất | `mean(top-k score)` | giảm false positive trên video Normal |
| Abnormal Background Separation | Abnormal | top-k cao nhất và top-k thấp nhất | `max(0, max(L) - min(H) + margin)` | tách abnormal segment khỏi background trong video abnormal |
| Ranking Margin | Normal + Abnormal | top-k score cao nhất mỗi video | `max(0, margin - abnormal_mean + normal_mean)` | làm abnormal video có score cao hơn normal video |

## 9. Tóm Tắt Phần Cần Triển Khai

Phần cần thêm vào VadCLIP gốc là:

```text
Abnormal Background Separation
```

Không thay đổi:

- kiến trúc `CLIPVAD`;
- cách encode text prompt gốc của VadCLIP;
- `L_cls`, `L_mil`, `L_text_reg`;
- inference và evaluation.

Chỉ thêm vào training loss:

```text
lambda_abnormal_bg(epoch) * L_abnormal_bg
```

Với:

```text
L_abnormal_bg =
mean over abnormal videos and score branches [
  max(0, max(score(low-k)) - min(score(high-k)) + margin)
]
```

Trực giác ngắn gọn:

```text
Trong video bất thường, model phải học cách làm nổi bật đoạn bất thường
và giữ các đoạn nền trong cùng video ở mức score thấp hơn.
```
