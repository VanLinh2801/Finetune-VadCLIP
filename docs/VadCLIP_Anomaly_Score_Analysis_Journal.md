# Nhật Ký Phân Tích Lỗi Anomaly Score Của VadCLIP

## 1. Mục Tiêu Chung

Sau nhiều thí nghiệm với class prompt, weighted prompt và abnormal top-k suppression, một hiện tượng lặp lại được ghi nhận: một số cấu hình làm tăng AP hoặc mAP nhưng AUC của nhánh classifier vẫn giảm. Vì AUC phụ thuộc vào thứ tự score của toàn bộ anomaly frame và normal frame, chỉ theo dõi loss hoặc metric cuối cùng không đủ để xác định nguyên nhân.

Chuỗi phân tích này được thực hiện để trả lời lần lượt bốn câu hỏi:

1. Score của anomaly và normal đang phân bố như thế nào, và lỗi thứ hạng xuất hiện ở đâu?
2. Các anomaly snippet và hard-negative snippet có phân bố liên tục theo thời gian hay chỉ là các điểm nhiễu rời rạc?
3. Lỗi xuất phát từ visual feature, classifier hay nhánh text alignment?
4. Normal visual memory và video context có thể giúp nhận diện hard negative hay không?

Bốn checkpoint được dùng xuyên suốt phần lớn các phép phân tích:

| Tên rút gọn | Mô tả |
|---|---|
| Baseline tự train | VadCLIP được train lại từ đầu, không thêm regularization |
| Baseline + abnormal top-k | VadCLIP gốc được train từ đầu với abnormal top-k suppression |
| Weighted prompt | VadCLIP dùng weighted multi-prompt, không regularization |
| Weighted prompt + abnormal top-k | Weighted multi-prompt kết hợp abnormal top-k suppression |

Baseline trong tài liệu là **baseline tự train**, không phải checkpoint AUC 88.02 do tác giả VadCLIP công bố.

## 2. Quy Ước Phân Tích

### 2.1. Nhãn snippet

Mỗi score đầu ra được xem là đại diện cho 16 frame. Một snippet được gán nhãn anomaly nếu có ít nhất một frame anomaly:

```text
snippet_is_anomaly = 1 nếu anomaly_frame_count > 0
```

Tỷ lệ anomaly trong snippet được tính bằng:

```text
anomaly_fraction = anomaly_frame_count / 16
```

### 2.2. Bốn nhóm snippet

Trong từng video, top 15% snippet có `logits1` cao nhất được xem là nhóm model ưu tiên. Khi đối chiếu với ground truth, các snippet được chia thành:

| Nhóm | Ground truth | Thuộc top 15% `logits1` |
|---|---|---|
| `true_high` | Anomaly | Có |
| `false_high` | Normal/background | Có |
| `missed_anomaly` | Anomaly | Không |
| `true_low` | Normal/background | Không |

`missed_anomaly` không nhất thiết có score dưới 0.5; nó chỉ không lọt vào top 15% của video.

### 2.3. Hai nhánh anomaly score

Nhánh classifier sử dụng:

```text
classifier_score = sigmoid(logits1)
```

Nhánh alignment sử dụng xác suất class Normal từ `logits2`:

```text
alignment_score = 1 - softmax(logits2)[Normal]
```

## 3. Giai Đoạn 1: Phân Tích Score Separation

### 3.1. Tại sao thực hiện

Weighted prompt và abnormal top-k có thể làm tăng khoảng cách score trong từng abnormal video nhưng AUC toàn cục vẫn giảm. Vì vậy cần tách hai khái niệm:

- **Intra-video separation**: anomaly score cao và thấp trong cùng một abnormal video cách nhau bao xa.
- **Global separation**: anomaly snippet của toàn test set có được xếp cao hơn tất cả normal/background snippet hay không.

Phân tích cũng kiểm tra các vùng nguy hiểm của phân phối: anomaly có score rất thấp và normal có score rất cao.

### 3.2. Các thống kê

Global gap:

```text
global_gap = mean(anomaly_score) - mean(normal_score)
```

`positive_p10` là score mà 10% anomaly nằm thấp hơn nó. Giá trị thấp cho thấy tồn tại một nhóm anomaly bị model bỏ sót nghiêm trọng.

`negative_p95` và `negative_p99` là các ngưỡng mà lần lượt 5% và 1% normal/background có score cao hơn. Chúng đo phần đuôi hard negative.

Top-k overlap là tỷ lệ snippet thuộc top-k score đồng thời là anomaly thật:

```text
topk_overlap = true_high / (true_high + false_high)
```

Intra-video gap là khoảng cách giữa nhóm score cao và nhóm score thấp trong từng abnormal video.

### 3.3. Kết quả tổng quan

| Model | Nhánh | AUC | AP | Positive mean | Negative mean | Global gap |
|---|---|---:|---:|---:|---:|---:|
| Baseline tự train | Classifier | 87.44 | 32.07 | 0.843 | 0.267 | 0.576 |
| Baseline tự train | Alignment | 85.64 | 26.15 | 0.897 | 0.342 | 0.555 |
| Baseline + abnormal top-k | Classifier | 87.24 | 31.98 | 0.819 | 0.245 | 0.575 |
| Baseline + abnormal top-k | Alignment | 86.15 | 27.67 | 0.897 | 0.321 | 0.576 |
| Weighted prompt | Classifier | 87.25 | 33.08 | 0.822 | 0.247 | 0.575 |
| Weighted prompt | Alignment | 85.28 | 25.73 | 0.862 | 0.317 | 0.546 |
| Weighted prompt + abnormal top-k | Classifier | 87.38 | 33.15 | 0.828 | 0.247 | 0.581 |
| Weighted prompt + abnormal top-k | Alignment | 85.24 | 25.25 | 0.869 | 0.319 | 0.549 |

### 3.4. Phần đuôi phân phối của nhánh classifier

| Model | Positive p10 | Positive median | Negative p95 | Negative p99 |
|---|---:|---:|---:|---:|
| Baseline tự train | 0.286 | 0.976 | 0.995 | 0.9997 |
| Baseline + abnormal top-k | 0.202 | 0.984 | 0.997 | 0.9998 |
| Weighted prompt | 0.154 | 0.990 | 0.998 | 0.99996 |
| Weighted prompt + abnormal top-k | 0.208 | 0.991 | 0.999 | 0.99994 |

Weighted prompt làm median anomaly rất cao nhưng `positive_p10` giảm mạnh từ 0.286 xuống 0.154. Đồng thời, normal p95 và p99 vẫn gần 1. Điều này cho thấy lỗi tập trung ở hai phần đuôi:

- Một nhóm anomaly vốn khó tiếp tục bị đẩy xuống rất thấp.
- Một nhóm normal/background vẫn nhận score gần tuyệt đối.

Do đó, mean global gap vẫn tương đối lớn nhưng AUC không tăng. AUC bị chi phối bởi số lượng cặp anomaly-normal bị đảo thứ tự, không chỉ bởi chênh lệch trung bình.

### 3.5. Top-k overlap và intra-video gap

| Model | Classifier top-k overlap | Classifier intra-video gap |
|---|---:|---:|
| Baseline tự train | 35.75% | 0.408 |
| Baseline + abnormal top-k | 33.59% | 0.446 |
| Weighted prompt | 34.48% | 0.436 |
| Weighted prompt + abnormal top-k | 33.49% | 0.448 |

Abnormal top-k làm intra-video gap tăng nhưng top-k overlap giảm. Nói cách khác, model trở nên tự tin hơn vào tập snippet được chọn, nhưng tập được chọn không chính xác hơn.

Khoảng hai phần ba snippet trong top-k vẫn là background. Vì vậy, áp dụng loss trực tiếp lên top-k có nguy cơ củng cố pseudo-label sai.

### 3.6. Correlation giữa hai nhánh

| Model | Toàn bộ frame | Anomaly frame | Normal frame |
|---|---:|---:|---:|
| Baseline tự train | 0.946 | 0.899 | 0.941 |
| Baseline + abnormal top-k | 0.921 | 0.859 | 0.914 |
| Weighted prompt | 0.928 | 0.893 | 0.921 |
| Weighted prompt + abnormal top-k | 0.924 | 0.862 | 0.918 |

Việc thêm prompt hoặc abnormal top-k làm correlation giảm. Điều này không tự động là xấu, nhưng nó cho thấy hai nhánh không còn đồng thuận mạnh như baseline, đặc biệt trên anomaly frame. Nếu hai nhánh cùng sai ở các snippet khác nhau, việc dùng một nhánh để chọn pseudo-label cho nhánh còn lại sẽ thiếu ổn định.

### 3.7. Khác biệt theo class

| Class | Baseline AUC | Weighted prompt AUC | Weighted + top-k AUC |
|---|---:|---:|---:|
| Arrest | 65.74 | 69.32 | 68.96 |
| Arson | 76.89 | 75.40 | 77.00 |
| Assault | 86.80 | 90.62 | 90.26 |
| Explosion | 48.10 | 50.92 | 53.14 |
| Fighting | 67.25 | 66.07 | 66.28 |
| Shoplifting | 66.37 | 63.82 | 63.04 |
| Vandalism | 75.27 | 77.64 | 77.46 |

Prompt không gây ảnh hưởng đồng nhất: `Assault`, `Arrest` và `Vandalism` được cải thiện, trong khi `Shoplifting` và `Fighting` giảm. `Explosion` tăng so với baseline nhưng vẫn ở mức rất thấp và tiếp tục là class khó nhất.

### 3.8. Insight của giai đoạn 1

1. Weighted prompt không làm giảm score đồng đều; nó làm phân phối phân cực hơn.
2. Global mean gap không đủ giải thích AUC vì lỗi nằm ở phần đuôi của phân phối.
3. Abnormal top-k tăng độ tự tin trong video nhưng không tăng độ chính xác của top-k.
4. Cần kiểm tra cấu trúc thời gian của true anomaly và false-high trước khi thiết kế temporal suppression.

## 4. Giai Đoạn 2: Phân Tích Snippet Topology

### 4.1. Tại sao thực hiện

Sau score separation, một giả thuyết được đặt ra: anomaly thường tạo thành chuỗi liên tục, còn false-high background có thể chỉ là các điểm rời rạc. Nếu đúng, có thể dùng temporal continuity để lọc hard negative.

Phân tích topology được thực hiện để kiểm tra trực tiếp:

- Anomaly snippet có liên tục hay không.
- Mỗi anomaly snippet chứa bao nhiêu anomaly frame.
- High-score normal snippet có liên tục hay rời rạc.
- Ranh giới block 256 snippet có làm đứt chuỗi sự kiện hay không.

### 4.2. Ground-truth anomaly topology

Trên 140 abnormal test videos:

| Thống kê | Kết quả |
|---|---:|
| Tổng anomaly snippet | 5.400 |
| Anomaly snippet trung bình/video | 38.57 |
| Số anomaly run trung bình/video | 1.11 |
| Longest anomaly run trung bình | 36.30 snippet |
| Mean run length trung bình | 35.25 snippet |
| Tỷ lệ anomaly snippet có hàng xóm anomaly | 100% |

Mỗi anomaly video thường chỉ có khoảng một chuỗi anomaly lớn. Ground-truth anomaly vì vậy có tính liên tục rất mạnh.

### 4.3. Độ tinh khiết của anomaly snippet

| Thống kê | Kết quả |
|---|---:|
| Trung bình tỷ lệ anomaly frame/snippet | 97.41% |
| Snippet có đủ 16/16 anomaly frame | 95.02% |
| Snippet trộn anomaly và normal frame | 4.98% |

Điều này bác bỏ khả năng phần lớn lỗi đến từ việc snippet chứa quá nhiều frame normal. Hầu hết anomaly snippet là anomaly gần như hoàn toàn.

### 4.4. Top-15% topology và số lượng bốn nhóm

| Model | True high | False high | Missed anomaly | True low |
|---|---:|---:|---:|---:|
| Baseline tự train | 1.613 | 8.924 | 3.787 | 55.044 |
| Baseline + abnormal top-k | 1.606 | 8.931 | 3.794 | 55.037 |
| Weighted prompt | 1.597 | 8.940 | 3.803 | 55.028 |
| Weighted prompt + abnormal top-k | 1.626 | 8.911 | 3.774 | 55.057 |

Với baseline, top-15% chỉ thu được khoảng 36.8% anomaly snippet thật. Trong 10.537 snippet được chọn, số false-high lớn hơn true-high khoảng 5,5 lần.

### 4.5. High-score normal cũng liên tục

Các high-score snippet không chủ yếu là singleton. Ở top-15%, hơn 99% selected snippet có ít nhất một selected neighbor. Trong các abnormal video, false-high background cũng tạo thành các run dài; trong Normal video, các vùng score cao tương tự cũng kéo dài liên tục.

Một số thống kê tiêu biểu của nhánh classifier:

- False-high background trong abnormal video có run trung bình khoảng 10,6 snippet, median khoảng 7 và có run dài tới 76 snippet.
- High-score vùng trong Normal video có run trung bình khoảng 11,3 snippet, median khoảng 7 và có run dài tới 168 snippet.
- Khoảng 99,3% high-score normal snippet có selected neighbor.

### 4.6. Insight của giai đoạn 2

1. Ground-truth anomaly liên tục và rất tinh khiết ở cấp snippet.
2. Hard negative cũng liên tục, không phải nhiễu rời rạc.
3. Temporal continuity đơn thuần không thể phân biệt anomaly với hard negative.
4. Cần chuyển sang phân tích visual feature để xem hai nhóm có khác nhau trong không gian biểu diễn hay không.

## 5. Giai Đoạn 3: Phân Tích Visual Feature

### 5.1. Tại sao thực hiện

Topology cho thấy true anomaly và hard negative đều có thể tạo thành chuỗi dài. Vì vậy cần kiểm tra liệu visual representation có còn chứa thông tin phân biệt hai nhóm hay model đã biến chúng thành các feature gần như giống nhau.

Ba tầng feature được phân tích:

| Tầng | Ý nghĩa |
|---|---|
| `raw_clip` | Feature CLIP đầu vào |
| `temporal` | Feature sau temporal transformer |
| `final_visual` | Feature sau toàn bộ temporal và graph processing |

### 5.2. Grouped-by-video linear probe

Linear probe được train bằng snippet ground truth và chia fold theo video. Đây là phép đo chẩn đoán: nếu probe AUC cao, feature vẫn chứa thông tin phân biệt dù classifier weakly supervised chưa chắc tận dụng được.

| Model | Raw CLIP | Temporal | Final visual |
|---|---:|---:|---:|
| Baseline tự train | 80.00 | 81.11 | 83.00 |
| Baseline + abnormal top-k | 81.53 | 82.00 | 83.22 |
| Weighted prompt | 82.12 | 84.12 | **86.37** |
| Weighted prompt + abnormal top-k | 82.95 | 83.93 | 83.64 |

Weighted prompt không regularization tạo final visual feature có khả năng phân tách tốt nhất. Khi thêm abnormal top-k, probe AUC giảm từ 86.37 xuống 83.64. Loss regularization đã làm yếu một phần thông tin tốt trong representation.

### 5.3. kNN purity

| Model | Raw CLIP | Temporal | Final visual |
|---|---:|---:|---:|
| Baseline tự train | 0.889 | 0.903 | 0.912 |
| Baseline + abnormal top-k | 0.884 | 0.902 | 0.914 |
| Weighted prompt | 0.897 | 0.909 | **0.920** |
| Weighted prompt + abnormal top-k | 0.890 | 0.906 | 0.916 |

Các tầng temporal và graph giúp anomaly/normal tạo cụm cục bộ rõ hơn. Weighted prompt tiếp tục có purity cao nhất, củng cố nhận định rằng feature chưa bị phá hỏng hoàn toàn.

### 5.4. Cosine similarity trong cùng abnormal video

Ở tầng `final_visual`:

| Model | False-high ↔ true-high | Missed anomaly ↔ true-high | False-high ↔ true-low | Missed anomaly ↔ false-high |
|---|---:|---:|---:|---:|
| Baseline tự train | 0.961 | 0.976 | 0.948 | 0.943 |
| Baseline + abnormal top-k | 0.952 | 0.972 | 0.939 | 0.927 |
| Weighted prompt | 0.922 | 0.954 | 0.892 | 0.877 |
| Weighted prompt + abnormal top-k | 0.933 | 0.963 | 0.908 | 0.898 |

False-high background thật sự gần true anomaly trong không gian feature, đặc biệt khi cùng nằm trong một abnormal video. Tuy nhiên, weighted prompt đã làm các nhóm tách nhau hơn baseline; vấn đề không phải feature hoàn toàn không phân biệt được.

### 5.5. Class attribution của false-high

Trong weighted prompt + abnormal top-k, false-high background thuộc abnormal video thường được nhánh alignment gán cho các class/prompt rộng:

| Class được gán | Tỷ lệ false-high |
|---|---:|
| RoadAccidents | 24.1% |
| Robbery | 20.4% |
| Burglary | 14.2% |
| Shoplifting | 7.8% |
| Stealing | 7.7% |
| Fighting | 7.3% |
| Arson | 7.2% |

Các prompt đóng góp nhiều nhất gồm `robbery robber intimidation theft` (18.8%), `roadAccidents vehicle collision roadway` (17.6%) và `burglary burglar carrying property` (11.3%). Đây đều là prompt có semantic tương đối rộng, dễ phản ứng với người, phương tiện, chuyển động hoặc vật thể trong background.

### 5.6. Insight của giai đoạn 3

1. Weighted prompt vẫn giữ, thậm chí tăng, tín hiệu phân biệt trong visual feature.
2. Abnormal top-k có thể cải thiện score cục bộ nhưng làm representation yếu đi.
3. False-high background có visual feature gần anomaly thật, nên không thể lọc chỉ bằng temporal continuity hoặc score.
4. Nhánh alignment còn bị ảnh hưởng bởi các prompt rộng và video context.
5. Cần tách riêng lỗi do feature, classifier, normal similarity và context.

## 6. Giai Đoạn 4: Xác Định Nguồn Lỗi

### 6.1. Phân tích classifier so với feature

#### Tại sao thực hiện

Visual diagnostics cho thấy feature probe tốt nhưng chưa so sánh trực tiếp feature ngay trước classifier với chính classifier trên cùng sample và fold. Phép đo này nhằm xác định lỗi nằm ở representation hay decision boundary học bằng MIL.

`final_visual` ký hiệu là (z). Đầu vào classifier là:

```text
h = z + MLP2(z)
```

Một logistic regression được train trên feature bằng frame-derived snippet label. AUC của classifier VadCLIP được tính trên đúng các fold đó.

#### Kết quả trên toàn test

| Model | Final visual probe | Classifier-input probe | VadCLIP classifier |
|---|---:|---:|---:|
| Baseline tự train | 83.00 | 82.68 | 84.00 |
| Baseline + abnormal top-k | 81.70 | 81.75 | 82.65 |
| Weighted prompt | **86.83** | **86.82** | 85.05 |
| Weighted prompt + abnormal top-k | 84.23 | 84.40 | 84.07 |

#### Kết quả chỉ trong abnormal videos

| Model | Final visual probe | Classifier-input probe | VadCLIP classifier | Probe - classifier |
|---|---:|---:|---:|---:|
| Baseline tự train | 61.59 | 61.11 | 63.50 | -2.40 |
| Baseline + abnormal top-k | 59.19 | 59.57 | 62.09 | -2.51 |
| Weighted prompt | 67.64 | **68.21** | 61.45 | **+6.76** |
| Weighted prompt + abnormal top-k | 66.18 | 66.40 | 62.23 | +4.17 |

#### Insight

Ở baseline, classifier đã khai thác feature tương đối tốt. Ở weighted prompt, feature chứa tín hiệu tuyến tính tốt hơn nhiều nhưng classifier MIL không học được decision boundary tương ứng. Đây là bằng chứng mạnh nhất cho thấy bottleneck chính của weighted prompt nằm ở **cách classifier được huấn luyện từ video-level pseudo-label**, không chỉ ở visual backbone.

Linear probe sử dụng snippet ground truth nên chỉ là upper bound chẩn đoán, không phải phương pháp training hợp lệ trong weakly supervised setting.

### 6.2. Phân tích normal visual memory bank

#### Tại sao thực hiện

Một hướng tiềm năng là xây dựng memory bank từ Normal training videos rồi hạ score những snippet giống normal. Trước khi đưa vào loss, cần kiểm tra liệu memory bank có thực sự tách false-high background khỏi missed anomaly hay không.

Memory bank được tạo từ 500 Normal feature rows thuộc 373 video, tối đa 50.000 feature mỗi tầng và 32 KMeans prototypes. Normal similarity của feature (x) là:

```text
normal_similarity(x) = max cosine(x, normal_prototype_j)
```

#### Phân biệt toàn bộ normal và anomaly

| Model | Raw CLIP | Final visual | Classifier input |
|---|---:|---:|---:|
| Baseline tự train | 56.10 | 83.73 | 83.62 |
| Baseline + abnormal top-k | 56.10 | 84.90 | 84.40 |
| Weighted prompt | 56.10 | **85.35** | **85.15** |
| Weighted prompt + abnormal top-k | 56.10 | 84.46 | 84.43 |

Sau temporal/graph processing, model tạo được không gian có khái niệm normal rõ hơn raw CLIP.

#### Phân biệt false-high background và missed anomaly

Positive class trong phép đo này là false-high background; AUC trên 50% mới có nghĩa false-high thường giống normal hơn missed anomaly.

| Model | Raw CLIP | Final visual | Classifier input |
|---|---:|---:|---:|
| Baseline tự train | 45.66 | 47.31 | 46.97 |
| Baseline + abnormal top-k | 45.55 | 44.33 | 45.47 |
| Weighted prompt | 45.92 | 41.52 | 40.10 |
| Weighted prompt + abnormal top-k | 45.02 | 40.64 | 40.49 |

Memory bank thất bại ở đúng bài toán khó. Missed anomaly thường còn gần normal prototypes hơn false-high background. Dùng normal similarity như suppression rule chung có nguy cơ hạ anomaly bị bỏ sót trước hard negative.

Kết quả phụ thuộc class: memory bank classifier-input có tín hiệu tốt hơn với `Stealing` (78.20%) và `Vandalism` (67.39%), nhưng đi ngược mạnh ở `Explosion` (16.53%), `Assault` (7.18%), `Fighting` (28.31%) và `Shooting` (27.24%). Vì vậy không nên áp dụng một normal-memory loss giống nhau cho mọi class.

### 6.3. Phân tích video-context leakage

#### Tại sao thực hiện

VadCLIP tạo context từ toàn video bằng cách dùng `logits1` để tổng hợp visual feature, sau đó cộng context này vào text/class feature trước khi tính `logits2`. Nếu top-k ban đầu chứa hard negative, context có thể lan tín hiệu anomaly sang background khác trong cùng video.

Context được điều chỉnh bằng hệ số gamma:

```text
text_feature_gamma = text_feature + gamma * video_context
```

- `gamma = 0`: loại bỏ context.
- `gamma = 1`: giữ nguyên model.

Sanity check xác nhận `gamma=1` tái tạo logits gốc với sai số tuyệt đối bằng 0 ở cả bốn model.

#### Kết quả gamma ablation

| Model | AUC2 gamma=1 | AUC2 tốt nhất | AP2 gamma=1 | AP2 gamma=0 |
|---|---:|---:|---:|---:|
| Baseline tự train | 85.64 | 85.76 tại 0.25 | 26.15 | 26.38 |
| Baseline + abnormal top-k | 85.62 | **86.08 tại 0** | 26.20 | **26.86** |
| Weighted prompt | 85.28 | 85.28 tại 0.75 | 25.73 | **26.39** |
| Weighted prompt + abnormal top-k | 85.29 | **85.46 tại 0** | 25.50 | **26.51** |

Giảm context làm AP2 tăng ở cả bốn model, cho thấy context tạo thêm false positive hoặc làm độ tự tin sai tăng. Tuy nhiên AUC2 chỉ cải thiện nhẹ và không nhất quán, nên context leakage không phải nguyên nhân duy nhất.

#### Ảnh hưởng theo class

- `Arrest`: context làm false-high nghiêm trọng hơn. Với weighted prompt + top-k, khoảng cách `missed anomaly - false high` cải thiện từ -0.345 ở gamma 1 lên -0.179 ở gamma 0.
- `Vandalism`: bỏ context giúp cải thiện nhẹ.
- `Explosion`: bỏ context làm khoảng cách xấu hơn, từ -0.048 xuống -0.080.
- `Arson`: context có ích; khoảng cách đang +0.048 ở gamma 1 nhưng thành -0.035 ở gamma 0.

Context còn giúp hạ background trong Normal videos rất mạnh. Với weighted prompt + abnormal top-k, false-high alignment trong Normal video tăng từ 0.124 ở gamma 1 lên 0.304 khi bỏ context. Vì vậy không nên xóa context toàn bộ; nếu điều chỉnh, cần context gating theo snippet hoặc class.

## 7. Chuỗi Insight Tổng Hợp

1. **Lỗi AUC là lỗi thứ hạng ở phần đuôi phân phối.** Một nhóm anomaly rất thấp và một nhóm normal rất cao tạo ra nhiều cặp đảo thứ tự, dù global mean gap vẫn lớn.
2. **Top-k hiện tại không đủ đáng tin.** Chỉ khoảng một phần ba top-k là anomaly thật; tăng intra-video gap không đồng nghĩa tăng AUC.
3. **Temporal continuity không giải quyết được hard negative.** True anomaly và false-high background đều tạo thành các chuỗi dài, liên tục.
4. **Snippet boundary không phải nguyên nhân chính.** Khoảng 95% anomaly snippet chứa đủ 16/16 anomaly frame.
5. **Weighted prompt không phá hỏng hoàn toàn visual feature.** Feature của weighted prompt có linear-probe AUC và kNN purity cao nhất.
6. **Classifier MIL không khai thác được feature tốt.** Trong abnormal videos, classifier-input probe cao hơn classifier tới 6.76 điểm AUC.
7. **Abnormal top-k có trade-off.** Nó có thể sửa decision boundary cục bộ nhưng đồng thời làm giảm chất lượng representation.
8. **Normal memory chỉ giải quyết bài toán dễ.** Nó tách normal/anomaly toàn cục tốt nhưng không tách false-high background khỏi missed anomaly.
9. **Hard negative trong abnormal video là background đặc biệt.** Nó khác background trong Normal video và thường giống anomaly thật về visual feature.
10. **Video context vừa giúp vừa gây hại.** Context leakage có thật nhưng phụ thuộc class; loại bỏ context toàn cục không phù hợp.
11. **Explosion là lỗi phức hợp.** Missed anomaly của Explosion giống normal memory, false-high giống anomaly, top-k overlap thấp và bỏ context không giúp. Đây không chỉ là lỗi prompt confusion.

## 8. Mô Hình Nguyên Nhân Hiện Tại

Các kết quả hỗ trợ vòng lặp lỗi weak supervision sau:

```text
Classifier ban đầu chọn top-k chưa chính xác
→ nhiều background được dùng như positive evidence
→ temporal/graph/context lan truyền tín hiệu từ vùng được chọn sai
→ MIL củng cố false-high và không cấp đủ gradient cho missed anomaly
→ top-k ở epoch tiếp theo tiếp tục sai
```

Weighted prompt làm thay đổi semantic alignment và decision boundary. Nó có thể giúp representation phân tách tốt hơn, nhưng nếu MIL vẫn chọn sai snippet thì classifier không chuyển được lợi ích của feature thành AUC.

## 9. Kết Luận

Vấn đề hiện tại không thể giải quyết chỉ bằng cách tăng suppression hoặc thay prompt. Bằng chứng mạnh nhất cho thấy visual feature vẫn còn tiềm năng, trong khi cơ chế lựa chọn positive snippet và huấn luyện classifier bằng MIL là bottleneck chính.

Hướng nghiên cứu hợp lý cần đáp ứng đồng thời ba yêu cầu:

- Không coi top-k score hiện tại là pseudo-label tuyệt đối.
- Không dùng temporal continuity hoặc normal similarity như quy tắc cứng.
- Khai thác được tín hiệu tốt đã tồn tại trong feature mà không cần frame label khi training.

## 10. Lưu Ý Về Tái Lập Kết Quả

Các thư mục phân tích được sinh ở những thời điểm khác nhau. Hai model có abnormal-top-k từng dùng lại cùng tên output checkpoint, vì vậy AUC trong `score separation` và `snippet topology` có chênh lệch nhỏ ở một số lần chạy. Ví dụ weighted prompt + abnormal top-k được ghi nhận khoảng 87.38 trong score-separation và 87.63 trong topology.

Các kết luận chính vẫn nhất quán vì dựa trên xu hướng phân phối, topology, feature và ablation. Tuy nhiên, các thí nghiệm tiếp theo nên lưu checkpoint bằng tên chứa đầy đủ cấu hình và mã lần chạy để mọi analyzer luôn dùng đúng một checkpoint bất biến.
