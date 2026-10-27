# Nhật Ký Thí Nghiệm: Fine-Tune VadCLIP Bằng Prompt

## 1. Bài toán đang giải quyết

VadCLIP được dùng để phát hiện sự kiện bất thường trong video giám sát. Đầu vào của model không phải ảnh gốc mà là chuỗi feature đã được trích xuất từ video.

Model tạo ra hai loại kết quả:

- **Classifier score**: mỗi thời điểm được chấm một điểm abnormal. Điểm này dùng để tính `classifier AUC` và `classifier AP`.
- **Video-text alignment score**: feature tại mỗi thời điểm được so sánh với biểu diễn text của 14 class. Điểm này dùng để tính `alignment AUC`, `alignment AP` và `mAP`.

Trong VadCLIP gốc, text của mỗi class chủ yếu là tên class, ví dụ `Explosion`, `Fighting` hoặc `Shoplifting`. Mục tiêu của các thí nghiệm là thay phần text đơn giản này bằng các prompt mô tả rõ hơn dấu hiệu thị giác của từng class.

Ví dụ:

```text
Tên class: Explosion
Prompt: blast debris spreading outward
```

Giả thuyết đặt ra:

- Một tên class có thể quá ngắn và không mô tả đủ những gì xuất hiện trong video.
- Nhiều prompt có thể giúp model nhận ra nhiều biểu hiện khác nhau của cùng một class.
- Prompt không phù hợp cũng có thể gây nhiễu, làm model nhầm normal với abnormal hoặc nhầm giữa các class.

## 2. Mốc dùng để so sánh

| Mốc | Ý nghĩa | Classifier AUC |
|---|---|---:|
| Baseline gốc | Checkpoint do tác giả VadCLIP cung cấp | 88.02 |
| Baseline tự train | Train lại VadCLIP chỉ bằng tên class, trên môi trường của dự án | 87.44 |

Ý nghĩa các metric:

- `Classifier AUC`: khả năng xếp abnormal frame cao hơn normal frame trên toàn bộ tập test.
- `Classifier AP`: độ chính xác của nhánh classifier khi abnormal frame ít hơn normal frame.
- `Alignment AUC/AP`: chất lượng anomaly score được suy ra từ nhánh so khớp video-text.
- `Ano-AUC`: AUC chỉ tính trên các video có sự kiện abnormal.
- `avg mAP`: khả năng xác định đúng đoạn thời gian xảy ra sự kiện và đúng class.

Classifier AUC là chỉ số chính. Checkpoint đại diện của mỗi lần chạy được chọn tại epoch có classifier AUC cao nhất.

---

## 3. Cách xây dựng prompt bank

Prompt bank là tập nhiều prompt ứng viên dành cho từng class. Quá trình được thực hiện theo các bước sau.

### Bước 1: Viết prompt ứng viên

- Viết nhiều cách mô tả cho cùng một class.
- Ưu tiên thông tin có thể nhìn thấy trong một frame: người, vật thể và hành động.
- Hạn chế mô tả ý định, nguyên nhân hoặc một chuỗi sự kiện quá dài.
- Hạn chế từ chung chung vì chúng có thể xuất hiện ở nhiều class.

Trong giai đoạn chuẩn bị dữ liệu text, nhiều prompt bank dài và ngắn khác nhau đã được xây dựng. Đây là bước tạo prompt chung; các phần thí nghiệm phía sau chỉ mô tả cách đưa những prompt bank này vào model.

Các dạng prompt đã được thử:

- Class description dài, mô tả nhiều khía cạnh của class.
- Class description ngắn, chỉ giữ hành động và đối tượng chính.
- Caption thị giác có cấu trúc gần với câu mô tả ảnh của CLIP.
- Cụm noun-verb ngắn.
- Keyword prompt chỉ giữ danh từ, động từ và tính từ thị giác quan trọng.
- Mỗi loại được thử với một prompt hoặc nhiều prompt cho mỗi class.

Nhóm thí nghiệm chính dùng prompt dạng noun-verb ngắn. Mỗi class giữ lại 5 prompt.

### Bước 2: Chuyển prompt thành vector

- Tất cả prompt và tên class được đưa qua đúng CLIP Text Encoder dùng trong VadCLIP.
- Ở bước phân tích này, chỉ dùng CLIP encoder gốc, chưa thêm learnable prompt của VadCLIP.
- Mỗi câu text được chuyển thành một vector embedding.
- Các vector được L2-normalize trước khi tính cosine similarity.

Cosine similarity càng lớn thì hai câu càng gần nhau về mặt semantic trong không gian CLIP.

### Bước 3: Chấm điểm từng prompt

Mỗi prompt được đánh giá theo bốn câu hỏi chính.

1. **Prompt có gần tên class đúng không?**

```text
anchor = similarity(prompt, tên class đúng)
```

2. **Prompt có gần class đúng hơn các class khác không?**

```text
margin = similarity với class đúng
       - similarity cao nhất với một class sai
```

Margin cao nghĩa là prompt có khả năng phân biệt class tốt hơn.

3. **Prompt có đại diện cho nội dung chung của class không?**

```text
centroid similarity = similarity(prompt, trung tâm nhóm prompt cùng class)
```

4. **Prompt có quá giống prompt của class dễ nhầm không?**

Ví dụ:

- `Explosion` được kiểm tra với prompt của `Arson` và `RoadAccidents`.
- `Fighting` được kiểm tra với `Assault`, `Abuse` và `Arrest`.
- `Shoplifting` được kiểm tra với `Stealing`, `Robbery` và `Burglary`.

Similarity với nhóm dễ nhầm càng cao thì prompt càng bị trừ điểm.

### Bước 4: Tính điểm tổng hợp

Các thành phần được chuẩn hóa trong phạm vi từng class rồi kết hợp như sau:

```text
Điểm prompt = 0.30 * độ gần tên class
             + 0.35 * khả năng tách class đúng và class sai
             + 0.05 * mức đại diện cho class
             - 0.30 * độ giống prompt của class dễ nhầm
```

Cách hiểu đơn giản:

- Prompt cần đúng nghĩa với class.
- Prompt cần khác rõ những class dễ nhầm.
- Prompt không nên quá riêng biệt hoặc quá xa nội dung chung.

Đây là công thức lọc prompt trong không gian text. Công thức không trực tiếp tối ưu AUC, vì vậy prompt được chọn vẫn phải kiểm chứng bằng training.

### Bước 5: Chọn 5 prompt vừa tốt vừa khác nhau

- Prompt có điểm cao nhất được chọn đầu tiên.
- Các prompt tiếp theo không chỉ cần điểm cao mà còn phải khác những prompt đã chọn.
- Phương pháp MMR được dùng để cân bằng hai yếu tố:

```text
Điểm chọn = 0.70 * chất lượng prompt
           - 0.30 * độ giống prompt đã chọn
```

- Hai prompt có cosine similarity từ `0.97` trở lên được xem là gần trùng nhau.
- Bước này giúp 5 prompt mô tả nhiều dấu hiệu khác nhau, thay vì lặp lại cùng một ý.

### Bước 6: Gán trọng số cho 5 prompt

- Prompt tốt hơn được gán trọng số lớn hơn.
- Trọng số được tính bằng softmax từ điểm prompt.
- Tổng trọng số của 5 prompt trong một class luôn bằng 1.

```text
weight_i = exp(score_i / 0.20) / tổng exp(score / 0.20)
```

Kết quả cuối cùng của quá trình này là một prompt bank gồm 14 class, mỗi class có 5 prompt và 5 trọng số tương ứng.

---

## 4. Prompt được đưa vào VadCLIP như thế nào?

Trong cấu hình chính, tên class vẫn được giữ lại. Prompt thực tế có dạng:

```text
class name + visual prompt
```

Ví dụ:

```text
Explosion blast debris spreading outward
```

Sau đó câu này được đặt giữa các token learnable của VadCLIP:

```text
learnable prefix + class/prompt text + learnable postfix
```

Điểm cần phân biệt:

- Khi **chọn prompt**, raw CLIP encoder được dùng để tránh kết quả bị ảnh hưởng bởi một checkpoint đã train.
- Khi **train VadCLIP**, prompt được encode cùng learnable prefix và postfix như thiết kế của model gốc.
- CLIP encoder được đóng băng. Các module xử lý video và learnable prompt context được cập nhật trong quá trình train.

---

## 5. Các loss chung

### 5.1. Loss của VadCLIP gốc

```text
L_VadCLIP = L_classifier + L_class_MIL + L_text_regularization
```

- `L_classifier`:
  - lấy những snippet có classifier score cao nhất trong video;
  - tính trung bình các score này;
  - học phân biệt video normal và video abnormal.
- `L_class_MIL`:
  - lấy các snippet có class-alignment score cao nhất;
  - tổng hợp thành dự đoán class cho toàn video;
  - học từ class label cấp video.
- `L_text_regularization`:
  - tách embedding của class `Normal` khỏi embedding của 13 abnormal class;
  - đây là thành phần có sẵn trong VadCLIP gốc.

### 5.2. Consistency loss khi dùng nhiều prompt

Mỗi prompt của cùng một class có thể tập trung vào một thời điểm khác nhau. Consistency loss được thêm để hạn chế các prompt mâu thuẫn quá mạnh.

Các bước:

1. Dùng classifier chọn 15% snippet có score cao nhất trong video.
2. Xem mỗi prompt của class thật tập trung vào snippet nào trong nhóm này.
3. Tính mức chú ý trung bình của cả nhóm prompt.
4. Dùng KL divergence để kéo mức chú ý của từng prompt gần mức trung bình.

```text
L_total = L_VadCLIP + 0.01 * L_consistency
```

Loss này không ép các prompt có embedding giống nhau. Nó chỉ khuyến khích chúng thống nhất hơn về vị trí thời gian chứa sự kiện.

### 5.3. Cấu hình train chung

| Thành phần | Giá trị |
|---|---|
| Dataset | UCF-Crime, 14 class |
| Test set | 290 video theo protocol VadCLIP |
| Prompt bank | 5 noun-verb prompt/class có trọng số |
| Learning rate | `2e-5` |
| Số epoch tối đa | `10` |
| Giảm learning rate | epoch 4 và 8 |
| Early stopping | 3 epoch không tăng classifier AUC |
| Consistency weight | `0.01` |
| Các regularization normal/abnormal bổ sung | tắt |
| CLIP backbone | đóng băng |

---

## 6. Các phương pháp fine-tune bằng prompt đã thử nghiệm

### 6.1. Phương pháp 1: Chỉ dùng prompt tốt nhất

Cách thực hiện:

1. Chọn prompt có điểm cao nhất của mỗi class.
2. Ghép prompt với tên class.
3. Gắn thêm learnable prompt của VadCLIP ở trước và sau phần text.
4. Train VadCLIP from scratch bằng hàm loss gốc.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo VadCLIP | Train from scratch |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Prompt text được chọn | Cố định; một prompt cho mỗi class |
| Cách tạo `logits2` | Một alignment score cho mỗi snippet và class, giống VadCLIP gốc |

Hàm loss:

```text
L_total = L_cls + L_mil + L_text_reg
```

Phương pháp này không thêm loss mới. Cả ba thành phần đều là loss của VadCLIP gốc.

| Classifier AUC | Classifier AP | Alignment AUC | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|
| 87.15 | 30.24 | 86.91 | 68.32 | 7.24 |

Nhận xét:

- Alignment AUC và mAP cao hơn baseline gốc.
- Classifier AUC và Ano-AUC thấp hơn baseline.
- Một prompt duy nhất không bao phủ đủ các hình thức xuất hiện khác nhau của một class.

### 6.2. Phương pháp 2a: Giữ riêng 5 prompt có trọng số

Mục tiêu của phương pháp này là giữ lại nhiều biểu hiện thị giác của một class. Ví dụ, cùng một class có thể được mô tả bằng hành động, đối tượng hoặc tương tác khác nhau; việc nén sớm năm prompt thành một vector trung bình có thể làm mất các tín hiệu riêng đó.

Cách thực hiện:

1. Mỗi class có 5 prompt đã được chọn và gán trọng số. Tổng trọng số của 5 prompt trong một class bằng 1.
2. Từng prompt được ghép với tên class, sau đó đặt giữa learnable prefix và learnable postfix của VadCLIP.
3. Cả 5 chuỗi text được đưa riêng qua CLIP Text Encoder. Do đó, mỗi class vẫn có 5 prompt embedding độc lập, không tạo centroid ở bước này.
4. Với mỗi visual snippet, model tính cosine similarity với từng prompt embedding. Một class tạo ra 5 alignment score riêng tại mỗi vị trí thời gian.
5. Với video có `T` snippet, toàn bộ prompt alignment có kích thước `[T, 14, 5]`: `14` class và `5` prompt cho mỗi class.
6. Năm score của cùng một class được gộp bằng weighted logsumexp. Kết quả cuối là `logits2 [T, 14]`, có cùng kích thước với nhánh alignment của VadCLIP gốc.
7. `logits2` sau khi gộp được đưa vào MIL loss theo class. Vì vậy, phần còn lại của quy trình train và inference vẫn tuân theo VadCLIP.

Với class `c`, snippet `t`, prompt thứ `k`, gọi `a(t,c,k)` là alignment score và `w(c,k)` là trọng số prompt:

```text
logits2(t,c) = log [ tổng_k w(c,k) * exp(a(t,c,k)) ]
```

Weighted logsumexp cho phép prompt có alignment cao đóng góp mạnh hơn, nhưng các prompt còn lại vẫn giữ ảnh hưởng theo trọng số. Đây không phải phép lấy trung bình prompt embedding và cũng không phải chỉ chọn prompt có score lớn nhất.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo VadCLIP | Train from scratch |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Prompt text viết sẵn | Cố định |
| Trọng số của 5 prompt | Cố định, không học trong quá trình train |
| Cách gộp prompt | Weighted logsumexp trên 5 alignment score |

Hàm loss:

```text
L_total = L_cls + L_mil + L_text_reg + 0.01 * L_consistency
```

- `L_cls`, `L_mil` và `L_text_reg` là ba loss của VadCLIP gốc.
- `L_consistency` là loss được thêm cho phương pháp multi-prompt. Loss này khuyến khích 5 prompt của class thật tập trung vào các vị trí thời gian tương đối nhất quán trong nhóm snippet có classifier score cao.
- Các normal/abnormal regularization bổ sung đều tắt trong thí nghiệm này.

| Classifier AUC | Classifier AP | Alignment AUC | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|
| 87.27 | 31.74 | 86.17 | 69.10 | 7.62 |

Nhận xét:

- Đây là phương pháp cân bằng nhất trong ba phương pháp prompt cơ bản.
- Kết quả tốt hơn việc chỉ giữ một prompt hoặc nén prompt thành một vector.

### 6.3. Phương pháp 2b: Nén 5 prompt thành một class prototype

Cách thực hiện:

1. Chọn 5 prompt có trọng số cho mỗi class.
2. Ghép từng prompt với tên class và gắn learnable prompt của VadCLIP.
3. Encode 5 prompt bằng CLIP Text Encoder.
4. Tính trung bình có trọng số để tạo một class prototype duy nhất.
5. Dùng class prototype thay cho 5 prompt khi so khớp với video.
6. Train VadCLIP from scratch; CLIP encoder được đóng băng, learnable prompt và các module xử lý video được huấn luyện.

```text
class prototype = Normalize(tổng weight_prompt * prompt_embedding)
```

Khác với Phương pháp 2a, việc gộp diễn ra trong không gian embedding trước khi so khớp với visual snippet. Vì vậy, mỗi class chỉ tạo một alignment map và `logits2` có trực tiếp kích thước `[T,14]`.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo VadCLIP | Train from scratch |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Prompt text và trọng số prompt | Cố định |
| Cách gộp prompt | Weighted mean-normalize trong không gian embedding |
| Loss bổ sung | Không có |

```text
L_total = L_cls + L_mil + L_text_reg
```

| Classifier AUC | Classifier AP | Alignment AUC | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|
| 86.12 | 30.12 | 86.01 | 66.42 | 6.38 |

Nhận xét:

- Kết quả thấp nhất trong ba phương pháp cơ bản.
- Khi gộp prompt trước khi so sánh với video, các dấu hiệu thị giác khác nhau bị trộn vào một vector trung bình.

### 6.4. Phương pháp 3: Class prototype có vector điều chỉnh

Cách thực hiện:

1. Tạo một class prototype như Phương pháp 2b từ 5 prompt có trọng số.
2. Thêm một vector residual `Delta(c)` có thể học cho mỗi class và khởi tạo bằng 0.
3. Trong mỗi lần forward, các prompt được encode lại cùng learnable prefix/postfix hiện tại, rồi gộp thành prototype `p(c)`.
4. Cộng `p(c)` với residual để tạo biểu diễn class dùng trong quá trình alignment.
5. Train các module xử lý video, learnable prefix/postfix và residual from scratch.

```text
text cuối(c) = Normalize(p(c) + Delta(c))
```

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo VadCLIP | Train from scratch |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Residual `Delta(c)` | Được cập nhật; một vector cho mỗi class |
| Prototype lưu trong buffer | Chỉ là bản chụp để thống kê, không dùng thay cho prototype được encode lại trong forward |
| Cách tạo `logits2` | Một alignment map/class từ prototype sau residual |

Hàm loss đầy đủ:

```text
L_total = L_cls + L_mil + L_text_reg + 0.001 * L_residual
L_residual = trung bình ||Delta(c)||²
```

Loss mới duy nhất là `L_residual`; phương pháp này không dùng `L_consistency` vì mỗi class chỉ còn một prototype sau khi gộp.

| Classifier AUC | Classifier AP | Alignment AUC | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|
| 86.76 | 31.91 | 85.25 | 68.64 | 7.18 |

Nhận xét:

- Tốt hơn prototype cố định.
- Vẫn thấp hơn phương pháp giữ riêng 5 prompt vì thông tin đã bị nén trước khi residual được học.

### 6.5. Phương pháp 2a+3: Mỗi prompt có một vector điều chỉnh

Phương pháp này giữ nguyên toàn bộ cơ chế của Phương pháp 2a, nhưng cho phép mỗi prompt có một hướng điều chỉnh nhỏ trong không gian embedding. Mục tiêu là để các prompt thích nghi với UCF-Crime mà không phải cập nhật CLIP Text Encoder.

Cách thực hiện:

1. Mỗi class vẫn sử dụng 5 prompt có trọng số; từng prompt được ghép với tên class và đặt giữa learnable prefix/postfix.
2. Ở mỗi lần forward, 70 prompt (`14 class x 5 prompt`) được encode lại bằng CLIP Text Encoder cùng learnable prompt hiện tại. Vì vậy, learnable prefix/postfix vẫn nhận gradient và được cập nhật.
3. Gọi embedding nền của prompt thứ `k` thuộc class `c` là `e(c,k)`. Model bổ sung một residual riêng `Delta(c,k)`, có cùng số chiều và được khởi tạo bằng 0.
4. Prompt embedding dùng để alignment được tính như sau:

```text
t(c,k) = Normalize(e(c,k) + Delta(c,k))
```

5. Mỗi visual snippet được so khớp riêng với cả 5 embedding đã điều chỉnh. Prompt alignment vẫn có kích thước `[T, 14, 5]`.
6. Năm alignment score của mỗi class tiếp tục được tổng hợp bằng đúng weighted logsumexp của Phương pháp 2a, tạo `logits2 [T, 14]`.
7. `logits2` được dùng để tính MIL loss theo class; residual không tạo thêm classifier hoặc nhánh dự đoán độc lập.

Residual loss được định nghĩa như sau:

```text
L_residual = trung bình ||Delta(c,k)||²
```

- `L_residual` hạn chế residual làm prompt thay đổi quá xa.
- `L_consistency` khuyến khích các prompt cùng class tập trung vào các vị trí thời gian nhất quán.

Điểm khác với Phương pháp 2a:

- Phương pháp 2a học learnable prefix/postfix chung, nhưng không có vector điều chỉnh riêng cho từng prompt.
- Phương pháp 2a+3 vừa học learnable prefix/postfix chung, vừa học 70 residual riêng cho `14 x 5` prompt.
- Residual tác động lên prompt embedding trước khi tính alignment; cách tạo `[T,14,5]` và weighted logsumexp phía sau không thay đổi.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo VadCLIP | Train from scratch |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Prompt text viết sẵn và trọng số prompt | Cố định |
| Residual `Delta(c,k)` | Được cập nhật; một vector cho mỗi prompt |
| Cách gộp prompt | Weighted logsumexp trên 5 alignment score |

Hàm loss đầy đủ:

```text
L_total = L_cls + L_mil + L_text_reg
        + 0.01 * L_consistency
        + 0.001 * L_residual
```

- `L_cls`, `L_mil`, `L_text_reg`: loss gốc của VadCLIP.
- `L_consistency`: loss thêm từ Phương pháp 2a để các prompt cùng class không tập trung vào các vị trí thời gian hoàn toàn mâu thuẫn.
- `L_residual`: loss mới của Phương pháp 3, giới hạn độ lớn của `Delta` để prompt không dịch chuyển quá xa embedding nền.
- Không sử dụng normal top-k, abnormal-background hay ranking-margin loss trong cấu hình kết quả bên dưới.

| Classifier AUC | Classifier AP | Alignment AUC | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|
| 86.55 | 29.82 | 85.10 | 67.53 | 8.39 |

Nhận xét:

- mAP tăng khá mạnh, nghĩa là model định vị đoạn sự kiện tốt hơn.
- AUC giảm vì nhiều residual tạo quá nhiều độ tự do và làm không gian text dễ dịch chuyển.

### 6.6. Đóng băng VAD, học learnable prompt và residual trên prototype model

Cách thực hiện:

1. Bắt đầu từ model đã train bằng Phương pháp 2b.
2. Thêm một residual cho mỗi class và khởi tạo bằng 0.
3. Đóng băng phần xử lý visual và classifier để giữ nguyên khả năng biểu diễn video đã học.
4. Trong mỗi forward, prototype vẫn được tính lại từ prompt cùng learnable prefix/postfix hiện tại.
5. Chỉ cập nhật learnable prefix/postfix và residual của class.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo | Warm-start từ model Phương pháp 2b |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Đóng băng |
| Residual `Delta(c)` | Được cập nhật |
| Cách tạo `logits2` | Một alignment map/class từ prototype sau residual |

```text
L_total = L_cls + L_mil + L_text_reg + 0.001 * L_residual
```

Các loss gốc vẫn được tính, nhưng gradient chỉ cập nhật learnable prompt và residual vì phần VAD đã bị đóng băng.

| Classifier AUC | Classifier AP | Alignment AUC | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|
| 86.12 | 30.12 | 86.02 | 66.42 | 6.12 |

Nhận xét:

- Kết quả gần như không đổi.
- Chỉ điều chỉnh phía text trong khi giữ nguyên phần visual không đủ để cải thiện model đã train.
- Cấu hình ban đầu từng đóng băng learnable prefix/postfix và chỉ học residual. Sau khi cho phép learnable prefix/postfix cùng residual được cập nhật, kết quả vẫn chỉ thay đổi rất ít; vì vậy bảng trên được xem là đại diện cho cùng xu hướng thực nghiệm.

### 6.7. Đóng băng VAD, học learnable prompt và residual trên multi-prompt model

Cách thực hiện:

1. Bắt đầu từ model đã train bằng Phương pháp 2a.
2. Thêm một residual riêng cho từng prompt và khởi tạo bằng 0.
3. Đóng băng phần xử lý visual và classifier.
4. Trong mỗi forward, 5 prompt/class vẫn được encode lại cùng learnable prefix/postfix hiện tại.
5. Cập nhật learnable prefix/postfix và 70 residual; CLIP encoder vẫn đóng băng.
6. Tính alignment riêng cho từng prompt rồi tổng hợp bằng weighted logsumexp như Phương pháp 2a.

Mục đích của cấu hình này là kiểm tra liệu chỉ điều chỉnh phía text có thể cải thiện model multi-prompt khi representation visual được giữ nguyên hay không.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo | Warm-start từ model Phương pháp 2a |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Đóng băng |
| Residual `Delta(c,k)` | Được cập nhật; một vector cho mỗi prompt |
| Cách gộp prompt | Weighted logsumexp trên alignment score |

```text
L_total = L_cls + L_mil + L_text_reg
        + 0.01 * L_consistency
        + 0.001 * L_residual
```

| Classifier AUC | Classifier AP | Alignment AUC | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|
| 87.27 | 31.74 | 84.34 | 69.10 | 7.27 |

Nhận xét:

- Classifier gần như không đổi vì các module xử lý video bị đóng băng.
- Alignment giảm, cho thấy residual có thể làm biểu diễn prompt lệch khỏi visual feature.
- Cấu hình hiện tại cập nhật cả learnable prefix/postfix và residual, không còn đóng băng learnable prompt như lần thử ban đầu.
- Kết quả thử lại không khác đáng kể, cho thấy việc mở khóa learnable prompt vẫn chưa đủ để cải thiện model khi toàn bộ phía visual bị đóng băng.

### 6.8. Tiếp tục train toàn bộ VAD với prototype residual

Cách thực hiện:

1. Bắt đầu từ model đã train bằng Phương pháp 2b.
2. Thêm một residual cho mỗi class và khởi tạo bằng 0.
3. Trong mỗi forward, prototype được tính lại từ prompt và learnable prefix/postfix hiện tại.
4. Đóng băng CLIP encoder nhưng tiếp tục cập nhật phần xử lý visual, classifier, learnable prefix/postfix và residual.
5. Sử dụng loss VadCLIP cùng regularization cho residual.

Khác với Phương pháp 6.6, phương pháp này cho phép phần xử lý video thích nghi với biểu diễn class đã được điều chỉnh. Đây là quá trình huấn luyện tiếp từ model 2b, không phải train from scratch.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo | Warm-start từ model Phương pháp 2b |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Residual `Delta(c)` | Được cập nhật |
| Cách tạo `logits2` | Một alignment map/class từ prototype sau residual |

```text
L_total = L_cls + L_mil + L_text_reg + 0.001 * L_residual
```

| Classifier AUC | Classifier AP | Alignment AUC | Alignment AP | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|---:|
| 86.68 | 30.83 | 86.33 | 31.58 | 67.41 | 7.52 |

Nhận xét:

- Tốt hơn cách đóng băng VAD và chỉ cập nhật phía text.
- Alignment AP và mAP tăng nhưng classifier AUC vẫn thấp.
- Việc tiếp tục cập nhật phần xử lý video cũng làm thay đổi anomaly ranking đã học ở giai đoạn trước.

### 6.9. Tiếp tục train toàn bộ VAD với multi-prompt residual

Cách thực hiện:

1. Bắt đầu từ model đã train bằng Phương pháp 2a.
2. Thêm một residual riêng cho từng prompt và khởi tạo bằng 0.
3. Trong mỗi forward, từng prompt được encode lại cùng learnable prefix/postfix hiện tại rồi cộng residual tương ứng.
4. Đóng băng CLIP encoder nhưng tiếp tục cập nhật phần xử lý visual, classifier, learnable prefix/postfix và residual.
5. Tính alignment riêng cho từng prompt và tổng hợp bằng weighted logsumexp.
6. Sử dụng loss VadCLIP, consistency loss và regularization cho residual.

Khác với Phương pháp 6.8, phương pháp này giữ 5 prompt riêng thay vì sử dụng một class prototype. Cấu hình có khả năng biểu diễn đa dạng hơn nhưng cũng có nhiều residual phải học hơn.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo | Warm-start từ model Phương pháp 2a |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Residual `Delta(c,k)` | Được cập nhật; một vector cho mỗi prompt |
| Cách gộp prompt | Weighted logsumexp trên alignment score |

```text
L_total = L_cls + L_mil + L_text_reg
        + 0.01 * L_consistency
        + 0.001 * L_residual
```

| Classifier AUC | Classifier AP | Alignment AUC | Alignment AP | Ano-AUC | avg mAP |
|---:|---:|---:|---:|---:|---:|
| 86.74 | 29.39 | 86.33 | 27.30 | 67.35 | 8.25 |

Nhận xét:

- Khả năng định vị theo thời gian tốt hơn baseline.
- Khả năng xếp hạng normal/abnormal trên toàn bộ test set giảm.
- Việc tăng độ linh hoạt giúp mAP nhưng không cải thiện classifier AUC.

### So sánh nhanh các phương pháp residual

| Phương pháp | Model bắt đầu | Text đại diện/class | Phần được cập nhật |
|---|---|---|---|
| 6.4 | Khởi tạo mới | 1 prototype + 1 residual | VAD + learnable prompt + residual |
| 6.5 | Khởi tạo mới | 5 prompt + 5 residual | VAD + learnable prompt + residual |
| 6.6 | Model prototype đã train | 1 prototype + 1 residual | Learnable prompt + residual |
| 6.7 | Model multi-prompt đã train | 5 prompt + 5 residual | Learnable prompt + residual |
| 6.8 | Model prototype đã train | 1 prototype + 1 residual | VAD + learnable prompt + residual |
| 6.9 | Model multi-prompt đã train | 5 prompt + 5 residual | VAD + learnable prompt + residual |

Cách đọc bảng:

- Phương pháp 6.4 và 6.5 train model từ đầu.
- Phương pháp 6.6 và 6.7 đóng băng VAD, chỉ cập nhật learnable prompt và residual.
- Phương pháp 6.8 và 6.9 tiếp tục huấn luyện phần xử lý video, learnable prompt và residual.

### 6.10. Thêm một nhánh prompt hỗ trợ

Cách thực hiện:

1. Giữ nguyên tên class và nhánh alignment của VadCLIP.
2. Tạo một class prototype từ các prompt của mỗi class.
3. Thêm một nhánh alignment phụ giữa visual feature và class prototype.
4. Tính thêm MIL loss trên nhánh phụ.
5. Tính thêm cosine loss giữa class prototype và biểu diễn class của nhánh gốc.
6. Fine-tune model từ checkpoint của tác giả.

Phương pháp này không thay class text gốc. Prompt chỉ đóng vai trò là tín hiệu hỗ trợ trong quá trình fine-tune.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo | Fine-tune từ checkpoint VadCLIP của tác giả |
| Nhánh alignment gốc | Giữ tên class và learnable prefix/postfix của VadCLIP |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix của nhánh gốc | Có sử dụng và được cập nhật |
| Prototype của nhánh phụ | Encode trước bằng CLIP rồi giữ cố định |
| Các module xử lý visual và classifier | Được cập nhật |
| Cách tạo output | `logits2` gốc và một `prototype_logits` phụ cùng có kích thước `[T,14]` |

Hàm loss:

```text
L_total = L_cls + L_mil + L_text_reg
        + lambda_proto_mil * L_proto_mil
        + lambda_proto_text * L_proto_text
```

- `L_proto_mil` áp dụng MIL classification lên alignment giữa visual snippet và prototype của nhánh phụ.
- `L_proto_text` kéo biểu diễn class của nhánh gốc gần prototype tương ứng.
- Với prototype dài, các hệ số được dùng là `lambda_proto_mil = 1.0` và `lambda_proto_text = 0.05`.
- Với compact prototype, hai hệ số được giảm còn `0.3` và `0.01` để hạn chế nhánh phụ làm lệch model gốc.

| Prompt dùng ở nhánh phụ | Classifier AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|
| Prototype dài | 86.40 | 66.75 | 4.74 |
| Keyword phrase | 87.26 | 68.96 | 7.49 |
| Action phrase | 87.25 | 69.04 | 7.26 |
| Câu thị giác ngắn | 87.32 | 69.27 | 6.44 |

Nhận xét:

- Rút ngắn prompt cải thiện kết quả rõ rệt.
- Nhánh phụ vẫn tạo thêm một mục tiêu text khác, có thể kéo visual feature lệch khỏi nhánh class gốc.

### 6.11. Dùng class prompt thay trực tiếp class text gốc

Cách thực hiện:

1. Chọn nhiều prompt mô tả cho mỗi class.
2. Gộp các prompt thành một class prototype.
3. Gắn learnable prompt của VadCLIP vào class prototype.
4. Dùng class prototype thay cho tên class trong nhánh alignment.
5. Train bằng loss VadCLIP gốc, không thêm nhánh phụ.

Hai cách biểu diễn text được thử:

- **Prompt only**: chỉ sử dụng prompt mô tả class.
- **Class + prompt**: ghép tên class với prompt mô tả class.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo | Thử cả checkpoint tác giả và train from scratch |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Nhiều prompt trong một class | Gộp thành một prototype trước khi alignment |
| Cách tạo `logits2` | Một alignment map/class, kích thước `[T,14]` |
| Loss bổ sung | Không có |

```text
L_total = L_cls + L_mil + L_text_reg
```

Điểm khác với Phương pháp 2b nằm ở nội dung text được dùng để thay class name và các cấu hình prompt-only/class-plus-prompt; kiến trúc loss vẫn là VadCLIP gốc.

| Cấu hình | Cách khởi tạo | Classifier AUC | Ano-AUC | avg mAP |
|---|---|---:|---:|---:|
| Prompt only | Checkpoint tác giả | 86.71 | 67.23 | 3.68 |
| Class + prompt | Checkpoint tác giả | 86.85 | 68.04 | 3.84 |
| Prompt only | Train từ đầu | 86.73 | 67.61 | 6.22 |
| Class + prompt | Train từ đầu | 85.93 | 66.24 | 6.40 |

Nhận xét:

- Thay class text trong một checkpoint đã train làm video-text alignment thay đổi mạnh.
- Train từ đầu giảm mismatch nhưng vẫn thấp hơn baseline.
- Gộp nhiều prompt thành một embedding tiếp tục làm mất một phần semantic diversity.

### 6.12. So sánh các prompt bank khi giữ nhiều prompt riêng

Cách thực hiện:

1. Giữ nguyên kiến trúc và cách train của Phương pháp 2a.
2. Thay đổi loại prompt và số prompt của mỗi class.
3. Giữ riêng từng prompt khi tính alignment.
4. Tổng hợp prompt score sau khi đã so khớp với visual feature.

Mục đích của nhóm thí nghiệm này là đánh giá ảnh hưởng của nội dung và số lượng prompt khi kiến trúc model không thay đổi.

Cấu hình và thành phần được học:

| Thành phần | Thiết lập |
|---|---|
| Khởi tạo | Train from scratch |
| CLIP Image/Text Encoder | Đóng băng |
| Learnable prefix/postfix | Có sử dụng và được cập nhật |
| Các module xử lý visual và classifier | Được cập nhật |
| Prompt text và trọng số | Cố định |
| Cách tạo alignment | Giữ từng prompt riêng, tạo `[T,14,K]` với `K` là số prompt/class |
| Cách gộp prompt | Weighted logsumexp sau khi tính alignment |

Với các cấu hình có nhiều hơn một prompt, hàm loss là:

```text
L_total = L_cls + L_mil + L_text_reg + 0.01 * L_consistency
```

Với cấu hình chỉ có một prompt/class, `L_consistency` bằng 0 vì không có nhiều prompt để so sánh mức tập trung theo thời gian. Các regularization normal/abnormal bổ sung không được dùng trong bảng kết quả này.

| Prompt bank | Số prompt/class | Classifier AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|
| Prototype sự kiện dài | 3 | 87.12 | 68.60 | 7.45 |
| Caption thị giác ban đầu | 10 | 87.15 | 69.43 | 9.56 |
| Caption thị giác đã rút gọn | 5 | 87.05 | 68.65 | 7.30 |
| Keyword, chỉ chọn một prompt | 1 | 86.45 | 66.60 | 5.91 |
| Keyword có trọng số | 5 | 87.15 | 68.19 | 9.14 |
| Visual keyword, checkpoint tốt nhất theo mAP | 5 | 86.59 | 67.78 | 10.24 |

Nhận xét:

- Giữ nhiều prompt thường tốt hơn chỉ chọn một prompt.
- Prompt ngắn thường tốt hơn prompt dài, nhưng rút ngắn không bảo đảm AUC tăng.
- Một số prompt bank tăng mAP mạnh nhưng vẫn làm classifier AUC và Ano-AUC giảm.
- Vì kiến trúc gần như không đổi, kết quả cho thấy nội dung prompt ảnh hưởng lớn đến chất lượng fine-tune.

---

## 7. Bảng tổng hợp các thí nghiệm chính

| Phương pháp | Classifier AUC | Classifier AP | Alignment AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|
| Baseline gốc | 88.02 | 33.56 | 85.69 | 70.23 | 6.68 |
| Một prompt/class | 87.15 | 30.24 | 86.91 | 68.32 | 7.24 |
| Năm prompt có trọng số | **87.27** | 31.74 | 86.17 | **69.10** | 7.62 |
| Một weighted prototype/class | 86.12 | 30.12 | 86.01 | 66.42 | 6.38 |
| Prototype có residual, train từ đầu | 86.76 | **31.91** | 85.25 | 68.64 | 7.18 |
| Mỗi prompt có residual, train từ đầu | 86.55 | 29.82 | 85.10 | 67.53 | **8.39** |
| Warm-start prototype, học learnable prompt và residual | 86.12 | 30.12 | 86.02 | 66.42 | 6.12 |
| Warm-start multi-prompt, học learnable prompt và residual | **87.27** | 31.74 | 84.34 | **69.10** | 7.27 |
| Continue prototype model và residual | 86.68 | 30.83 | **86.33** | 67.41 | 7.52 |
| Continue multi-prompt model và residual | 86.74 | 29.39 | **86.33** | 67.35 | 8.25 |

## 8. Kết luận chung

- Chưa có phương pháp prompt nào vượt baseline gốc về classifier AUC và Ano-AUC.
- Giữ riêng nhiều prompt có trọng số hiệu quả hơn chọn một prompt hoặc nén prompt thành centroid.
- Prototype dài không phù hợp vì chứa nhiều thông tin cấp video mà một frame không thể hiện được.
- Prompt ngắn, tập trung vào danh từ và động từ thị giác phù hợp với CLIP hơn.
- Residual giúp model linh hoạt hơn và có thể tăng mAP, nhưng cũng dễ làm không gian video-text bị lệch.
- Score trong text space giúp loại prompt rõ ràng không phù hợp, nhưng không bảo đảm prompt sẽ giúp AUC sau training.
- Kết quả chung cho thấy prompt thường hỗ trợ xác định class và đoạn thời gian, nhưng chưa giải quyết tốt việc phân biệt abnormal frame với normal frame trên toàn bộ tập test.
