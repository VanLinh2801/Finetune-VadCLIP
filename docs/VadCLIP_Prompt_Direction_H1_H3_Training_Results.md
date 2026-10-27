# Nhật Ký Thí Nghiệm: Ba Hướng Dùng Prompt Cho VadCLIP

## 1. Bối Cảnh Và Mục Tiêu

VadCLIP là mô hình phát hiện bất thường trong video giám sát (UCF-Crime). Model nhận chuỗi frame đã được mã hóa sẵn, rồi dự đoán đoạn nào bất thường. Một phần quan trọng của VadCLIP là **nhánh căn chỉnh với text**: mỗi loại sự kiện (nổ, đánh nhau, bắt giữ, …) được mô tả bằng câu/cụm từ ngắn, đưa qua CLIP text encoder để tạo “mỏ neo” ngôn ngữ. Model so khớp hình ảnh với các mỏ neo này để hỗ trợ phát hiện bất thường.

Các thí nghiệm trước cho thấy:

- mô tả quá dài dễ nhiễu;
- dùng vài cụm từ khóa ngắn (keyword prompt) ổn định hơn;
- giữ nhiều prompt/class và gán trọng số thường tốt hơn chỉ chọn một prompt duy nhất.

Nhóm thí nghiệm này đi tiếp một bước: **cùng một bộ prompt đã chọn**, nhưng thử **bốn cách khác nhau để đưa prompt vào model**, nhằm trả lời:

> Cách biểu diễn text nào giúp phân biệt frame bình thường / bất thường tốt nhất? Việc gộp nhiều prompt thành một vector, rồi cho phép chỉnh nhẹ vector đó, có giúp ích không?

Bốn hướng được so sánh:

| Hướng | Nói đơn giản |
|---|---|
| Hướng 1 | Mỗi class chỉ lấy **1 câu prompt tốt nhất** |
| Hướng 2a | Mỗi class giữ **5 câu prompt**, câu tốt hơn được trọng số cao hơn |
| Hướng 2b | Gộp 5 câu thành **một vector đại diện** cho class |
| Hướng 3 | Lấy vector đại diện đó, rồi **học thêm một chỉnh sửa nhỏ** khi train |

Hai mốc so sánh chính:

| Mốc | Ý nghĩa | Chỉ số classifier AUC |
|---|---|---:|
| Baseline gốc | Checkpoint VadCLIP chuẩn trên UCF-Crime | 88.02 |
| Baseline tự train | Train lại từ đầu chỉ với tên class gốc (tham chiếu lịch sử) | 87.19 |

Chỉ số chính dùng trong báo cáo:

- **classifier AUC**: phân biệt frame bất thường vs bình thường (càng cao càng tốt);
- **Ano-AUC**: AUC chỉ trên các video có bất thường;
- **avg mAP**: khả năng định vị đoạn bất thường theo thời gian.

---

## 2. Cách Tạo Prompt

### 2.1. Prompt viết thế nào?

Mỗi class trong UCF-Crime được mô tả bằng các **cụm từ ngắn**, gần với những gì mắt người nhìn thấy trong frame: người, hành động, vật thể, hiệu ứng. Không viết thành đoạn văn dài.

Ví dụ ý tưởng viết:

| Class | Kiểu cụm từ mong muốn |
|---|---|
| Explosion | nổ, lửa cầu, khói, mảnh vỡ |
| Arrest | cảnh sát khống chế / còng tay nghi phạm |
| Fighting | nhiều người đấm đá, ẩu đả |
| Assault | một người tấn công người khác |
| Normal | đi bộ, nói chuyện, xe chạy, mua sắm |

### 2.2. Từ nhiều ứng viên đến 5 prompt/class

Quy trình chọn prompt gồm các bước:

1. **Thu thập ứng viên**  
   Nhiều cụm từ ngắn cho mỗi class (từ các bộ keyword / visual keyword đã thử trước đó, rồi lọc–bổ sung).

2. **Chấm điểm trong không gian text của CLIP** (chưa train video)  
   Mỗi prompt được đánh giá xem:
   - có gần nghĩa với class đích không;
   - có tách khỏi các class dễ lẫn không (ví dụ Explosion dễ lẫn Arson / RoadAccidents);
   - có nhất quán với các prompt khác cùng class không.

3. **Lọc đa dạng rồi giữ Top-5**  
   Không chỉ lấy 5 prompt điểm cao nhất nếu chúng gần trùng nhau. Có bước đa dạng hóa để tránh 5 câu gần như cùng một ý.

4. **Gán trọng số**  
   Trong 5 prompt còn lại, prompt điểm cao hơn nhận trọng số lớn hơn (softmax theo điểm). Khi train, cả 5 vẫn được dùng, nhưng ảnh hưởng không đều nhau.

### 2.3. Ví dụ bộ 5 prompt đã dùng

Khi đưa vào model, mỗi cụm thường được gắn thêm tên class phía trước (ví dụ `Explosion` + `building blast debris`), để neo rõ hơn với nhãn sự kiện.

**Explosion**

| Trọng số | Prompt |
|---:|---|
| 0.219 | building blast debris |
| 0.206 | blast fireball debris |
| 0.199 | vehicle explosion smoke |
| 0.196 | detonation shockwave fragments |
| 0.180 | fireball plume rubble |

**Arrest**

| Trọng số | Prompt |
|---:|---|
| 0.218 | police restraining suspect |
| 0.204 | police handcuffing suspect |
| 0.200 | patrol car suspect custody |
| 0.190 | officer escorting suspect |
| 0.187 | officer searching suspect |

**Fighting**

| Trọng số | Prompt |
|---:|---|
| 0.231 | crowd joining fight |
| 0.204 | people exchanging punches |
| 0.202 | people kicking each other |
| 0.184 | group brawl street |
| 0.180 | fighters grappling ground |

**Assault**

| Trọng số | Prompt |
|---:|---|
| 0.215 | person punching passerby |
| 0.208 | person kicking target |
| 0.202 | person stomping target |
| 0.199 | person knocking target down |
| 0.176 | attacker swinging weapon |

**Normal**

| Trọng số | Prompt |
|---:|---|
| 0.235 | people standing talking |
| 0.197 | vehicle driving roadway |
| 0.197 | customer browsing shelf |
| 0.192 | cashier scanning products |
| 0.179 | pedestrian walking sidewalk |

Cùng một bộ 5 prompt + trọng số như trên được dùng cho cả bốn hướng. Chỉ khác **cách model dùng chúng**.

---

## 3. Bốn Hướng Dùng Prompt

### 3.1. Hướng 1 — Chỉ lấy 1 prompt tốt nhất

Với mỗi class, chỉ giữ câu có trọng số cao nhất.

Ví dụ:

| Class | Prompt dùng |
|---|---|
| Explosion | building blast debris |
| Arrest | police restraining suspect |
| Fighting | crowd joining fight |
| Assault | person punching passerby |
| Normal | people standing talking |

**Ưu điểm:** đơn giản, ít nhiễu.  
**Rủi ro:** một class bất thường thường có nhiều “mặt” (nổ nhà / nổ xe / lửa cầu). Một câu có thể không bao phủ hết.

### 3.2. Hướng 2a — Giữ 5 prompt, có trọng số (multi-prompt)

Model encode cả 5 câu của mỗi class thành 5 vector text riêng. Khi căn chỉnh với hình ảnh, các vector này được kết hợp theo trọng số: câu quan trọng hơn đóng góp mạnh hơn.

Với `Explosion`, model “nhìn” đồng thời các ý như *building blast debris*, *blast fireball debris*, *vehicle explosion smoke*, … thay vì chỉ một ý.

Đây là hướng gần nhất với cấu hình keyword có trọng số từng làm tốt trước đây.

### 3.3. Hướng 2b — Gộp 5 prompt thành một vector đại diện

Vẫn encode 5 câu, nhưng **trước khi** đưa vào bước căn chỉnh, gộp chúng thành một vector duy nhất cho mỗi class:

```text
vector đại diện class =
  chuẩn hóa( tổng có trọng số của 5 embedding prompt )
```

Sau bước này, mỗi class chỉ còn **một** mỏ neo text. Model không còn “chọn mềm” giữa nhiều câu theo từng đoạn video.

**Ý tưởng:** cô đọng semantic của class thành một điểm trong không gian CLIP.  
**Rủi ro:** mất đi sự linh hoạt của nhiều kênh prompt.

### 3.4. Hướng 3 — Học chỉnh sửa nhỏ trên vector đại diện

Bắt đầu từ cùng vector đại diện như Hướng 2b (giữ cố định, không học lại phần encode prompt). Mỗi class thêm một vector chỉnh sửa `Δ`, khởi tạo bằng 0:

```text
text dùng khi train =
  chuẩn hóa( vector đại diện + Δ )
```

- Phần encode ngôn ngữ của CLIP vẫn đóng băng.
- Model học `Δ` cùng các lớp xử lý video, train từ đầu (không lấy sẵn trọng số đã train của Hướng 2b).
- Có phạt nhẹ để `Δ` không lệch quá xa:

```text
phạt = trung bình độ lớn bình phương của Δ
hệ số phạt ≈ 0.001
```

Lúc mới bắt đầu, `Δ = 0` nên hành vi gần như Hướng 2b. Nếu residual hữu ích, kết quả sẽ nhỉnh hơn 2b nhưng không nhất thiết vượt được hướng giữ 5 kênh riêng (2a).

---

## 4. Cấu Hình Train Chung

Để so sánh công bằng, bốn hướng dùng chung thiết lập:

| Thành phần | Thiết lập |
|---|---|
| Dữ liệu | UCF-Crime, 14 class |
| Cách viết prompt khi đưa vào model | tên class + cụm từ prompt |
| CLIP text encoder | đóng băng (không fine-tune) |
| Phần xử lý video / phân loại | train từ đầu |
| Learning rate | 2e-5 |
| Số epoch tối đa | 10 |
| Dừng sớm | nếu classifier AUC trên tập test không cải thiện trong 3 epoch |
| Các mất mát phụ tách nền / đè score normal | tắt |

Checkpoint cuối cùng của mỗi hướng là checkpoint có **classifier AUC cao nhất** trong quá trình train (không chọn theo mAP).

---

## 5. Kết Quả

### 5.1. Nhìn tổng quan

| Hướng | classifier AUC | classifier AP | alignment AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|
| Baseline gốc | 88.02 | 33.56 | 85.69 | 70.23 | 6.68 |
| Hướng 1 — Top-1 | 87.15 | 30.24 | 86.91 | 68.32 | 7.24 |
| Hướng 2a — 5 prompt có trọng số | **87.27** | **31.74** | 86.17 | **69.10** | **7.62** |
| Hướng 2b — một vector đại diện | 86.12 | 30.12 | 86.01 | 66.42 | 6.38 |
| Hướng 3 — residual trên vector đại diện | 86.76 | 31.91 | 85.25 | 68.64 | 7.18 |

Khoảng cách classifier AUC so với các mốc:

| Hướng | So với baseline gốc 88.02 | So với baseline tự train 87.19 |
|---|---:|---:|
| Hướng 1 | −0.87 | −0.04 |
| Hướng 2a | −0.75 | **+0.08** |
| Hướng 2b | −1.90 | −1.07 |
| Hướng 3 | −1.26 | −0.43 |

**Tóm tắt một dòng:**  
Hướng 2a tốt nhất trong nhóm; Hướng 1 gần sát; Hướng 2b yếu rõ; Hướng 3 kéo 2b lên được một đoạn nhưng chưa bắt kịp 2a.

### 5.2. Hướng 1 — Top-1

Train peaking ở epoch 3 (AUC 87.15), sau đó hơi giảm rồi đi ngang.

| Epoch | classifier AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|
| 1 | 86.08 | 66.43 | 5.87 |
| 2 | 86.77 | 67.91 | 7.25 |
| 3 | **87.15** | **68.32** | 7.24 |
| 4 | 86.40 | 66.24 | 5.98 |
| 5 | 86.74 | 67.18 | 6.79 |
| 6 | 86.74 | 67.21 | 6.53 |

So với baseline gốc: AUC −0.87, Ano-AUC −1.91, nhưng avg mAP +0.56.

Một số class đổi mạnh:

| Class | Baseline | Sau Hướng 1 | Thay đổi |
|---|---:|---:|---:|
| Abuse | 56.98 | 64.76 | +7.78 |
| Assault | 84.34 | 91.57 | +7.23 |
| Vandalism | 70.97 | 76.35 | +5.38 |
| Explosion | 59.18 | 46.42 | −12.76 |
| Arson | 76.85 | 72.51 | −4.34 |

Nhận xét: chỉ một prompt vẫn “sống được” trên bộ prompt này (gần baseline tự train), nhưng class khó như Explosion bị tổn thất lớn.

### 5.3. Hướng 2a — Năm prompt có trọng số

Đây là hướng mạnh nhất. Epoch 3 đạt AUC cao nhất; epoch 4 có mAP hơi cao hơn nhưng AUC thấp hơn nên không được chọn.

| Epoch | classifier AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|
| 1 | 86.21 | 66.74 | 6.54 |
| 2 | 86.70 | 67.88 | 5.80 |
| 3 | **87.27** | **69.10** | 7.62 |
| 4 | 86.91 | 68.01 | **7.97** |
| 5 | 86.95 | 68.22 | 7.84 |
| 6 | 86.69 | 67.55 | 7.27 |

So với baseline gốc: AUC −0.75, Ano-AUC −1.13, avg mAP +0.94.  
So với baseline tự train: AUC **+0.08**.

Class đáng chú ý:

| Class | Baseline | Sau Hướng 2a | Thay đổi |
|---|---:|---:|---:|
| Vandalism | 70.97 | 79.95 | +8.98 |
| Abuse | 56.98 | 62.69 | +5.71 |
| Shooting | 65.63 | 69.47 | +3.84 |
| Explosion | 59.18 | 53.14 | −6.04 |
| Arson | 76.85 | 72.05 | −4.80 |

So với Hướng 1, thiệt hại ở Explosion nhẹ hơn nhiều (−6 so với −13). Điều này hỗ trợ ý: **giữ nhiều biến thể prompt giúp class đa dạng thị giác ổn định hơn**.

### 5.4. Hướng 2b — Một vector đại diện

AUC tốt nhất ở epoch 2 (86.12). Epoch 3 có avg mAP rất cao (8.92) nhưng AUC không nhỉnh hơn nên early-stop không chọn epoch đó.

| Epoch | classifier AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|
| 1 | 85.03 | 64.72 | 2.75 |
| 2 | **86.12** | 66.42 | 6.38 |
| 3 | 86.03 | 66.71 | **8.92** |
| 4 | 85.87 | 65.82 | 7.68 |
| 5 | 86.08 | 66.39 | 7.68 |

So với baseline gốc: AUC −1.90, Ano-AUC −3.81, avg mAP −0.30.

Class yếu nổi bật: Explosion (−13.67), Robbery (−7.16).  
Abuse lại tăng khá (+9.23), cho thấy gộp vector không phải “xấu đều mọi class”, nhưng tổng thể phân phối score toàn cục kém hơn rõ so với giữ 5 kênh.

### 5.5. Hướng 3 — Residual trên vector đại diện

Epoch 3 là checkpoint cuối theo AUC. Epoch 5 có mAP cao hơn (8.54).

| Epoch | classifier AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|
| 1 | 84.68 | 64.82 | 2.23 |
| 2 | 85.98 | 66.93 | 6.45 |
| 3 | **86.76** | **68.64** | 7.18 |
| 4 | 86.32 | 67.15 | 7.56 |
| 5 | 86.67 | 68.19 | **8.54** |
| 6 | 86.66 | 68.21 | 8.39 |

So với Hướng 2b (cùng ý tưởng vector đại diện):

| Chỉ số | Hướng 2b | Hướng 3 | Thay đổi |
|---|---:|---:|---:|
| classifier AUC | 86.12 | 86.76 | +0.64 |
| Ano-AUC | 66.42 | 68.64 | +2.22 |
| avg mAP | 6.38 | 7.18 | +0.80 |

Residual **có giúp** nhánh prototype. Nhưng so với Hướng 2a vẫn thấp hơn khoảng 0.5 điểm AUC và khoảng 0.5 điểm Ano-AUC.

Explosion vẫn yếu (−9.91 so với baseline), dù đã đỡ hơn Hướng 2b (−13.67).

---

## 6. Nhận Xét Cho Người Đọc Nhanh

1. **Cách hay nhất trong nhóm này là giữ 5 prompt có trọng số (Hướng 2a).**  
   Model được “nhìn” nhiều biến thể của cùng một sự kiện, nên vừa gần baseline tự train, vừa có Ano-AUC và mAP tốt nhất nhóm.

2. **Chỉ lấy 1 prompt tốt nhất (Hướng 1) vẫn ổn**, gần ngang baseline tự train, nhưng thường thiệt hơn ở các class khó / đa dạng như Explosion.

3. **Gộp hết thành một vector đại diện (Hướng 2b) làm mất linh hoạt.**  
   Đây là bước tụt rõ trên classifier AUC và Ano-AUC, dù đôi lúc mAP theo epoch nhìn hấp dẫn.

4. **Cho phép chỉnh nhẹ vector đại diện (Hướng 3) cứu được một phần so với 2b**, đặc biệt Ano-AUC.  
   Nhưng residual không thay thế được lợi thế của việc giữ nhiều prompt riêng như Hướng 2a.

5. **Luôn có đánh đổi AUC ↔ mAP.**  
   Nhiều hướng có epoch mAP cao hơn checkpoint được chọn. Vì mục tiêu chính của chuỗi thí nghiệm này là frame-level AUC, việc chọn checkpoint theo classifier AUC là nhất quán; nếu ưu tiên định vị đoạn bất thường, cần quy tắc chọn khác.

6. **Explosion vẫn là “điểm nghẽn” chung.**  
   Prompt ngắn kiểu *blast / fireball / debris* vẫn dễ chồng với cháy và tai nạn giao thông. Cải thiện prompt giúp giảm thiệt hại (2a tốt hơn 1 và 2b), nhưng chưa giải quyết triệt để.

### Cấu hình nên nhớ nếu chỉ giữ một kết luận

```text
Dùng 5 cụm prompt ngắn / class
Gán trọng số theo độ phù hợp text
Đưa vào model dưới dạng: tên class + cụm prompt
Train phần video từ đầu, giữ nguyên CLIP text
Chọn checkpoint theo classifier AUC
```

Trên bộ prompt hiện tại, cấu hình đó đạt khoảng **classifier AUC 87.27**, cao hơn baseline tự train một chút, nhưng vẫn thấp hơn baseline gốc 88.02 khoảng 0.75 điểm. Hướng residual trên prototype là hướng phụ hữu ích khi muốn cứu nhánh “một vector/class”, không phải lựa chọn mặc định để thay multi-prompt.
