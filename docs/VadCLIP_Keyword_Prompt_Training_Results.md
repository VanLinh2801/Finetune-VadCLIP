# Nhật Ký Thí Nghiệm: Keyword Prompt Cho VadCLIP

## 1. Mục Tiêu

Sau các thí nghiệm dùng prompt dạng câu mô tả dài, một hướng mới được kiểm tra là rút gọn prompt về dạng **keyword prompt**. Thay vì mô tả đầy đủ một sự kiện bằng câu dài, mỗi prompt chỉ giữ lại các danh từ, động từ hoặc cụm từ thị giác quan trọng nhất.

Câu hỏi chính của thí nghiệm:

> Nếu prompt ngắn hơn, ít stopword hơn và tập trung vào tín hiệu thị giác chính, model có giảm nhiễu semantic và giữ AUC tốt hơn không?

Động lực của hướng này đến từ hai quan sát trước đó:

- Prompt dài có thể chứa nhiều chi tiết không thể nhận biết rõ trong một frame đơn lẻ.
- Prompt càng dài thì CLIP text embedding càng dễ bị kéo bởi các từ phụ, scene context hoặc mô tả chuỗi hành động.
- VadCLIP làm việc ở mức frame/snippet feature, nên prompt nên gần với caption ảnh hoặc cụm visual concept hơn là mô tả toàn bộ video.

Vì vậy, keyword prompt được dùng để kiểm tra xem một biểu diễn ngắn, trực tiếp và ít nhiễu hơn có giúp model học ổn định hơn hay không.

## 2. Cách Làm

### 2.1. Tạo Keyword Prompt

Với mỗi class trong UCF-Crime, 5 keyword prompt được chuẩn bị. Các prompt này ưu tiên:

- danh từ mô tả actor hoặc object;
- động từ mô tả hành động chính;
- một số tính từ hoặc hiệu ứng thị giác thật sự quan trọng;
- hạn chế các từ mơ hồ như `routine`, `sudden`, `stolen item`, hoặc mô tả quá dài.

Ví dụ:

| Class | Ví dụ keyword prompt |
|---|---|
| Explosion | rupture explosion debris |
| Arson | person setting fire |
| Arrest | officers restraining suspect |
| Fighting | two sides fighting |
| Vandalism | graffiti spraying wall |
| Normal | routine public activity |

Các prompt này không nhằm viết thành câu hoàn chỉnh. Mục tiêu là tạo các cụm từ ngắn giúp CLIP tập trung vào concept thị giác chính của từng class.

### 2.2. Weighted Keyword 5x

`Weighted keyword 5x` là bộ prompt keyword đầu tiên trong nhóm thí nghiệm này. Tên gọi này có nghĩa là:

- `keyword`: prompt được viết dưới dạng cụm từ khóa ngắn, không phải câu mô tả dài.
- `5x`: mỗi class có đúng 5 prompt.
- `weighted`: 5 prompt trong cùng một class không đóng góp ngang nhau, mà được gán trọng số dựa trên điểm text-space.

Mỗi class vẫn giữ 5 prompt thay vì chọn duy nhất một prompt. Lý do là một class bất thường thường có nhiều biểu hiện khác nhau, nếu chỉ dùng một prompt thì rất dễ làm hẹp semantic của class.

Với mỗi prompt, một điểm số text-space được tính dựa trên:

```text
score =
  0.45 * class_anchor_similarity
+ 0.35 * anchor_margin
+ 0.15 * intra_class_similarity
- 0.05 * inter_class_similarity
```

Trong đó:

- `class_anchor_similarity`: prompt gần tên class gốc.
- `anchor_margin`: prompt gần class gốc hơn các class khác.
- `intra_class_similarity`: prompt nhất quán với các prompt cùng class.
- `inter_class_similarity`: prompt không quá gần prompt của class khác.

Sau đó, score được chuyển thành trọng số cho từng prompt. Khi train, model vẫn encode cả 5 prompt/class, nhưng prompt có score cao hơn sẽ ảnh hưởng mạnh hơn trong quá trình aggregate.

Ví dụ với `Fighting`, thay vì chỉ dùng một prompt như `two sides fighting`, model dùng cả nhóm prompt:

```text
two sides fighting
opponents exchanging blows
crowd combat struggle
mutual punches brawl
groups grappling striking
```

Điểm khác biệt của `weighted keyword 5x` là mỗi prompt trong nhóm này có trọng số riêng, giúp prompt rõ nghĩa hơn ảnh hưởng mạnh hơn prompt yếu hơn.

Ví dụ với `Explosion`, các prompt và trọng số chính:

| Prompt | Trọng số |
|---|---:|
| rupture explosion debris | 0.236 |
| detonation fireball smoke | 0.217 |
| sudden blast cloud | 0.203 |
| blast shockwave debris | 0.196 |
| outward fragments pressure | 0.149 |

### 2.3. Visual-Keyword 5x

`Visual-keyword 5x` là bộ prompt keyword thứ hai. Bộ này vẫn giữ 5 prompt/class, nhưng cách viết khác `weighted keyword 5x`.

Nếu `weighted keyword 5x` vẫn có một số cụm mang tính semantic class như `routine public activity`, `deliberate ignition object`, `robber intimidation theft`, thì `visual-keyword 5x` cố gắng viết gần hơn với những thứ có thể nhìn thấy trực tiếp trong frame.

Ví dụ:

```text
blast fireball debris
building blast debris
accelerant flames ground
police restraining suspect
sign breaking impact
```

Nói ngắn gọn:

| Biến thể | Ý nghĩa |
|---|---|
| Weighted keyword 5x | 5 prompt/class dạng keyword, có trọng số, vẫn giữ một số từ semantic class |
| Visual-keyword 5x | 5 prompt/class ngắn hơn, trực tiếp hơn, tập trung vào actor/action/object/effect nhìn thấy trong frame |

Song song với `visual-keyword 5x`, công thức chấm điểm mới được thử, trong đó phần phạt được tập trung vào prompt của các class dễ lẫn thay vì toàn bộ class khác. Ý tưởng là giảm chồng lấn ở các cặp như:

```text
Explosion - Arson - RoadAccidents
Assault - Fighting - Abuse
Robbery - Stealing - Burglary - Shoplifting
```

Tuy nhiên, công thức mới chủ yếu tác động trong text-space. Kết quả fine-tune vẫn cần được kiểm tra bằng metric video-level.

## 3. Cấu Hình Training

Kiến trúc chính của VadCLIP được giữ nguyên. Thay đổi nằm ở phần text prompt dùng để tạo class embedding.

Thiết lập chính:

| Thành phần | Cấu hình |
|---|---|
| Dataset | UCF-Crime |
| Số class | 14 |
| Số prompt mỗi class | 5 |
| Prompt mode | class name + keyword prompt |
| Text encoder | CLIP text encoder, frozen |
| Visual/model-specific layers | train từ scratch |
| Learning rate | 2e-5 |
| Số epoch | 3-10 tùy thí nghiệm |
| Inference | giống VadCLIP gốc, không dùng description |

Loss chính:

```text
L_total =
  L_cls
+ L_mil
+ L_text_reg
+ lambda_consistency * L_consistency
```

Trong các thí nghiệm dưới đây, description/video-level text không được dùng ở inference. Prompt chỉ thay đổi cách tạo class text embedding cho nhánh alignment của VadCLIP.

Hai mốc so sánh được dùng:

| Mốc so sánh | Ý nghĩa | classifier AUC |
|---|---|---:|
| Baseline gốc VadCLIP checkpoint | checkpoint công bố/đã train sẵn trên UCF-Crime | 88.02 |
| Baseline tự train | VadCLIP train lại từ scratch bằng class prompt gốc | 87.19 |

## 4. Kết Quả Sau Fine-Tune

### 4.1. Kết Quả Tổng Quan

Bảng dưới đây so sánh các thí nghiệm keyword prompt với baseline gốc VadCLIP.

| Mô hình | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | avg mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline gốc VadCLIP | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 11.72 | 7.83 | 6.40 | 4.53 | 2.93 | 6.68 |
| Keyword mixed top-1 | 86.45 | 27.63 | 85.00 | 24.20 | 66.60 | 10.72 | 7.67 | 4.82 | 3.92 | 2.41 | 5.91 |
| Weighted keyword 5x | 87.15 | 30.03 | 85.34 | 24.98 | 68.19 | 15.56 | 11.70 | 8.49 | 5.30 | 4.66 | 9.14 |
| Visual-keyword 5x, best AUC | 86.80 | 30.21 | 84.32 | 23.78 | 67.71 | 14.21 | 11.90 | 8.69 | 4.42 | 3.10 | 8.46 |
| Visual-keyword 5x, best mAP | 86.59 | 29.57 | 85.79 | 24.34 | 67.78 | 14.55 | 12.56 | 9.55 | 8.32 | 6.21 | 10.24 |

So với baseline gốc `88.02` và baseline tự train `87.19`, phần classifier AUC thay đổi như sau:

| Mô hình | classifier AUC | Delta so với baseline gốc 88.02 | Delta so với baseline tự train 87.19 |
|---|---:|---:|---:|
| Keyword mixed top-1 | 86.45 | -1.57 | -0.74 |
| Weighted keyword 5x | 87.15 | -0.87 | -0.04 |
| Visual-keyword 5x, best AUC | 86.80 | -1.22 | -0.39 |
| Visual-keyword 5x, best mAP | 86.59 | -1.43 | -0.60 |

Kết quả cho thấy `weighted keyword 5x` là cấu hình giữ classifier AUC tốt nhất trong nhóm keyword prompt. AUC `87.15` vẫn thấp hơn baseline gốc `88.02`, nhưng gần như ngang với baseline tự train `87.19`, chỉ thấp hơn khoảng `0.04` điểm.

Ngược lại, `keyword mixed top-1` kém hơn rõ rệt. Điều này phù hợp với các thí nghiệm trước: chọn duy nhất một prompt cho mỗi class làm mất semantic diversity, đặc biệt với các class có biểu hiện thị giác đa dạng.

### 4.2. Weighted Keyword 5x

Đây là thí nghiệm dùng bộ keyword prompt 5x đầu tiên. Mỗi class có 5 cụm keyword, và 5 cụm này được gán trọng số theo công thức text-space. Đây là thí nghiệm tốt nhất trong nhóm keyword prompt nếu ưu tiên classifier AUC. Kết quả tốt nhất đạt ở epoch 3:

| Epoch | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 84.98 | 28.75 | 83.45 | 21.24 | 63.92 | 6.17 |
| 2 | 86.45 | 30.98 | 84.47 | 23.48 | 67.12 | 8.14 |
| 3 | 87.15 | 30.03 | 85.34 | 24.98 | 68.19 | 9.14 |
| 4 | 86.53 | 28.41 | 85.51 | 27.23 | 66.60 | 10.73 |
| 5 | 86.66 | 28.46 | 85.39 | 26.68 | 66.94 | 10.51 |
| 6 | 86.74 | 30.39 | 85.21 | 26.35 | 67.21 | 9.71 |

Có một trade-off khá rõ:

- Epoch 3 có `classifier AUC` cao nhất: `87.15`.
- Epoch 4 có `avg mAP` cao nhất: `10.73`.

Điều này cho thấy khi train lâu hơn, model có xu hướng cải thiện khả năng localization theo mAP, nhưng classifier AUC và Ano-AUC lại giảm. Nói cách khác, prompt keyword giúp model tập trung tốt hơn vào một số đoạn bất thường rõ ràng, nhưng nếu tối ưu quá lâu thì phân phối score toàn cục trở nên kém ổn định hơn.

So với baseline gốc:

| Metric | Baseline gốc | Weighted keyword 5x | Delta |
|---|---:|---:|---:|
| classifier AUC | 88.02 | 87.15 | -0.87 |
| classifier AP | 33.56 | 30.03 | -3.53 |
| alignment AUC | 85.69 | 85.34 | -0.35 |
| alignment AP | 26.50 | 24.98 | -1.52 |
| Ano-AUC | 70.23 | 68.19 | -2.04 |
| avg mAP | 6.68 | 9.14 | +2.46 |

So với baseline tự train, classifier AUC gần như ngang bằng:

```text
87.15 so với 87.19 -> -0.04
```

Đây là tín hiệu tích cực nhất của nhóm keyword prompt: tuy chưa vượt baseline gốc, cấu hình này gần đạt baseline tự train trong khi tăng `avg mAP` đáng kể.

### 4.3. Visual-Keyword 5x Với Công Thức Tách Class Mới

Thí nghiệm `visual-keyword 5x` dùng bộ prompt ngắn hơn, trực tiếp hơn và gần với tín hiệu frame hơn. Đồng thời, công thức scoring mới được dùng để giảm chồng lấn giữa các class dễ lẫn. Kết quả có hai điểm cần tách riêng:

- Nếu chọn checkpoint theo `classifier AUC`, kết quả tốt nhất khoảng `86.80`.
- Nếu chọn checkpoint theo `avg mAP`, kết quả tốt nhất là `10.24` ở epoch 2.

Diễn biến chính:

| Epoch | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 86.18 | 31.03 | 84.65 | 23.67 | 66.45 | 6.13 |
| 2 | 86.59 | 29.57 | 85.79 | 24.34 | 67.78 | 10.24 |
| 3 | 86.37 | 29.95 | 84.22 | 23.39 | 67.28 | 8.39 |
| 7 | 86.80 | 30.21 | 84.32 | 23.78 | 67.71 | 8.46 |

So với baseline gốc ở checkpoint best AUC:

| Metric | Baseline gốc | Visual-keyword best AUC | Delta |
|---|---:|---:|---:|
| classifier AUC | 88.02 | 86.80 | -1.22 |
| classifier AP | 33.56 | 30.21 | -3.35 |
| alignment AUC | 85.69 | 84.32 | -1.37 |
| alignment AP | 26.50 | 23.78 | -2.72 |
| Ano-AUC | 70.23 | 67.71 | -2.52 |
| avg mAP | 6.68 | 8.46 | +1.78 |

So với baseline tự train:

```text
86.80 so với 87.19 -> -0.39
```

Nếu xét theo mAP, visual-keyword có thể đạt `avg mAP = 10.24`, cao hơn baseline gốc `+3.56`. Tuy nhiên classifier AUC ở checkpoint này chỉ là `86.59`, thấp hơn baseline tự train `-0.60`.

Điều này cho thấy công thức mới giúp localization khá tốt, nhưng chưa giải quyết được bài toán chính là giữ AUC. Việc giảm prompt xuống dạng visual keyword quá ngắn có thể làm mất một phần semantic anchor cần thiết cho phân biệt bình thường/bất thường.

### 4.4. Nhận Xét Theo Class

Với `weighted keyword 5x`, một số class tăng mạnh ở classifier AUC:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Assault | 84.34 | 94.78 | +10.44 |
| Vandalism | 70.97 | 81.31 | +10.34 |
| Robbery | 73.30 | 77.46 | +4.16 |
| Fighting | 67.90 | 69.93 | +2.04 |

Các class giảm mạnh:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Explosion | 59.18 | 47.28 | -11.89 |
| Shoplifting | 64.37 | 62.56 | -1.81 |
| Shooting | 65.63 | 65.09 | -0.55 |
| Arson | 76.85 | 76.47 | -0.38 |

`Explosion` tiếp tục là class yếu nhất. Dù prompt đã được rút gọn thành keyword như `rupture explosion debris`, `detonation fireball smoke`, `blast shockwave debris`, class này vẫn bị giảm mạnh. Nguyên nhân có thể là các dấu hiệu của `Explosion` như lửa, khói, mảnh vỡ, phá hủy và va chạm vẫn chồng lấn với `Arson` và `RoadAccidents`.

Với visual-keyword 5x, một số class vẫn tăng:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Vandalism | 70.97 | 76.80 | +5.83 |
| Assault | 84.34 | 88.34 | +4.00 |
| Fighting | 67.90 | 69.93 | +2.03 |
| Abuse | 56.98 | 58.98 | +1.99 |

Nhưng các class giảm đáng chú ý:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Explosion | 59.18 | 49.86 | -9.31 |
| Arrest | 68.57 | 65.40 | -3.17 |
| Shooting | 65.63 | 63.40 | -2.23 |
| Burglary | 70.44 | 69.07 | -1.37 |
| Arson | 76.85 | 75.65 | -1.20 |

So với weighted keyword 5x, visual-keyword giảm ít hơn ở `Explosion` nhưng lại làm nhiều metric tổng thể thấp hơn. Điều này cho thấy chỉ sửa prompt của một vài class chưa đủ; vấn đề còn nằm ở cách các prompt ảnh hưởng đến phân phối score toàn cục.

## 5. Nhận Xét Kết Quả

Keyword prompt là hướng tốt hơn so với một số prompt dài hoặc prototype dài trước đó, đặc biệt khi dùng theo kiểu weighted multi-prompt. Kết quả `weighted keyword 5x` đạt `classifier AUC = 87.15`, gần bằng baseline tự train `87.19`, đồng thời tăng `avg mAP` từ `6.68` lên `9.14`.

Điều này cho thấy keyword prompt có tác dụng nhất định:

- giảm bớt nhiễu từ câu mô tả dài;
- giữ được nhiều biến thể semantic nhờ 5 prompt/class;
- giúp localization tốt hơn qua mAP;
- ít làm tụt AUC hơn so với top-1 prompt hoặc nhiều prototype dài.

Tuy nhiên, keyword prompt vẫn chưa vượt baseline gốc `88.02`. Nguyên nhân chính là prompt chỉ tác động trực tiếp đến class/text alignment, trong khi `classifier AUC` phụ thuộc mạnh vào khả năng phân tách frame bất thường và frame bình thường. Nếu prompt làm model tập trung tốt hơn vào một số pattern bất thường nhưng đồng thời làm score nền tăng hoặc làm score anomaly không ổn định, AUC vẫn giảm.

Thí nghiệm cũng cho thấy không nên lọc prompt quá mạnh. `Keyword mixed top-1` chỉ đạt `classifier AUC = 86.45`, thấp hơn weighted keyword 5x. Một prompt duy nhất không đủ bao phủ các class phức tạp như `Explosion`, `Arson`, `Robbery`, `Assault`.

Trong nhóm keyword prompt hiện tại, cấu hình đáng giữ lại nhất là:

```text
Weighted keyword 5x
class name + keyword prompt
train từ scratch
chọn checkpoint theo classifier AUC
```

Cấu hình này chưa vượt baseline gốc, nhưng là một trong những hướng prompt-based ổn định nhất vì giữ được AUC gần baseline tự train và cải thiện mAP rõ ràng.

# Nhật Ký Thí Nghiệm: Weighted Keyword Prompt Kết Hợp Abnormal Background Separation

## 1. Mục Tiêu

Sau các thí nghiệm keyword prompt, một vấn đề lặp lại là mAP thường tăng nhưng classifier AUC lại giảm. Điều này cho thấy model có thể phát hiện tốt hơn một số đoạn bất thường rõ ràng, nhưng phân phối score frame-level trên toàn bộ video chưa đủ ổn định. Một số frame nền hoặc frame bình thường trong video bất thường vẫn có thể bị đẩy score lên cao, làm giảm khả năng phân tách abnormal frame và normal/background frame.

Thí nghiệm này giữ lại cấu hình `weighted keyword 5x`, vì đây là cấu hình keyword prompt ổn định nhất trước đó, rồi bổ sung thêm một ràng buộc nhỏ để tách nhóm score cao và nhóm score thấp trong video bất thường. Mục tiêu không phải thay đổi kiến trúc VadCLIP, mà là điều chỉnh cách train để model học rõ hơn sự khác biệt giữa đoạn bất thường nổi bật và đoạn nền trong cùng một video.

Hai mốc so sánh chính vẫn được giữ nguyên:

| Mốc so sánh | Ý nghĩa | classifier AUC |
|---|---|---:|
| Baseline gốc VadCLIP checkpoint | checkpoint công bố/đã train sẵn trên UCF-Crime | 88.02 |
| Baseline tự train | VadCLIP train lại từ scratch bằng class prompt gốc | 87.19 |

## 2. Cách Làm

Thí nghiệm vẫn sử dụng bộ prompt `weighted keyword 5x`. Mỗi class có 5 prompt ngắn dạng keyword, các prompt được gán trọng số dựa trên mức độ phù hợp semantic với class và mức độ tách khỏi các class khác. Khi đưa vào model, các prompt này đóng vai trò thay thế class text embedding trong nhánh alignment của VadCLIP.

Điểm mới của thí nghiệm này là thêm `abnormal background separation`. Ý tưởng có thể hiểu đơn giản như sau:

```text
Trong một video bất thường,
model không chỉ cần biết video đó chứa bất thường,
mà còn cần làm cho các đoạn bất thường rõ ràng có score cao hơn các đoạn nền.
```

Vì UCF-Crime chỉ có nhãn yếu ở cấp video và không có frame label đầy đủ cho training, không thể trực tiếp chỉ định frame nào là bất thường. Thay vào đó, model dùng chính anomaly score hiện tại để chia tương đối trong mỗi video bất thường:

- nhóm score cao được xem như các đoạn nghi ngờ bất thường;
- nhóm score thấp được xem như các đoạn nền hoặc ít liên quan hơn;
- loss mới khuyến khích khoảng cách giữa hai nhóm này đủ rõ.

Ràng buộc này không thay thế loss gốc của VadCLIP. Nó chỉ là một thành phần phụ, nhằm giảm hiện tượng score bị nâng đều trên cả video bất thường.

## 3. Cấu Hình Training

Training vẫn chạy từ scratch, thống nhất với các thí nghiệm gần đây. Cấu hình chính:

| Thành phần | Cấu hình |
|---|---|
| Prompt | weighted keyword 5x |
| Kiểu prompt | class name + keyword prompt |
| Số prompt mỗi class | 5 |
| Cách dùng prompt | multi-prompt có trọng số |
| Khởi tạo model | train từ scratch |
| Số epoch | 5 |
| Learning rate | 2e-5 |
| Normal regularization | tắt |
| Ranking regularization | tắt |
| Abnormal background separation | bật |

Cấu hình riêng của abnormal background separation:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| lambda_abnormal_bg | 0.05 | trọng số của loss tách nền trong video bất thường |
| abnormal_bg_margin | 0.15 | khoảng cách tối thiểu mong muốn giữa nhóm score cao và score thấp |
| abnormal_bg_warmup_epochs | 1 | chưa áp dụng loss này ở epoch đầu để model ổn định trước |

Hàm loss tổng của thí nghiệm này gồm loss gốc của VadCLIP, consistency loss của multi-prompt và regularization mới cho abnormal background:

```text
L_total =
  L_cls
+ L_mil
+ L_text_reg
+ lambda_consistency * L_consistency
+ lambda_abnormal_bg(epoch) * L_abnormal_bg
```

Trong đó:

- `L_cls` là loss phân loại abnormal/normal của nhánh classifier, tương ứng với `logits1`.
- `L_mil` là multiple-instance learning loss của nhánh alignment giữa visual feature và class/prompt text embedding, tương ứng với `logits2`.
- `L_text_reg` là regularization gốc của VadCLIP trên text embedding, giúp các embedding class/prompt không bị co cụm quá gần nhau.
- `L_consistency` là consistency loss của multi-prompt, giúp các prompt khác nhau trong cùng một class không tạo ra tín hiệu quá lệch nhau trên các đoạn được model xem là quan trọng.
- `L_abnormal_bg` là loss mới trong thí nghiệm này, dùng để tách đoạn bất thường nổi bật khỏi đoạn nền trong video bất thường.

Các regularizer khác đã được tắt trong thí nghiệm này:

| Regularizer | Trọng số |
|---|---:|
| normal top-k suppression | 0.0 |
| abnormal background separation | 0.05 |
| ranking margin giữa normal và abnormal video | 0.0 |

### 3.1. Công Thức Abnormal Background Separation

Với mỗi video bất thường, model tạo anomaly score theo hai nhánh:

```text
s_cls(t) = sigmoid(logits1(t))
s_align(t) = 1 - softmax(logits2(t))[Normal]
```

`s_cls(t)` là score từ nhánh classifier. `s_align(t)` là score từ nhánh alignment, được hiểu là mức độ frame không giống class `Normal`. Vì cấu hình hiện tại dùng `regularize_score_branch = both`, loss được tính trên cả hai loại score rồi lấy trung bình.

Với một video bất thường có `T` snippet hợp lệ, chọn:

```text
k = max(1, floor(T * top_k_ratio))
```

Trong thí nghiệm này:

```text
top_k_ratio = 0.15
```

Tức là model lấy khoảng 15% snippet có score cao nhất và 15% snippet có score thấp nhất trong cùng video bất thường:

```text
H = top-k snippets có score cao nhất
L = top-k snippets có score thấp nhất
```

Sau đó regularizer yêu cầu nhóm score cao phải tách khỏi nhóm score thấp ít nhất một khoảng `margin`:

```text
min(score(H)) >= max(score(L)) + margin
```

Viết dưới dạng hinge loss:

```text
L_abnormal_bg =
  mean over abnormal videos and score branches [
    max(0, max(score(L)) - min(score(H)) + margin)
  ]
```

Trong thí nghiệm này:

```text
margin = 0.15
lambda_abnormal_bg = 0.05
```

Nói đơn giản, nếu đoạn có score thấp nhất trong nhóm bất thường cao chưa đủ xa so với đoạn có score cao nhất trong nhóm nền, model sẽ bị phạt. Nếu hai nhóm đã cách nhau ít nhất `0.15`, loss này bằng 0.

Một chi tiết quan trọng là các snippet thuộc nhóm `H` và `L` được chọn bằng score hiện tại nhưng chỉ dùng chỉ số lựa chọn; còn giá trị loss vẫn tính trên score gốc của model. Vì vậy, regularizer vẫn truyền gradient để cập nhật model, nhưng việc chọn nhóm high/low ổn định hơn trong từng bước train.

### 3.2. Warm-Up Cho Regularizer

Regularizer này không được bật ngay từ epoch đầu. Trọng số thực tế được dùng như sau:

```text
lambda_abnormal_bg(epoch) =
  0.0   nếu epoch < abnormal_bg_warmup_epochs
  0.05  nếu epoch >= abnormal_bg_warmup_epochs
```

Với:

```text
abnormal_bg_warmup_epochs = 1
```

Lý do là ở epoch đầu, score của model còn chưa ổn định. Nếu ép tách top-k và bottom-k quá sớm, model có thể học theo các score nhiễu ban đầu. Warm-up cho phép model học các loss chính trước, sau đó mới dùng regularizer để làm rõ phân phối score trong video bất thường.

## 4. Kết Quả Sau Fine-Tune

Do folder kết quả hiện tại chỉ có các file score dạng `.npz`, phần này kết luận chắc chắn được các chỉ số frame-level `classifier AUC/AP` và `alignment AUC/AP`. Các chỉ số `Ano-AUC` và `mAP` cần được bổ sung sau khi có đầy đủ kết quả evaluate/analyze.

Diễn biến theo epoch:

| Epoch | classifier AUC | classifier AP | alignment AUC | alignment AP |
|---|---:|---:|---:|---:|
| 1 | 86.05 | 28.85 | 83.78 | 22.25 |
| 2 | 86.80 | 31.99 | 84.87 | 24.24 |
| 3 | 87.14 | 29.28 | 85.38 | 25.58 |
| 4 | **87.52** | **32.70** | **85.66** | **26.69** |
| 5 | 87.50 | 32.40 | 85.51 | 26.66 |

Checkpoint tốt nhất theo classifier AUC là epoch 4.

So sánh với các mốc chính:

| Mô hình | classifier AUC | classifier AP | alignment AUC | alignment AP |
|---|---:|---:|---:|---:|
| Baseline gốc VadCLIP | 88.02 | 33.56 | 85.69 | 26.50 |
| Baseline tự train | 87.19 | chưa tổng hợp | chưa tổng hợp | chưa tổng hợp |
| Weighted keyword 5x | 87.15 | 30.03 | 85.34 | 24.98 |
| Weighted keyword 5x + abnormal background separation | **87.52** | **32.70** | **85.66** | **26.69** |

So với baseline gốc VadCLIP:

| Metric | Baseline gốc | Kết quả thí nghiệm | Delta |
|---|---:|---:|---:|
| classifier AUC | 88.02 | 87.52 | -0.50 |
| classifier AP | 33.56 | 32.70 | -0.86 |
| alignment AUC | 85.69 | 85.66 | -0.03 |
| alignment AP | 26.50 | 26.69 | +0.19 |

So với baseline tự train:

```text
classifier AUC: 87.52 so với 87.19 -> +0.33
```

So với `weighted keyword 5x` không dùng abnormal background separation:

| Metric | Weighted keyword 5x | Thêm abnormal background separation | Delta |
|---|---:|---:|---:|
| classifier AUC | 87.15 | 87.52 | +0.37 |
| classifier AP | 30.03 | 32.70 | +2.67 |
| alignment AUC | 85.34 | 85.66 | +0.32 |
| alignment AP | 24.98 | 26.69 | +1.71 |

## 5. Nhận Xét Kết Quả

Đây là kết quả tích cực nhất trong nhóm keyword prompt tính đến thời điểm hiện tại nếu xét theo classifier AUC. Khác với nhiều thí nghiệm trước, biến thể này không chỉ giúp localization mà còn cải thiện trực tiếp AUC/AP so với cấu hình weighted keyword 5x ban đầu.

Điểm đáng chú ý nhất là `classifier AUC = 87.52`, cao hơn baseline tự train `87.19`. Điều này cho thấy keyword prompt vẫn có thể đem lại lợi ích cho VadCLIP nếu đi kèm một cơ chế giúp kiểm soát score nền trong video bất thường. Nói cách khác, vấn đề không chỉ nằm ở prompt tốt hay xấu, mà còn nằm ở cách model phân phối anomaly score trên timeline.

Tuy vậy, kết quả vẫn chưa vượt baseline gốc `88.02`. Khoảng cách còn lại là `0.50` điểm AUC. Điều này cho thấy phương pháp hiện tại đã đi đúng hướng hơn so với các thí nghiệm prompt trước, nhưng vẫn chưa đủ để thay thế hoàn toàn cấu hình gốc đã được train mạnh.

Một điểm cần cẩn thận khi kết luận là folder kết quả hiện chưa có đầy đủ `Ano-AUC` và `mAP`. Vì vậy, thí nghiệm này hiện chỉ chứng minh được rằng abnormal background separation giúp cải thiện frame-level AUC/AP. Để kết luận đầy đủ hơn, cần chạy hoặc lưu lại đầy đủ bộ evaluate/analyze để xem liệu cải thiện AUC có đi kèm giữ được mAP hay không.
