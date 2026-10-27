# Nhật ký thí nghiệm: VadCLIP với subclass xây dựng từ video description

## 1. Mục tiêu

VadCLIP gốc biểu diễn mỗi class bằng một tên duy nhất, chẳng hạn `Explosion`, `Fighting` hoặc `Robbery`. Trong thực tế, cùng một class có thể xuất hiện dưới nhiều hình thức thị giác khác nhau. Ví dụ, một vụ nổ có thể liên quan đến phương tiện, công trình hoặc khu vực ven đường.

Thí nghiệm này chia mỗi class abnormal thành một số subclass ngắn. Mục tiêu là cung cấp nhiều khái niệm text cụ thể hơn cho nhánh alignment, nhưng vẫn chỉ dùng nhãn class gốc ở cấp video. Không có nhãn subclass và không sử dụng description trong inference.

## 2. Cách xây dựng subclass

### 2.1. Nguồn thông tin

- Subclass được xây dựng sau khi rà soát description của các video thuộc train split.
- Có 1.162 video train có description hợp lệ, gồm 574 video abnormal và 588 video Normal.
- Description của test split không được dùng để xây taxonomy, nhằm tránh test leakage.
- Description chỉ dùng để nhận diện các tình huống lặp lại trong từng class, không được dùng làm nhãn frame hoặc nhãn subclass.

### 2.2. Nguyên tắc đặt tên

Mỗi subclass abnormal gồm đúng hai thành phần:

```text
<Tên class gốc> <Từ mô tả>
```

Ví dụ:

```text
Explosion Vehicle
Fighting Group
Stealing Bicycle
```

Các nguyên tắc lựa chọn từ mô tả:

- Luôn giữ tên class gốc làm semantic anchor.
- Ưu tiên tín hiệu lớn và dễ quan sát trong frame, như số người, phương tiện, tư thế, bối cảnh và kiểu tương tác.
- Hạn chế vật thể nhỏ như ví, điện thoại, bật lửa hoặc còng tay vì feature cấp snippet có thể không giữ đủ chi tiết.
- Không dùng câu mô tả diễn biến dài vì CLIP Text Encoder chủ yếu phù hợp với caption thị giác ngắn.
- Số subclass không bắt buộc giống nhau. Class đa dạng có nhiều subclass hơn; class ít dữ liệu có ít subclass hơn.
- Dùng `Other` cho các trường hợp vẫn thuộc class gốc nhưng không phù hợp rõ với những nhóm cụ thể.
- Không chia subclass cho Normal; class này giữ một prompt `Normal`.

### 2.3. Bộ subclass sử dụng

| Class | Subclass | Lý do phân chia |
|---|---|---|
| Normal | `Normal` | Không chia nhỏ Normal để tập trung mở rộng biểu diễn cho các class abnormal. |
| Abuse | `Abuse Child`, `Abuse Wheelchair`, `Abuse Custodial`, `Abuse Group`, `Abuse Grounded`, `Abuse Other` | Phân biệt theo đối tượng dễ tổn thương, môi trường có người giám sát, số người tham gia và trạng thái nạn nhân. |
| Arrest | `Arrest Grounded`, `Arrest Escorted`, `Arrest Vehicle`, `Arrest Group`, `Arrest Other` | Dùng các trạng thái khống chế, dẫn đi, tương tác gần phương tiện và hoạt động theo nhóm. Không dùng còng tay vì quá nhỏ. |
| Arson | `Arson Vehicle`, `Arson Building`, `Arson Vegetation`, `Arson Outdoor`, `Arson Other` | Phân chia theo đối tượng hoặc môi trường bị đốt, thay vì dụng cụ gây cháy nhỏ. |
| Assault | `Assault Individual`, `Assault Group`, `Assault Armed`, `Assault Grounded`, `Assault Other` | Phân biệt số người tấn công, tình trạng nạn nhân và trường hợp có công cụ tấn công đủ rõ. |
| Burglary | `Burglary Door`, `Burglary Window`, `Burglary Vehicle`, `Burglary ATM`, `Burglary Interior`, `Burglary Other` | Các phương thức và vị trí đột nhập này xuất hiện lặp lại rõ trong description. |
| Explosion | `Explosion Vehicle`, `Explosion Building`, `Explosion Roadside`, `Explosion Other` | Chỉ giữ các bối cảnh lớn vì class này có ít description train. Không chia theo khói hoặc lửa để giảm chồng lấn với Arson. |
| Fighting | `Fighting Pair`, `Fighting Group`, `Fighting Grappling`, `Fighting Grounded`, `Fighting Other` | Số người và kiểu tương tác cơ thể là các tín hiệu thị giác rõ hơn loại vũ khí. |
| RoadAccidents | `RoadAccidents Pedestrian`, `RoadAccidents Motorcycle`, `RoadAccidents Bicycle`, `RoadAccidents Multivehicle`, `RoadAccidents Truck`, `RoadAccidents Bus`, `RoadAccidents Rollover`, `RoadAccidents Barrier`, `RoadAccidents Other` | Đây là class đa dạng nhất, được chia theo tác nhân giao thông và hình thức va chạm dễ quan sát. |
| Robbery | `Robbery Armed`, `Robbery Counter`, `Robbery Street`, `Robbery Vehicle`, `Robbery Bag`, `Robbery Group`, `Robbery Other` | Phân biệt theo hình thức đe dọa, bối cảnh, tài sản lớn và số người tham gia. `Armed` được dùng thay cho loại vũ khí cụ thể. |
| Shooting | `Shooting Indoor`, `Shooting Outdoor`, `Shooting Vehicle`, `Shooting Crowd`, `Shooting Police`, `Shooting Other` | Dựa chủ yếu vào bối cảnh lớn thay vì loại súng, vì súng thường chiếm diện tích nhỏ trong frame. |
| Shoplifting | `Shoplifting Bag`, `Shoplifting Clothing`, `Shoplifting Shelf`, `Shoplifting Counter`, `Shoplifting Group`, `Shoplifting Other` | Phân biệt cách che giấu, vị trí thao tác hàng hóa và số người; không chia theo loại sản phẩm nhỏ. |
| Stealing | `Stealing Vehicle`, `Stealing Motorcycle`, `Stealing Bicycle`, `Stealing Bag`, `Stealing Cargo`, `Stealing Indoor`, `Stealing Other` | Phân biệt phương tiện, tài sản mang theo, hàng hóa lớn và hành vi trong nhà. |
| Vandalism | `Vandalism Vehicle`, `Vandalism Glass`, `Vandalism Graffiti`, `Vandalism Building`, `Vandalism Structure`, `Vandalism Group`, `Vandalism Other` | Phân chia theo đối tượng bị phá hoại và số người tham gia. |

Bộ dữ liệu text cuối cùng có 78 subclass abnormal. Số subclass của một class biến thiên từ 4 đến 9; Normal có một prompt duy nhất.

## 3. Cách đưa subclass vào VadCLIP

Kiến trúc visual, nhánh classifier và các loss chính của VadCLIP được giữ nguyên. CLIP Text Encoder vẫn đóng băng, còn shared learnable prompt tiếp tục được cập nhật trong quá trình training.

Thay vì encode một class name, model encode riêng các subclass của class đó. Tại mỗi snippet, các subclass được gộp bằng Max hoặc Soft để quay lại 14 class score. Temporal Top-K sau đó hoạt động trên 14 class giống VadCLIP gốc.

- **Max aggregation:** tại mỗi snippet, chỉ lấy score cao nhất trong các subclass của cùng một class. Ví dụ, score của `Explosion` là score lớn nhất giữa `Explosion Vehicle`, `Explosion Building`, `Explosion Roadside` và `Explosion Other`.
- **Soft aggregation:** tại mỗi snippet, kết hợp score của tất cả subclass trong cùng class. Subclass có score cao đóng góp nhiều hơn nhưng các subclass còn lại vẫn tham gia và nhận gradient.

Mỗi cách aggregate được huấn luyện theo hai cách khởi tạo:

- **From scratch:** khởi tạo lại các module cần học của VadCLIP và huấn luyện với subclass ngay từ đầu; CLIP pretrained vẫn đóng băng.
- **Từ baseline:** nạp checkpoint VadCLIP đã được nhóm tự train bằng class name gốc, sau đó tiếp tục fine-tune bằng subclass.

Do đó thí nghiệm có bốn cấu hình: `Scratch-Max`, `Scratch-Soft`, `Checkpoint-Max` và `Checkpoint-Soft`.

```text
Visual snippets
→ subclass alignment
→ Max hoặc Soft theo subclass tại từng snippet
→ 14 class scores
→ Temporal Top-K
→ MIL class loss
```

Model vẫn chỉ dùng nhãn class gốc ở cấp video; không cần nhãn subclass và không dùng description khi inference.

## 4. Hàm loss

Không thêm subclass loss hoặc description loss riêng. Tổng loss vẫn gồm ba thành phần chính:

```text
L_total = L_classifier + L_MIL_class + L_text_regularization
```

- `L_classifier`: Binary Cross-Entropy trên video-level abnormal score của nhánh classifier.
- `L_MIL_class`: Cross-Entropy giữa 14 video-level class score và nhãn class gốc.
- `L_text_regularization`: giữ text embedding đại diện của Normal tách khỏi các class abnormal.

Text embedding đại diện cho một class được tính từ các subclass thật của class đó và chỉ dùng trong `L_text_regularization`; centroid này không được dùng để tạo alignment map.

## 5. Cấu hình huấn luyện

### 5.1. Cấu hình chung

| Thành phần | Giá trị |
|---|---|
| Train set | Toàn bộ 16.100 chunks |
| Test set | 290 video |
| Batch | 64 normal + 64 abnormal |
| Optimizer | AdamW |
| Seed | `234` |
| Mixed precision | Không sử dụng |
| Soft temperature | `1.0` |
| Checkpoint selection | Dùng epoch cố định, không chọn bằng test AUC |

### 5.2. Bốn lượt chạy

| Cấu hình | Khởi tạo | Aggregate | Epoch | Learning rate | Scheduler |
|---|---|---|---:|---:|---|
| Scratch-Max | Train từ đầu | Max | 10 | `2e-5` | Giảm 10 lần sau epoch 4 và 8 |
| Scratch-Soft | Train từ đầu | Soft | 10 | `2e-5` | Giảm 10 lần sau epoch 4 và 8 |
| Checkpoint-Max | Baseline tự train | Max | 3 | `2e-5` | Giảm 10 lần sau epoch 2 |
| Checkpoint-Soft | Baseline tự train | Soft | 3 | `2e-5` | Giảm 10 lần sau epoch 2 |

Nhánh checkpoint sử dụng baseline tự train, không sử dụng checkpoint công bố của tác giả.

## 6. Kết quả

### 6.1. Baseline dùng trong cùng lượt đánh giá

| Model | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | AVG mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline cùng evaluator | 87.72 | 33.13 | 86.52 | 28.23 | 69.94 | 7.57 |

Mốc baseline tự train được sử dụng trong các báo cáo chung trước đây có AUC1 `87.44`. Tuy nhiên, checkpoint baseline được evaluator đọc trong lượt chạy này cho AUC1 `87.72`. Các delta dưới đây được tính với kết quả `87.72` để bảo đảm so sánh trực tiếp trong cùng evaluator và cùng bộ file hiện tại.

### 6.2. Kết quả bốn cấu hình

| Cấu hình | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | AVG mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 87.72 | 33.13 | **86.52** | **28.23** | **69.94** | 14.12 | 9.47 | 6.63 | 4.27 | 3.34 | 7.57 |
| Scratch-Max | 86.60 | 30.90 | 85.09 | 25.52 | 67.43 | **15.77** | 10.20 | 6.76 | 4.09 | 1.66 | 7.70 |
| **Scratch-Soft** | **87.78** | **35.11** | 85.37 | 25.60 | 69.88 | 15.05 | 9.97 | 6.00 | 4.03 | 2.56 | 7.52 |
| Checkpoint-Max | 86.58 | 31.78 | 85.22 | 26.80 | 67.39 | 13.57 | **10.38** | 5.79 | 3.85 | 3.05 | 7.33 |
| **Checkpoint-Soft** | 86.57 | 30.56 | 84.99 | 26.16 | 67.53 | 13.86 | 10.35 | **7.22** | **4.53** | **3.79** | **7.95** |

### 6.3. Chênh lệch so với baseline cùng evaluator

| Cấu hình | Δ AUC1 | Δ AP1 | Δ AUC2 | Δ AP2 | Δ Ano-AUC | Δ AVG mAP |
|---|---:|---:|---:|---:|---:|---:|
| Scratch-Max | -1.12 | -2.24 | -1.43 | -2.71 | -2.51 | +0.13 |
| **Scratch-Soft** | **+0.06** | **+1.98** | -1.15 | -2.64 | **-0.06** | -0.05 |
| Checkpoint-Max | -1.14 | -1.35 | -1.30 | -1.43 | -2.55 | -0.24 |
| Checkpoint-Soft | -1.15 | -2.57 | -1.53 | -2.07 | -2.41 | **+0.38** |

## 7. Phân tích kết quả

### 7.1. Scratch-Soft giữ khả năng phát hiện abnormal tốt nhất

Scratch-Soft là cấu hình duy nhất vượt baseline cùng evaluator về AUC1, từ `87.72` lên `87.78`. AP1 tăng từ `33.13` lên `35.11`, trong khi Ano-AUC gần như giữ nguyên (`69.88` so với `69.94`).

Kết quả này cho thấy Soft aggregation trước Temporal Top-K giúp nhánh classifier học được tín hiệu abnormal ổn định hơn Max. Nhiều subclass có thể cùng đóng góp tại một snippet thay vì chỉ một subclass nhận gradient.

Tuy nhiên, AUC2 và AP2 vẫn giảm. Subclass giúp classifier nhưng chưa tạo được không gian alignment text-visual tốt hơn class name gốc.

### 7.2. Checkpoint-Soft có AVG mAP cao nhất

Checkpoint-Soft đạt AVG mAP `7.95`, cao hơn baseline `0.38` điểm. Cải thiện tập trung ở các ngưỡng IoU từ 0.2 đến 0.5:

| Metric | Baseline | Checkpoint-Soft | Chênh lệch |
|---|---:|---:|---:|
| mAP@0.2 | 9.47 | 10.35 | +0.88 |
| mAP@0.3 | 6.63 | 7.22 | +0.59 |
| mAP@0.4 | 4.27 | 4.53 | +0.26 |
| mAP@0.5 | 3.34 | 3.79 | +0.44 |

Đổi lại, AUC1 và Ano-AUC giảm hơn 1 và 2 điểm. Cấu hình này cải thiện localization/classification theo event nhưng làm yếu khả năng phân biệt normal và abnormal trên toàn bộ frame.

### 7.3. Max aggregation vẫn bị subclass collapse

Trong Scratch-Max, một subclass chi phối phần lớn frame của nhiều class:

- `Burglary ATM`: khoảng 99,8%.
- `Stealing Motorcycle`: gần 100%.
- `RoadAccidents Multivehicle`: khoảng 95%.
- `Shooting Police`: khoảng 76%.
- `Shoplifting Group`: khoảng 80%.

Trong Checkpoint-Max, các subclass `Other` thường chi phối:

- `Vandalism Other`: 100%.
- `Stealing Other`: khoảng 97,7%.
- `Abuse Other`: khoảng 94,9%.
- `Arson Other`: khoảng 90,7%.
- `Explosion Other`: khoảng 60%.

Điều này cho thấy Max dễ chọn một prompt thuận lợi làm đường tắt cho cả class. Soft phân phối đóng góp giữa các subclass tốt hơn, dù chưa cải thiện được alignment AUC/AP.

## 8. Kết luận

- Thứ tự aggregate subclass trước rồi mới Temporal Top-K phù hợp hơn với cơ chế MIL của VadCLIP gốc.
- Scratch-Soft là cấu hình cân bằng nhất nếu ưu tiên AUC1, AP1 và Ano-AUC.
- Checkpoint-Soft đạt AVG mAP cao nhất nhưng đánh đổi đáng kể AUC1 và Ano-AUC.
- Không có cấu hình nào cải thiện AUC2 hoặc AP2; nhánh alignment vẫn là điểm hạn chế chính.
- Max aggregation tiếp tục gặp subclass collapse, đặc biệt với các prompt `Other` hoặc các subclass có semantic quá rộng.
- Kết quả mới chỉ được chạy với một seed, nên các chênh lệch nhỏ như `+0.06` AUC1 chưa đủ để khẳng định cải thiện ổn định về mặt thống kê.
