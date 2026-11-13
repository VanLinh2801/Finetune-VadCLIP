# Nhật ký thí nghiệm Temporal CNN trên VadCLIP

## 1. Kiến trúc, huấn luyện và cách tính score

### Sơ đồ mô hình

1. **CLIP visual features** `[B,T,512]` -> **LGT-Adapter** -> features `[B,T,512]`.
2. **Temporal CNN** thay classifier head gốc và tạo **classifier logits** `logits1 [B,T,1]`.
3. `logits1` cùng LGT features đi vào **text adjustment** của VadCLIP; class names, learnable prompt và CLIP text encoder vẫn giữ nguyên.
4. **Class alignment** tạo `logits2 [B,T,14]` như đường tính còn lại của VadCLIP. Vì dùng `logits1` ở bước 3, thay classifier có thể ảnh hưởng gián tiếp đến alignment.

### Cấu tạo Temporal CNN

1. **Input projection:** mỗi feature 512 chiều đi qua `Linear(512 -> P)`, LayerNorm và GELU.
2. **Conv block 1:** hai lớp Conv1D lần lượt chuyển `P -> C` và `C -> C`, mỗi lớp có GELU. Nếu bật residual, cộng thêm input của block (chiếu sang `C` chiều khi cần); sau đó dùng LayerNorm và mask padding.
3. **Conv block 2:** lặp cấu trúc hai lớp Conv1D `C -> C` và residual như block 1. Convolution chạy dọc trục thời gian với stride 1, padding giữ nguyên số snippet.
4. **Snippet scoring:** *scoring head* dùng `Linear(C -> 1)`; biến thể *kernel response* lấy phản hồi channel mạnh nhất rồi học scale và bias. Cả hai tạo **một logit cho mỗi snippet**.

Trong đó, `P` là số chiều sau projection, `C` là số output channels (số kernel) của CNN và `k` là kích thước kernel theo thời gian. Đây là các **biến thể kiến trúc**, được ghi trên từng dòng kết quả ở phần 2.

### Score và loss

- **Video score Top-K:** trong `T` snippet hợp lệ, chọn `K = max(1, ceil(T/16))` logit cao nhất rồi lấy trung bình. `K` viết hoa là số snippet được chọn, khác `k` (kernel size).
- **Video score Max:** chỉ lấy logit snippet cao nhất (`K=1`). Score video được đưa thẳng vào `BCEWithLogitsLoss` với nhãn Normal/abnormal cấp video.
- **Loss tổng:** `L = L_video_BCE + lambda_align × L_alignment_MIL + lambda_text × L_text_reg`. `L_alignment_MIL` dùng Top-K trên `logits2` để học lớp sự kiện; `L_text_reg` regularize class text embeddings. Thông thường `lambda_align = lambda_text = 1`; một lượt đối chứng đặt cả hai bằng 0. Đây **không phải** classifier loss của VadCLIP gốc.
- **Đánh giá frame-level:** AUC1/AP1 lấy `sigmoid(logits1)`; AUC2/AP2 lấy `1 - softmax(logits2)[Normal]`. Top-K/Max chỉ tạo score video để train, **không** thay thế score snippet khi tính AUC/AP. Snippet score được lặp tương ứng với 16 frame; mAP đánh giá nhận diện/định vị lớp sự kiện.

### Hyperparameters huấn luyện

| Tham số | Cấu hình thông thường |
|---|---|
| Khởi tạo | From scratch; CLIP pretrained frozen |
| Optimizer | AdamW |
| Learning rate ban đầu | `1e-5` |
| Batch size | 64 Normal + 64 abnormal |
| Seed | `234`, trừ lượt đổi seed ở phần 2 |
| Learning-rate scheduler | Nhân `0,1` sau epoch 4 và 8 |
| Số epoch | 10 |

Train dùng 8.000 Normal và 8.100 abnormal feature sequences (các crop/flip views); test dùng 290 video. Các lượt CNN là những lần train độc lập, không tiếp tục từ checkpoint của lượt trước. Baseline tự train được đánh giá bằng cùng evaluator. Test AUC không dùng để chọn checkpoint.

## 2. Kết quả giữ LGT-Adapter

**Chú giải bảng:** cấu hình `P/C/k` lần lượt là chiều projection / số channels CNN / kernel size. `H` = scoring head, `R` = kernel response; `T` = Top-K, `M` = Max. `B`/`Tắt` ở cột residual là bật/tắt residual. Mặc định dùng 2 Conv blocks, mỗi block 2 Conv1D. `Top-15` là tỷ lệ snippet có anomaly trong nhóm 15% score cao nhất của từng abnormal video, lấy trung bình theo video; dấu `-` nghĩa là không có thống kê. Các chỉ số AUC/AP/mAP ghi theo **điểm phần trăm**.

| Lượt chạy | P/C/k | Score/Agg | Residual | Số epoch chạy | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP | Top-15 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VadCLIP tự train (baseline) | - | Gốc | - | - | **87,72** | **33,13** | 86,52 | 28,23 | **69,94** | 7,57 | 35,29% |
| Kernel ban đầu | 128/128/1 | H/T | B | 10 | 86,26 | 28,96 | 86,10 | 28,27 | 65,96 | 6,91 | - |
| Kernel ban đầu | 128/128/3 | H/T | B | 10 | 85,38 | 25,81 | 84,63 | 24,18 | 64,24 | 7,09 | - |
| Kernel ban đầu | 128/128/5 | H/T | B | 10 | 86,94 | 30,99 | 85,93 | 25,58 | 67,83 | 7,05 | - |
| Kernel ban đầu | 128/128/7 | H/T | B | 10 | 85,39 | 25,00 | 85,26 | 26,45 | 63,85 | 7,14 | - |
| Kernel ban đầu | 128/128/9 | H/T | B | 10 | 86,45 | 26,73 | 85,79 | 26,78 | 66,80 | 8,36 | - |
| Kernel response | 128/128/3 | R/T | B | 10 | 85,43 | 27,36 | 85,96 | 26,58 | 66,21 | 9,11 | - |
| Kernel response | 128/128/5 | R/T | B | 10 | 86,75 | 32,27 | 86,17 | 29,30 | 68,16 | 7,93 | - |
| Max aggregation | 128/128/3 | H/M | B | 10 | 85,37 | 25,50 | 84,51 | 24,36 | 64,35 | 6,43 | - |
| Max; không loss phụ | 128/128/3 | H/M | B | 10 | 85,66 | 27,99 | N/A | N/A | 64,12 | N/A | - |
| Kernel, seed khác | 128/128/1 | H/T | B | 10 | 85,20 | 24,96 | 84,58 | 24,54 | 63,18 | 6,30 | - |
| Kernel, seed khác | 128/128/3 | H/T | B | 10 | 85,79 | 29,51 | 84,75 | 26,64 | 65,58 | 7,17 | - |
| Kernel, seed khác | 128/128/5 | H/T | B | 10 | 85,22 | 24,45 | 83,90 | 24,01 | 63,95 | 7,51 | - |
| Kernel, seed khác | 128/128/7 | H/T | B | 10 | 86,05 | 30,89 | 85,15 | 23,72 | 65,66 | 7,16 | - |
| Kernel, seed khác | 128/128/9 | H/T | B | 10 | 85,33 | 26,48 | 84,11 | 22,52 | 64,42 | 7,01 | - |
| Kernel, lặp seed 234 | 128/128/1 | H/T | B | 10 | 86,18 | 29,22 | 86,11 | 28,18 | 65,89 | 6,99 | - |
| Kernel, lặp seed 234 | 128/128/3 | H/T | B | 10 | 85,39 | 25,53 | 84,46 | 23,76 | 64,34 | 6,47 | - |
| Kernel, lặp seed 234 | 128/128/5 | H/T | B | 10 | 86,56 | 29,54 | 85,41 | 24,79 | 66,90 | 7,08 | - |
| Kernel, lặp seed 234 | 128/128/7 | H/T | B | 10 | 85,75 | 25,24 | 85,50 | 27,12 | 64,32 | 7,70 | - |
| Kernel, lặp seed 234 | 128/128/9 | H/T | B | 10 | 86,19 | 25,53 | 85,65 | 25,84 | 66,16 | 7,54 | - |
| Residual tắt, lượt riêng | 128/128/5 | H/T | Tắt | 10 | 86,28 | 26,93 | 86,91 | 29,02 | 66,92 | 7,26 | 34,17% |
| Residual tắt, lượt 15 epoch | 128/128/5 | H/T | Tắt | 10 | 85,84 | 26,66 | 86,23 | 27,70 | 65,57 | 6,61 | 35,10% |
| Residual tắt, cùng lượt | 128/128/5 | H/T | Tắt | 15 | 85,85 | 26,73 | 86,21 | 27,65 | 65,55 | 6,92 | 34,80% |
| Tăng projection | 256/128/5 | H/T | B | 10 | 85,81 | 25,10 | 85,17 | 25,35 | 65,04 | 8,90 | 33,76% |
| Đối chứng width, cùng thứ tự dữ liệu | 128/128/5 | H/T | B | 10 | 86,75 | 30,30 | 86,13 | 26,02 | 67,20 | 8,09 | 36,55% |
| Giảm channels, cùng thứ tự dữ liệu | 128/64/5 | H/T | B | 10 | 86,54 | 27,60 | 85,68 | 26,31 | 66,81 | 8,59 | 40,08% |
| Tăng đồng thời P/C | 256/256/5 | H/T | B | 10 | 86,46 | 27,07 | 85,41 | 24,92 | 66,69 | 7,90 | 36,01% |
| Tăng đồng thời P/C | 512/512/5 | H/T | B | 10 | 86,47 | 28,04 | 85,43 | 25,85 | 66,83 | 8,21 | 35,39% |
| P/C=256, train 30 epoch | 256/256/5 | H/T | B | 30 | 85,90 | 28,68 | 85,28 | 25,57 | 66,04 | 6,79 | 33,05% |
| P/C=512, train 30 epoch | 512/512/5 | H/T | B | 30 | 84,99 | 27,33 | 84,99 | 27,15 | 64,19 | 6,95 | 35,90% |

`N/A` ở lượt không loss phụ: alignment branch chưa được train đầy đủ, nên AUC2/AP2/mAP không dùng để so sánh. Hai lượt 30 epoch dùng lịch giảm learning rate sau epoch 10 và 20, khác lượt 10 epoch. Các lượt train độc lập có dao động; chưa cấu hình nào vượt baseline về AUC1.

## 3. Kết quả khi bỏ LGT-Adapter

**Kiến trúc:** CLIP visual features `[B,T,512]` -> input projection `512 -> 128` -> hai residual Conv1D blocks (`C=128`) -> scoring head tạo `logits1`. CNN features được projection trở lại 512 chiều rồi dùng cho text adjustment và `logits2`. LGT-Adapter và classifier head gốc đều bị bỏ; class names, learnable prompt và frozen CLIP text encoder vẫn giữ. Loss và cách tạo video score giống phần 1 (Top-K).

Ba lượt train from scratch dùng `k=5, 7, 9`, batch 64 Normal + 64 abnormal, seed `234`, AdamW, learning rate đầu `1e-5`, tối đa 30 epoch; learning rate giảm sau epoch 12 và 24. Checkpoint lấy theo train loss, không theo test AUC. Test gồm cùng 290 video.

| Cấu hình | Epoch chọn / đã chạy | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP | Top-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VadCLIP tự train (baseline) | - | **87,72** | **33,13** | **86,52** | **28,23** | **69,94** | **7,57** | 35,29% |
| Bỏ LGT, P/C/k = 128/128/5 | 25/30 | 84,26 | 24,92 | 83,79 | 24,90 | 63,37 | 3,89 | 38,96% |
| Bỏ LGT, P/C/k = 128/128/7 | 30/30 | 84,42 | 23,11 | 83,74 | 24,17 | 62,79 | 4,73 | 37,91% |
| Bỏ LGT, P/C/k = 128/128/9 | 30/30 | 84,54 | 25,21 | 83,51 | 23,15 | 63,39 | 4,50 | 35,54% |

Các lượt bỏ LGT đều thấp hơn baseline về AUC1 và mAP. Top-15 của `k=5` cao hơn baseline nhưng không đồng nghĩa AUC tốt hơn: AUC còn phụ thuộc thứ hạng score giữa **mọi** normal và abnormal frame trong test set. Vì lượt này vừa bỏ LGT vừa dùng classifier loss CNN, bảng chưa tách riêng tác động của từng thay đổi.
