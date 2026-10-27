# Báo cáo thí nghiệm VadCLIP với Temporal CNN

## 1. Dữ liệu và quy trình đánh giá

- **Đầu vào:** feature thị giác 512 chiều đã được trích xuất bằng CLIP cho từng snippet; không huấn luyện lại CLIP. Một mẫu huấn luyện là một chuỗi feature ứng với một phiên bản crop/flip của video và mang nhãn cấp video.
- **Tập huấn luyện:** toàn bộ 16.100 chuỗi feature, gồm 8.000 chuỗi Normal và 8.100 chuỗi abnormal. Mỗi chuỗi dài tối đa 256 snippet: chuỗi dài hơn được lấy mẫu đều và gộp feature về 256 vị trí; chuỗi ngắn hơn được đệm bằng số 0. Phần đệm không tham gia phép gộp score.
- **Tập kiểm tra:** 290 video. Video dài được tách thành các khối tối đa 256 snippet để suy luận; score được ghép lại theo thứ tự và bỏ phần đệm. Mỗi score snippet được lặp 16 lần để đối chiếu với nhãn frame.
- **Mốc so sánh:** checkpoint VadCLIP **tự huấn luyện từ đầu**, được đánh giá lại với cùng tập kiểm tra và evaluator của lượt thí nghiệm này. Mốc AUC classifier thu được ở đây là **87,72%**.

## 2. Cách đưa Temporal CNN vào VadCLIP

Pipeline tính score là:

```text
CLIP visual features [B,T,512]
    -> LGT-Adapter gốc (temporal attention + graph)
    -> visual features [B,T,512]
    -> projection 512 -> 128
    -> 2 residual Temporal CNN blocks
    -> logits1 [B,T,1]
    -> điều chỉnh text features như VadCLIP gốc
    -> alignment logits2 [B,T,14]
```

Temporal CNN **thay thế hoàn toàn** đường classifier theo từng snippet của VadCLIP gốc (`mlp2` và lớp classifier), không cộng thêm score CNN vào classifier cũ. Phần mã hóa thị giác LGT-Adapter, 14 tên class, learnable prefix/postfix của text prompt và cách tính alignment còn lại được giữ nguyên. `logits1` mới cũng được sử dụng tại bước VadCLIP lấy thông tin thị giác để điều chỉnh text features trước khi tính `logits2`.

Chi tiết nhánh CNN:

1. Mỗi feature 512 chiều được chiếu xuống 128 chiều bằng `Linear -> LayerNorm -> GELU`.
2. Mỗi trong hai block thực hiện `Conv1D(128 -> 128) -> GELU -> Conv1D(128 -> 128) -> GELU`, cộng residual với đầu vào block rồi chuẩn hóa bằng `LayerNorm`. Convolution trượt dọc trục thời gian, stride 1 và đệm hai phía để giữ nguyên số snippet. Kernel được thử với kích thước 1, 3 hoặc 5.
3. Mask độ dài hợp lệ được áp dụng sau từng block; snippet đệm không được dùng để lấy video score.
4. Ở chế độ **scoring head**, `Linear(128 -> 1)` kết hợp 128 channel sau block cuối thành một logit cho từng snippet.
5. Ở chế độ **kernel response**, lấy phản hồi lớn nhất trong 128 channel của convolution thứ hai ở block cuối, *trước* bước cộng residual và chuẩn hóa; sau đó áp dụng hệ số dương học được và bias học được để tạo logit. Đây không phải phép chọn cứng snippet abnormal.

## 3. Gộp score và hàm loss

Gọi `z_t` là logit classifier của snippet hợp lệ thứ `t` và `T` là số snippet hợp lệ trong mẫu. Hai cách tạo một logit ở cấp video được thử nghiệm:

- **Top-K:** `K = max(1, ceil(T/16))`; lấy trung bình `K` giá trị `z_t` cao nhất. Với `T = 256`, `K = 16`.
- **Max:** lấy giá trị `z_t` lớn nhất. Chỉ cách gộp video score thay đổi; CNN vẫn tạo score cho mọi snippet.

Video logit được đưa thẳng vào `BCEWithLogitsLoss` với nhãn video Normal = 0, abnormal = 1; không sigmoid trước khi tính loss. Tổng loss là:

```text
L_total = L_video_BCE + lambda_alignment * L_alignment_MIL
                      + lambda_text * L_text_reg
```

- `L_video_BCE`: binary cross-entropy của video logit từ nhánh CNN và nhãn cấp video. Loss classifier MIL cũ của VadCLIP **không** được giữ lại.
- `L_alignment_MIL`: với mỗi class, chọn các logit alignment cao nhất trên snippet hợp lệ (`K_align = floor(T/16) + 1`), lấy trung bình theo thời gian, rồi tính cross-entropy với nhãn class của video.
- `L_text_reg`: phạt độ tương đồng cosine tuyệt đối giữa embedding của Normal và embedding của từng class abnormal, lấy trung bình và nhân hệ số `0,1` bên trong định nghĩa loss.
- Trong các biến thể thông thường, `lambda_alignment = 1` và `lambda_text = 1`. Biến thể `no_aux` đặt cả hai bằng 0, nên chỉ huấn luyện bằng `L_video_BCE`; nhánh text/alignment không được huấn luyện trong biến thể này.

## 4. Cấu hình huấn luyện

| Thành phần | Cấu hình |
|---|---|
| Khởi tạo | VadCLIP train from scratch; chỉ CLIP pretrained được giữ frozen, không nạp checkpoint VadCLIP đã train |
| Seed | 234 |
| Batch mỗi bước | 64 chuỗi Normal + 64 chuỗi abnormal |
| Số epoch | 10 |
| Optimizer | AdamW |
| Learning rate ban đầu | `1e-5` |
| Lịch learning rate | `MultiStepLR`, nhân `0,1` tại các mốc epoch 4 và 8 |
| Độ dài chuỗi tối đa | 256 snippet |
| Số channel ẩn của CNN | 128 |
| Số residual CNN block | 2 |
| AMP | Tắt |
| Checkpoint và chọn kết quả | Lưu mỗi epoch; báo cáo checkpoint epoch 10, không chọn epoch theo AUC trên test |

Các biến thể được huấn luyện độc lập với cùng cấu hình chung:

| Biến thể | Kernel | Cách tạo score/snippet | Gộp score/video | `lambda_alignment` | `lambda_text` |
|---|---:|---|---|---:|---:|
| `k1_head_topk` | 1 | Scoring head | Top-K | 1 | 1 |
| `k3_head_topk` | 3 | Scoring head | Top-K | 1 | 1 |
| `k5_head_topk` | 5 | Scoring head | Top-K | 1 | 1 |
| `k3_kernel_topk` | 3 | Kernel response | Top-K | 1 | 1 |
| `k5_kernel_topk` | 5 | Kernel response | Top-K | 1 | 1 |
| `k3_head_max` | 3 | Scoring head | Max | 1 | 1 |
| `k3_head_max_no_aux` | 3 | Scoring head | Max | 0 | 0 |

## 5. Kết quả trên tập kiểm tra

Các số trong bảng là phần trăm. **AUC1/AP1** được tính trên toàn bộ frame kiểm tra từ `sigmoid(logits1)`; phép gộp Top-K hoặc Max chỉ dùng để tính loss cấp video khi train, **không** dùng để tính AUC1. **AUC2/AP2** được tính từ score alignment của `logits2`. **Ano-AUC** là AUC trên các video abnormal, dùng nhánh có AUC toàn cục cao hơn trong các nhánh đã huấn luyện. **mAP** là trung bình trên các ngưỡng IoU từ 0,1 đến 0,5.

| Cấu hình | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP |
|---|---:|---:|---:|---:|---:|---:|
| VadCLIP tự train, đánh giá lại | 87,72 | 33,13 | 86,52 | 28,23 | 69,94 | 7,57 |
| `k1_head_topk` | 86,26 | 28,96 | 86,10 | 28,27 | 65,96 | 6,91 |
| `k3_head_topk` | 85,38 | 25,81 | 84,63 | 24,18 | 64,24 | 7,09 |
| `k5_head_topk` | 86,94 | 30,99 | 85,93 | 25,58 | 67,83 | 7,05 |
| `k3_kernel_topk` | 85,43 | 27,36 | 85,96 | 26,58 | 66,21 | 9,11 |
| `k5_kernel_topk` | 86,75 | 32,27 | 86,17 | 29,30 | 68,16 | 7,93 |
| `k3_head_max` | 85,37 | 25,50 | 84,51 | 24,36 | 64,35 | 6,43 |
| `k3_head_max_no_aux` | 85,66 | 27,99 | — | — | 64,12 | — |

Chi tiết mAP theo ngưỡng IoU cho các mô hình có huấn luyện alignment:

| Cấu hình | IoU 0,1 | IoU 0,2 | IoU 0,3 | IoU 0,4 | IoU 0,5 |
|---|---:|---:|---:|---:|---:|
| VadCLIP tự train, đánh giá lại | 14,12 | 9,47 | 6,63 | 4,27 | 3,34 |
| `k1_head_topk` | 12,03 | 8,84 | 5,97 | 4,84 | 2,89 |
| `k3_head_topk` | 13,06 | 9,05 | 5,55 | 4,85 | 2,93 |
| `k5_head_topk` | 12,41 | 9,09 | 6,28 | 4,66 | 2,82 |
| `k3_kernel_topk` | 15,38 | 11,72 | 8,92 | 5,26 | 4,25 |
| `k5_kernel_topk` | 13,35 | 10,26 | 8,00 | 4,39 | 3,66 |
| `k3_head_max` | 12,54 | 8,55 | 5,19 | 3,54 | 2,32 |

Ở `k3_head_max_no_aux`, evaluator vẫn có thể xuất giá trị số cho alignment từ các tham số chưa huấn luyện, nhưng AUC2/AP2/mAP đó không được xem là kết quả đánh giá hợp lệ nên không đưa vào bảng.
