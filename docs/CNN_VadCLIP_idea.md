
# Research & Implementation Plan
# Improving VadCLIP with Temporal Convolutional Classification

## 1. Bối cảnh nghiên cứu

Dự án nghiên cứu cải tiến VadCLIP (Weakly Supervised Video Anomaly Detection).

VadCLIP sử dụng:
- CLIP Image Encoder để trích xuất visual features.
- LGT-Adapter để học temporal relationships giữa các snippets.
- Classification Branch để dự đoán anomaly scores.
- Alignment Branch để kết hợp visual và text features.
- MIL/Top-K selection để học từ video-level labels.

Dataset chính: UCF-Crime.

### Hạn chế muốn nghiên cứu

Trong phương pháp MIL, model học snippet-level anomaly scores thông qua video-level supervision.

Quá trình training sử dụng Top-K snippets có anomaly scores cao để tổng hợp thành video-level prediction.

Vấn đề đặt ra:
- Top-K selection có thể chọn nhầm những normal snippets.
- Model không được cung cấp trực tiếp snippet-level labels.
- Các snippets có score cao chưa chắc thực sự chứa anomaly.
- Chất lượng localization phụ thuộc vào cách model học snippet representations và lựa chọn snippets.

Mục tiêu của nghiên cứu là kiểm tra xem có thể cải thiện khả năng nhận diện abnormal snippets bằng cách thiết kế lại temporal feature learning và classification mechanism hay không.

## 2. Ý tưởng nghiên cứu ban đầu

Ý tưởng high-level được đề xuất bởi giảng viên.

Thay vì tập trung vào việc sử dụng MIL để lựa chọn Top-K snippets ngay trong cơ chế học, chúng ta muốn thiết kế một classifier có khả năng học những temporal patterns mang tính phân biệt giữa normal và abnormal videos.

Ý tưởng tương tự convolution trong CNN:

1. Một video được biểu diễn bằng chuỗi snippet features.
2. Sử dụng một tập các temporal kernels để quét qua chuỗi.
3. Các kernel dùng chung trọng số trên tất cả temporal positions.
4. Mỗi kernel học cách phản hồi với những temporal patterns xuất hiện trong dữ liệu.
5. Kết hợp các kernel responses để tạo đặc trưng cho mỗi snippet.
6. Dùng shared classifier để tạo abnormality activation tại từng temporal position.
7. Tổng hợp các anomaly logits thành một video-level prediction.
8. So sánh prediction với video-level label để huấn luyện toàn bộ mô hình.

Tư tưởng quan trọng:

Không huấn luyện riêng từng snippet bằng nhãn giả định.

Thay vào đó, model được supervised ở cấp video, nhưng bộ lọc và classifier hoạt động ở cấp snippet/temporal window.

Thông qua backpropagation, các temporal kernels học những patterns hữu ích để phân biệt normal và abnormal videos.

Sau training, response của classifier tại từng temporal position được sử dụng làm snippet-level anomaly score.

### Minh họa

Ví dụ một abnormal video gồm các snippets:

S1: Walking
S2: Walking
S3: Walking
S4: Fighting
S5: Fighting
S6: Walking
S7: Walking

Mục tiêu mong muốn:

S1 → Low anomaly score
S2 → Low anomaly score
S3 → Low anomaly score
S4 → High anomaly score
S5 → High anomaly score
S6 → Low anomaly score
S7 → Low anomaly score

Classifier học thông qua nhãn toàn video:

Video label = Abnormal (1).

Ngược lại, đối với normal video, tất cả snippets lý tưởng đều có anomaly scores thấp.

Điểm cần lưu ý:
- Video-level supervision không đảm bảo classifier tự động học đúng abnormal snippets.
- CNN có thể học các patterns tương quan với video labels nhưng không phải anomalies thực sự.
- Vì vậy, phải đánh giá bằng frame-level localization metrics.

Đây vẫn là một bài toán Weakly Supervised Learning và có thể được diễn giải theo framework MIL.

Điểm khác biệt muốn nghiên cứu nằm ở cách học temporal features, shared classification và aggregation, không phải việc loại bỏ hoàn toàn MIL về mặt lý thuyết.

## 3. Dữ liệu đầu vào

VadCLIP công bố pre-extracted CLIP visual features.

Cấu trúc:

UCFClipFeatures/
  Abuse/
    Abuse001_x264__0.npy
    Abuse001_x264__1.npy
    ...
    Abuse001_x264__9.npy

Mỗi file:
- NumPy array.
- Shape: [T, 512].
- Dtype có thể là float16.
- T là số temporal positions.
- Mỗi hàng đại diện cho một snippet-level visual embedding.

Ví dụ:

Abuse001_x264__0.npy → [170, 512]

Mỗi video có 10 crop/flip variants trong training.

Lưu ý:
- 10 crops là các spatial augmentations.
- Không phải 10 temporal segments.
- Các crops của một video dùng cùng video-level label.

Training list hiện tại:
- 1610 videos.
- 16100 feature files.

Testing list:
- 290 videos.
- 290 feature files, sử dụng crop __5.

### Feature source options

Option A:

Pre-extracted CLIP Features
          ↓
Temporal CNN

Option B:

Pre-extracted CLIP Features
          ↓
Pretrained LGT-Adapter
          ↓
Temporal CNN

Theo góp ý của giảng viên, cần ưu tiên kiểm tra Option B vì LGT-Adapter đã học temporal relationships.

Tuy nhiên, vẫn giữ Option A để làm ablation và đánh giá đóng góp thực tế của LGT-Adapter.

Không giả định `.npy` đã chứa output của LGT-Adapter.

## 4. Proposed Model Architecture

### 4.1. Tổng quan

Video Feature Sequence [B, T, 512]
                 |
                 v
       Optional LGT-Adapter
                 |
                 v
       Feature Projection
        Linear(512, 128)
        LayerNorm + GELU
                 |
                 v
       Temporal CNN Block 1
                 |
                 v
         Features H1
          [B, T, 128]
                 |
                 |------> Scoring Head 1
                 |         [B, T]
                 v
       Temporal CNN Block 2
                 |
                 v
         Features H2
          [B, T, 128]
                 |
                 |------> Scoring Head 2
                 |         [B, T]
                 v
       Final Classification Head
                 |
                 v
         Final Snippet Logits
              [B, T]
                 |
                 v
      Scoring Strategy / Fusion
                 |
                 v
       Snippet Anomaly Logits
              [B, T]
                 |
                 v
        Temporal Aggregation
                 |
                 v
          Video Logit
              [B]
                 |
                 v
         BCEWithLogitsLoss
                 |
                 v
          Backpropagation

### 4.2. Vai trò từng module

LGT-Adapter:
- Học temporal dependencies.
- Có thể sử dụng pretrained weights.
- So sánh frozen và fine-tuned modes.

Feature Projection:
- Giảm feature dimension từ 512 xuống 128.
- Giảm chi phí tính toán của CNN.
- Chuẩn hóa representation trước temporal modeling.

Temporal CNN Blocks:
- Áp dụng shared temporal kernels trên feature sequence.
- Học local hoặc multi-scale temporal patterns.
- Giữ nguyên temporal resolution.

Scoring Heads:
- Chuyển temporal features thành scalar anomaly logits.
- Có thể lấy output từ từng CNN block.
- Cho phép so sánh anomaly localization giữa các tầng.

Temporal Aggregation:
- Tổng hợp snippet logits thành video-level logit.
- Cho phép sử dụng video-level supervision.

Training:
- Chỉ dùng video-level labels.
- Backpropagation từ video classification loss.
- Không cần snippet-level annotations.

## 5. Temporal CNN Architecture Options

### 5.1. Standard Temporal CNN

Input: [B, T, 128]

Sử dụng hai Conv1D layers liên tiếp:

Conv1D 128 → 128
      ↓
GELU + Normalization
      ↓
Conv1D 128 → 128
      ↓
Residual Connection
      ↓
Output [B, T, 128]

Thử kernel sizes:
- k=1
- k=3
- k=5

Stride=1, same padding.

Mục đích:

k=1:
- Chỉ xử lý feature của một temporal position.
- Không kết hợp thông tin giữa các snippets trong CNN.
- Dùng làm baseline để đo hiệu quả của temporal convolution.

k=3:
- Học local temporal transitions.
- Có receptive field lớn hơn k=1.

k=5:
- Học temporal context rộng hơn.
- Có thể phù hợp với patterns kéo dài.

Nếu input đã qua LGT-Adapter, k=1 vẫn có thể chứa temporal context từ LGT-Adapter.

### 5.2. Multi-Scale Temporal CNN

Mỗi block sử dụng 3 Conv1D branches song song.

Input [B, T, 128]
         |
         |--------------|--------------|
         v              v              v
    Conv1D k1      Conv1D k2      Conv1D k3
    64 channels    64 channels    64 channels
         |              |              |
         |--------------|--------------|
                        |
                        v
                Concatenation
                 192 channels
                        |
                        v
                Conv1D k=1
                 192 → 128
                        |
                        v
               Residual + Norm
                        |
                        v
                [B, T, 128]

Thử 2 configurations:

MSCNN-A:
- Kernels = [1, 3, 5]

MSCNN-B:
- Kernels = [3, 5, 9]

Sử dụng 2 Multi-Scale CNN Blocks liên tiếp.

Giữ:
- Stride=1.
- Dilation=1.
- Same padding.
- Không temporal downsampling.

Lý do:
- Abnormal events có thời lượng khác nhau.
- Các kernels học patterns ở nhiều temporal scales.
- Feature fusion kết hợp short-term và longer local context.
- Shared weights cho phép phát hiện patterns tại mọi vị trí thời gian.

## 6. Multi-Level Anomaly Scoring

Theo góp ý của giảng viên, cần thử lấy anomaly scores từ nhiều tầng CNN.

### S1. Final-Layer Scoring

H2 → Linear(128,1) → Snippet Logits

Đây là baseline đơn giản.

### S2. Intermediate-Layer Scoring

H1 → Linear(128,1) → S1

H2 → Linear(128,1) → S2

Đánh giá riêng:
- Block 1 score.
- Block 2 score.

Lưu ý:
- CNN feature maps không trực tiếp là anomaly scores.
- Cần scoring head để tạo scalar logits.

### S3. Multi-Level Fusion

Kết hợp:

S_final = α*S1 + (1-α)*S2

Thử:
- α=0.25
- α=0.5
- α=0.75

Sau đó có thể thử learnable fusion weights.

Mục đích:
- Kiểm tra thông tin anomaly từ các tầng CNN.
- Đánh giá xem fusion nhiều tầng có cải thiện localization hay không.

## 7. Temporal Aggregation

Snippet logits:

S = [s1, s2, ..., sT]

Cần tổng hợp thành một video logit.

### A1. Max Pooling

z_video = max(s_t)

Chỉ tính valid temporal positions.

Phù hợp khi anomaly xuất hiện ngắn, nhưng nhạy với outliers.

### A2. Top-K Pooling

z_video = mean(TopK(S, K))

Thử:
- K=max(1, ceil(T_valid/16))
- K=max(1, ceil(T_valid/8))

Ưu điểm:
- Tập trung vào nhóm snippets nghi ngờ abnormal.
- Ít phụ thuộc vào một snippet duy nhất hơn Max Pooling.

### A3. LogSumExp Pooling

z_video = τ * log(
    (1/T_valid) * sum(exp(s_t/τ))
)

Thử:
- τ=1.0

Implement bằng masked logsumexp để ổn định số học.

Ưu tiên Max và Top-K theo góp ý của giảng viên.

LogSumExp được giữ làm phương án đối chứng.

## 8. Training Strategy

### 8.1. Configuration

Dataset: UCF-Crime
Input: CLIP features hoặc LGT-Adapter features
Batch size: 32
Optimizer: AdamW
Learning rate: 1e-4
Epochs: 50
Early stopping: Có
Loss: BCEWithLogitsLoss

Các thông số chỉ là giá trị khởi đầu.

### 8.2. Loss Function

Với batch B videos:

z_i = video-level logit
y_i = video-level label

p_i = sigmoid(z_i)

L_BCE = -(1/B) * sum_i [
    y_i*log(p_i) +
    (1-y_i)*log(1-p_i)
]

Trong PyTorch:

loss = BCEWithLogitsLoss(video_logits, labels)

Không áp dụng sigmoid trước khi truyền logits vào loss.

### 8.3. Training Flow

1. Load CLIP features và video labels.
2. Tạo batch với dynamic padding và valid masks.
3. Chạy optional LGT-Adapter.
4. Chạy Feature Projection.
5. Chạy Temporal CNN Blocks.
6. Tính snippet logits bằng scoring heads.
7. Thực hiện score fusion nếu được cấu hình.
8. Aggregate snippet logits thành video logits.
9. Tính BCEWithLogitsLoss.
10. Backpropagation và optimizer step.

Yêu cầu quan trọng:
- Padding không được ảnh hưởng đến video aggregation.
- Kiểm soát padding contamination trong temporal convolution.
- Đảm bảo alignment giữa output temporal scores và ground truth.
- Không dùng frame-level test labels để train.
- Không làm thay đổi test protocol giữa các experiments.

## 9. Implementation Plan

### Phase 1: Repository Inspection

Trước khi code:
- Đọc implementation hiện tại.
- Kiểm tra Dataset/DataLoader.
- Kiểm tra feature shapes.
- Kiểm tra LGT-Adapter.
- Kiểm tra pretrained checkpoints.
- Kiểm tra evaluation pipeline.

Không tự giả định các cấu trúc chưa được xác minh.

### Phase 2: Baseline Model

Implement:
- Feature Projection.
- Standard CNN.
- Final-Layer Scoring.
- Max Pooling.
- BCE Loss.

Thử k=1,3,5.

### Phase 3: Multi-Scale CNN

Implement:
- MSCNN [1,3,5].
- MSCNN [3,5,9].

Giữ các thành phần còn lại giống Phase 2 để so sánh công bằng.

### Phase 4: Aggregation Experiments

Trên các kiến trúc tốt nhất:

- Max Pooling.
- Top-K Pooling, K=T/16.
- Top-K Pooling, K=T/8.
- LogSumExp Pooling.

### Phase 5: Multi-Level Scoring

Thử:
- Block 1 scoring.
- Block 2 scoring.
- Final-Layer scoring.
- Score fusion.

### Phase 6: LGT-Adapter Integration

Thử:
- CLIP → CNN.
- CLIP → Frozen LGT → CNN.
- CLIP → Fine-tuned LGT → CNN (optional).

Với pretrained LGT-Adapter, kiểm tra nguồn checkpoint và training protocol để tránh leakage.

## 10. Evaluation

Primary metric:
- Frame-level ROC-AUC trên UCF-Crime.

Additional metrics:
- Frame-level AP.
- Video-level Accuracy.
- Parameter count.
- GPU memory.
- Inference latency.

Baseline:
- Original VadCLIP.

Mỗi experiment cần lưu:
- Configuration.
- Training logs.
- Validation metrics.
- Checkpoint.
- Random seed.

Chạy ít nhất 3 random seeds cho các model tốt nhất.

## 11. Implementation Requirements for Codex

Yêu cầu Codex:

1. Đọc và phân tích repository trước khi sửa code.
2. Xác nhận data pipeline và tensor dimensions thực tế.
3. Đề xuất cấu trúc module trước khi implement.
4. Không sửa implementation VadCLIP gốc nếu không cần thiết.
5. Xây dựng code có khả năng cấu hình:
   - Feature source.
   - CNN architecture.
   - Kernel sizes.
   - Aggregation.
   - Scoring strategy.
   - LGT-Adapter mode.
6. Implement và kiểm thử từng phase.
7. Viết shape tests và masking tests.
8. Đảm bảo experiments reproducible.
9. Không tự tuyên bố phương pháp cải thiện VadCLIP nếu chưa có kết quả.
10. Báo cáo những vấn đề kiến trúc hoặc dữ liệu phát hiện được trước khi thay đổi thiết kế.

Ưu tiên triển khai baseline đơn giản trước, sau đó mở rộng sang Multi-Scale CNN, aggregation ablation, multi-level scoring và LGT-Adapter integration.
