Ý tưởng training CHiLS cho VadCLIP

Giữ nguyên phần lớn kiến trúc và cơ chế training của VadCLIP gốc. Thay đổi chính nằm ở A-branch, cụ thể là cách biểu diễn text label.

Với mỗi class gốc, tạo một tập các subclass có ý nghĩa semantic cụ thể hơn. Ví dụ:

Explosion
→ Vehicle Explosion
→ Building Explosion
→ Indoor Explosion
→ Outdoor Explosion
→ Explosion with Fire/Smoke

Mỗi subclass được đưa qua shared learnable prompt của VadCLIP và CLIP Text Encoder để tạo một subclass embedding riêng. Tất cả subclass dùng chung bộ learnable context token, còn CLIP Text Encoder vẫn được freeze như kiến trúc gốc.

Với mỗi frame/snippet feature x_t, model tính similarity với từng subclass embedding:

M(t,c,j) = sim(x_t, T(c,j))

Trong đó:
- c là superclass
- j là subclass thuộc superclass đó

Sau đó, với từng subclass, sử dụng Temporal Top-K giống cơ chế MIL-Align của VadCLIP để lấy các frame có mức alignment cao nhất:

q(c,j) = Mean(TopK_t(M(t,c,j)))

Kết quả là mỗi subclass có một video-level score.

Vì dataset chỉ có ground truth ở mức original class, các subclass score thuộc cùng một superclass sẽ được aggregate về class gốc.

Cách 1: Max aggregation

q(c) = max_j q(c,j)

Ý nghĩa là chỉ cần một subclass có evidence mạnh thì superclass được coi là có score cao. Ví dụ:

Vehicle Explosion   0.82
Building Explosion  0.45
Outdoor Explosion   0.63

→ Explosion = 0.82

Ưu điểm là đơn giản và phù hợp với trực giác rằng một video chỉ cần thể hiện một dạng cụ thể của Explosion. Nhược điểm là gradient chủ yếu đi qua subclass có score lớn nhất.

Cách 2: Soft aggregation

Có thể sử dụng LogSumExp:

q(c) = tau * log( sum_j exp(q(c,j) / tau) )

hoặc phiên bản chuẩn hóa theo số subclass:

q(c) = tau * log( (1/m) * sum_j exp(q(c,j) / tau) )

Trong đó tau là temperature.

- tau nhỏ: kết quả gần Max.
- tau lớn: nhiều subclass cùng đóng góp hơn.

Soft aggregation vẫn ưu tiên subclass có score cao nhất nhưng cho phép các subclass còn lại tiếp tục nhận gradient trong training.

Sau khi aggregate, output quay trở lại original label space của VadCLIP:

Normal, Abuse, Explosion, Fighting, Robbery, ...

Từ đó tiếp tục tính L_nce với video-level category ground truth như VadCLIP gốc.

Tổng thể pipeline:

Class
→ Subclasses
→ Shared Learnable Prompt
→ CLIP Text Encoder
→ Subclass embeddings
→ Frame-subclass similarity
→ Temporal Top-K
→ Subclass scores
→ Max hoặc Soft aggregation
→ Superclass score
→ L_nce với original video-level label

Điểm chính là không cần subclass ground truth. Model vẫn chỉ được supervise bằng original video-level label, nhưng sẽ tự học subclass nào phù hợp nhất với video và frame nào cung cấp evidence mạnh nhất cho subclass đó.