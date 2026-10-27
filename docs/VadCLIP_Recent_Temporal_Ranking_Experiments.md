# Nhật ký thí nghiệm: Thông tin thời gian và Ranking Margin

## Mốc đối chiếu

Các lượt chạy đều dùng nhãn **cấp video**, không dùng nhãn cho từng snippet khi huấn luyện. Kết quả được đo trên cùng 290 video kiểm tra. VadCLIP tự huấn luyện là mốc so sánh; checkpoint được đánh giá trong các lượt này đạt AUC1 **87,72%**. 

AUC1/AP1 đánh giá nhánh chấm điểm bất thường trực tiếp; AUC2/AP2 đánh giá điểm bất thường từ nhánh so khớp hình ảnh với lớp. Ano-AUC chỉ xét các frame trong video abnormal; mAP đánh giá nhận diện và định vị loại bất thường. Các chỉ số đều tính theo %.

**Ba loss chung của VadCLIP:** Với mỗi chuỗi có `T` snippet hợp lệ, mô hình gốc chọn `floor(T/16) + 1` snippet điểm cao nhất. `L_cls` là binary cross-entropy giữa điểm bất thường trung bình của nhóm này và nhãn normal/abnormal của mẫu. `L_mil` là cross-entropy giữa điểm top-k của 14 lớp và nhãn lớp cấp video. `L_text` bằng `0,1` lần trung bình độ tương tự cosine tuyệt đối giữa embedding chữ của Normal và 13 lớp còn lại; mục đích là hạn chế các embedding đó quá giống nhau.

## 1. Hiệu chỉnh điểm bằng tích chập thời gian

**Cách làm và cấu trúc**

- VadCLIP gốc vẫn tạo đặc trưng 512 chiều cho từng snippet, điểm của nhánh phát hiện bất thường và điểm so khớp với các lớp. Thêm một nhánh tích chập 1D nhìn ba snippet liền kề, tạo đặc trưng trung gian 128 chiều rồi chiếu về một giá trị điều chỉnh cho mỗi snippet.
- Thử hai đầu vào riêng: **chuỗi đặc trưng** của các snippet liên tiếp; hoặc **độ thay đổi đặc trưng** từ snippet trước sang snippet hiện tại và từ snippet hiện tại sang snippet sau. Cách thứ hai nhấn mạnh sự thay đổi theo thứ tự thời gian thay vì đặc trưng tĩnh.
- Giá trị điều chỉnh được cộng vào *logit* của nhánh phát hiện trước khi đổi thành xác suất. Nó bằng 0 khi bắt đầu train và bị chặn trong khoảng `-0,5` đến `0,5`. Nhánh so khớp lớp vẫn dùng đường tính gốc, không nhận trực tiếp giá trị này.

**Loss và cấu hình**

- Tổng loss là **`L_cls + L_mil + L_text`**, không thêm loss riêng cho nhánh thời gian. Gradient từ `L_cls` huấn luyện nhánh mới cùng VadCLIP.
- Train từ đầu trong 10 epoch, tốc độ học ban đầu `1e-5`, giảm ở epoch 4 và 8. Mỗi batch có 64 mẫu chuỗi normal và 64 mẫu chuỗi abnormal; CLIP pretrained được giữ cố định. Kết quả báo theo epoch 10.

**Kết quả**

| Phương pháp | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP |
|---|---:|---:|---:|---:|---:|---:|
| VadCLIP tự huấn luyện | **87,72** | **33,13** | **86,52** | **28,23** | **69,94** | 7,57 |
| Dùng chuỗi đặc trưng | 87,17 | 30,84 | 85,88 | 27,19 | 68,53 | 7,59 |
| Dùng độ thay đổi đặc trưng | 87,40 | 32,70 | 85,98 | 27,33 | 69,08 | 7,09 |

Cả hai biến thể đều giảm AUC1. Biến thể chuỗi đặc trưng tăng mAP rất ít, khoảng **0,03 điểm**, còn biến thể độ thay đổi giảm mAP khoảng **0,48 điểm**.

## 2. Học ngữ cảnh giữa các nhóm snippet bằng GRU

**Cách làm và cấu trúc**

- VadCLIP xử lý từng chuỗi feature như cũ. Trong từng góc nhìn của video, cứ **32 snippet liên tiếp** được lấy trung bình thành một đại diện 512 chiều. Các đại diện được đưa theo thứ tự vào GRU hai chiều, mỗi hướng có 128 chiều, để mang thông tin từ trước và sau. Mười góc nhìn/crop của cùng video được xử lý thành các chuỗi riêng, không bị hiểu nhầm là mười thời điểm liên tiếp.
- Với từng snippet, một mạng hai lớp nhỏ kết hợp đặc trưng cục bộ với ngữ cảnh GRU rồi cộng một giá trị vào logit phát hiện bất thường. Giá trị này bằng 0 lúc khởi tạo và bị giới hạn trong `-0,1` đến `0,1`. Nhánh so khớp lớp vẫn dùng điểm gốc, không nhận trực tiếp phần cộng thêm.
- Trạng thái tổng hợp của GRU còn dự đoán cả chuỗi video là normal hay abnormal. Đây là tín hiệu từ **nhãn video**, không phải nhãn cho 32 snippet trong từng nhóm.

**Loss và cấu hình**

- Giữ ba loss gốc `L_cls`, `L_mil`, `L_text` như trên và thêm `L_video`: binary cross-entropy giữa dự đoán GRU và nhãn normal/abnormal của video. Tổng loss là **`L_cls + L_mil + L_text + λ × L_video`**. Mỗi video được lấy trung bình qua các góc nhìn trước khi gộp loss, tránh để nhiều crop của một video được tính như nhiều video độc lập.
- Train từ đầu trong 10 epoch, tốc độ học `1e-5`, giảm ở epoch 4 và 8; mỗi bước xử lý 2 video normal và 2 video abnormal. Hệ số `λ` là `0,05` ở epoch 1, sau đó `0,1`; CLIP pretrained được giữ cố định. Kết quả báo theo epoch 10.

**Kết quả**

| Phương pháp | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP |
|---|---:|---:|---:|---:|---:|---:|
| VadCLIP tự huấn luyện | **87,72** | **33,13** | **86,52** | **28,23** | **69,94** | 7,57 |
| Thêm GRU ngữ cảnh | 87,02 | 28,54 | 86,08 | 26,46 | 68,02 | **8,26** |

mAP tăng khoảng **0,69 điểm**, nhưng AUC1 giảm **0,69 điểm**, AP1 giảm **4,59 điểm** và Ano-AUC giảm **1,92 điểm**. Như vậy, kết quả nhận diện/định vị lớp tốt hơn đôi chút nhưng khả năng phát hiện bất thường ở frame yếu đi.

## 3. Tách điểm video abnormal và normal bằng Ranking Margin

**Cách làm và cấu trúc**

- Không thêm nhánh mới: dùng toàn bộ cấu trúc VadCLIP gốc, gồm nhánh chấm điểm bất thường và nhánh so khớp lớp. Với **mỗi mẫu chuỗi trong batch**, lấy trung bình điểm của **15% snippet cao nhất**, rồi tính trung bình riêng cho nhóm mẫu normal và nhóm mẫu abnormal.
- Nếu điểm trung bình của nhóm abnormal cao hơn nhóm normal ít nhất **0,20**, loss mới bằng 0. Nếu chưa đủ khoảng cách, loss phạt phần còn thiếu. Phép so sánh được thực hiện riêng trên điểm của hai nhánh rồi lấy trung bình. Phương pháp không biết snippet nào trong mẫu abnormal là bất thường thật.
- Cụ thể, gọi `N1`, `A1` là trung bình top 15% của nhóm normal/abnormal ở nhánh 1; `N2`, `A2` tương ứng ở nhánh 2. Khi đó **`L_rank = [max(0; 0,20 + N1 - A1) + max(0; 0,20 + N2 - A2)] / 2`**. Hai nhánh đều dùng điểm bất thường sau khi chuyển về khoảng 0-1.

**Loss và cấu hình**

- Tổng loss là **`L_cls + L_mil + L_text + λ × L_rank`**, trong đó `L_rank` là phần thiếu của khoảng cách `0,20` vừa mô tả. `L_cls` và `L_mil` vẫn dùng cách chọn top-k gốc của VadCLIP; **top 15% chỉ dành cho `L_rank`**, không thay cách tính các loss gốc.
- Train từ đầu trong 10 epoch, tốc độ học `1e-5`, giảm ở epoch 4 và 8; mỗi batch có 64 mẫu chuỗi normal và 64 mẫu chuỗi abnormal. Hệ số `λ` bằng 0 ở epoch 1 và bằng `0,05` từ epoch 2; CLIP pretrained được giữ cố định. Kết quả chính là epoch 10, không chọn epoch theo điểm kiểm tra.

**Kết quả**

| Phương pháp | AUC1 | AP1 | AUC2 | AP2 | Ano-AUC | mAP |
|---|---:|---:|---:|---:|---:|---:|
| VadCLIP tự huấn luyện | **87,72** | **33,13** | **86,52** | **28,23** | **69,94** | **7,57** |
| Ranking Margin | 86,57 | 30,62 | 85,62 | 26,30 | 67,36 | 6,94 |

AUC1 giảm **1,15 điểm**, mAP giảm **0,63 điểm**. `L_rank` trung bình gần bằng 0 từ epoch 2, cho thấy khoảng cách yêu cầu giữa **hai nhóm mẫu** hầu như đã có; điều này không đồng nghĩa mọi snippet abnormal đều được chấm cao hơn snippet normal.
