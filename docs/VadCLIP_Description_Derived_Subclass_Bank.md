# Bộ subclass xây dựng từ video description của UCF-Crime

## 1. Mục tiêu

Bộ subclass này được xây dựng để biểu diễn các dạng xuất hiện khác nhau của từng class abnormal trong UCF-Crime. Model vẫn chỉ nhận nhãn class gốc ở cấp video; subclass đóng vai trò cung cấp các khái niệm text cụ thể hơn cho nhánh alignment của VadCLIP.

Mỗi subclass abnormal có đúng hai từ theo cấu trúc:

```text
<Tên class gốc> <Từ mô tả>
```

Ví dụ: `Explosion Vehicle`, `Fighting Group`, `Stealing Bicycle`.

## 2. Nguồn dữ liệu

- Taxonomy chỉ được xây dựng từ train split để tránh sử dụng thông tin của test set.
- Có 1.162 video train có description hợp lệ, gồm 574 video abnormal và 588 video Normal.
- Description của mỗi video được tổng hợp từ các câu mô tả theo timestamp trong UCA.
- Description được dùng để nhận diện các tình huống lặp lại trong từng class, không được dùng làm nhãn subclass chính xác cho từng frame.

Số description abnormal đã xem xét:

| Class | Số description train |
|---|---:|
| Abuse | 36 |
| Arrest | 41 |
| Arson | 18 |
| Assault | 25 |
| Burglary | 64 |
| Explosion | 7 |
| Fighting | 37 |
| RoadAccidents | 106 |
| Robbery | 83 |
| Shooting | 31 |
| Shoplifting | 30 |
| Stealing | 66 |
| Vandalism | 30 |

## 3. Nguyên tắc thiết kế

| Nguyên tắc | Cách áp dụng |
|---|---|
| Giữ class anchor | Tên class gốc luôn đứng trước từ mô tả để embedding không rời khỏi khái niệm chính. |
| Ưu tiên tín hiệu thị giác lớn | Dùng số người, phương tiện, bối cảnh, tư thế và kiểu tương tác có khả năng xuất hiện rõ trong frame. |
| Hạn chế vật thể nhỏ | Không dùng điện thoại, ví, bật lửa, còng tay hoặc loại vũ khí cụ thể làm subclass chính. Những chi tiết này có thể quá nhỏ trong feature cấp snippet. |
| Không mô tả chuỗi dài | Subclass không yêu cầu quan sát toàn bộ diễn biến trước và sau sự kiện. Mỗi subclass cần có ý nghĩa từ một frame hoặc một đoạn ngắn. |
| Số lượng biến thiên | Class đa dạng có nhiều subclass hơn; class ít dữ liệu hoặc dễ nhiễu có ít subclass hơn. |
| Một fallback duy nhất | Mỗi class dùng hậu tố `Other` cho các trường hợp hợp lệ nhưng không thuộc nhóm cụ thể nào. |
| Không chia Normal | `Normal` được giữ nguyên như class gốc, tránh tạo nhiều normal subclass không cần thiết. |

`Other` không mang nghĩa thiếu dữ liệu hoặc không biết nhãn. Ví dụ, `Explosion Other` vẫn là một video Explosion, nhưng bối cảnh không thể hiện rõ là phương tiện, công trình hay ven đường. Description rỗng hoặc không hợp lệ không được đưa vào quá trình xây taxonomy.

## 4. Bộ subclass cuối cùng

| Class | Subclass | Lý do lựa chọn |
|---|---|---|
| Normal | `Normal` | Normal không cần chia nhỏ vì mục tiêu chính là mở rộng các dạng abnormal. |
| Abuse | `Abuse Child`, `Abuse Wheelchair`, `Abuse Custodial`, `Abuse Group`, `Abuse Grounded`, `Abuse Other` | Description thường khác nhau theo đối tượng dễ tổn thương, môi trường có người giám sát, số người tham gia và trạng thái nạn nhân. Đây là các dấu hiệu tương đối lớn trong frame. |
| Arrest | `Arrest Grounded`, `Arrest Escorted`, `Arrest Vehicle`, `Arrest Group`, `Arrest Other` | Các trạng thái khống chế dưới đất, dẫn đi, tương tác gần phương tiện và nhiều người cùng tham gia xuất hiện lặp lại. Không dùng `Handcuff` vì còng tay thường quá nhỏ. |
| Arson | `Arson Vehicle`, `Arson Building`, `Arson Vegetation`, `Arson Outdoor`, `Arson Other` | Đối tượng bị đốt là dấu hiệu dễ quan sát hơn dụng cụ gây cháy. `Outdoor` bao phủ các vụ đốt ngoài trời không gắn rõ với phương tiện, công trình hoặc cây cối. |
| Assault | `Assault Individual`, `Assault Group`, `Assault Armed`, `Assault Grounded`, `Assault Other` | Phân chia theo số người tấn công, sự xuất hiện của công cụ đủ rõ và trạng thái nạn nhân. Các nhóm này mô tả hình thức tương tác thay vì chi tiết cảnh. |
| Burglary | `Burglary Door`, `Burglary Window`, `Burglary Vehicle`, `Burglary ATM`, `Burglary Interior`, `Burglary Other` | Cửa, cửa sổ, phương tiện dùng để đột nhập, ATM và hoạt động bên trong công trình là các tình huống lặp lại rõ trong description. |
| Explosion | `Explosion Vehicle`, `Explosion Building`, `Explosion Roadside`, `Explosion Other` | Explosion chỉ có 7 description train nên chỉ chia theo các bối cảnh lớn. `Roadside` giữ lại các sự kiện xảy ra trên hoặc sát đường nhưng không thể hiện rõ phương tiện là nguồn nổ. Không dùng khói hoặc lửa vì dễ chồng lấn với Arson. |
| Fighting | `Fighting Pair`, `Fighting Group`, `Fighting Grappling`, `Fighting Grounded`, `Fighting Other` | Số người và kiểu tương tác cơ thể là tín hiệu chính để phân biệt các dạng đánh nhau. Chúng đáng tin cậy hơn loại vũ khí nhỏ. |
| RoadAccidents | `RoadAccidents Pedestrian`, `RoadAccidents Motorcycle`, `RoadAccidents Bicycle`, `RoadAccidents Multivehicle`, `RoadAccidents Truck`, `RoadAccidents Bus`, `RoadAccidents Rollover`, `RoadAccidents Barrier`, `RoadAccidents Other` | Đây là class có nhiều description và nhiều tình huống nhất. Các tác nhân giao thông và dạng va chạm đều có kích thước lớn, phù hợp với feature thị giác. |
| Robbery | `Robbery Armed`, `Robbery Counter`, `Robbery Street`, `Robbery Vehicle`, `Robbery Bag`, `Robbery Group`, `Robbery Other` | Description thể hiện các khác biệt về hình thức đe dọa, bối cảnh tại quầy hoặc ngoài đường, phương tiện, cướp túi và số người tham gia. `Armed` được dùng thay cho loại súng hoặc dao cụ thể. |
| Shooting | `Shooting Indoor`, `Shooting Outdoor`, `Shooting Vehicle`, `Shooting Crowd`, `Shooting Police`, `Shooting Other` | Súng thường chiếm diện tích nhỏ và khó nhận diện. Vì vậy phần lớn subclass dựa trên bối cảnh lớn và số người thay vì loại vũ khí. `Police` dùng tín hiệu tổng thể như đồng phục, đội hình và phương tiện thực thi pháp luật, không phụ thuộc vào việc nhìn thấy khẩu súng. |
| Shoplifting | `Shoplifting Bag`, `Shoplifting Clothing`, `Shoplifting Shelf`, `Shoplifting Counter`, `Shoplifting Group`, `Shoplifting Other` | Cách che giấu, vị trí thao tác hàng hóa và số người là các dấu hiệu lặp lại. Không chia theo loại sản phẩm vì sản phẩm thường nhỏ. |
| Stealing | `Stealing Vehicle`, `Stealing Motorcycle`, `Stealing Bicycle`, `Stealing Bag`, `Stealing Cargo`, `Stealing Indoor`, `Stealing Other` | Description thể hiện rõ sự khác biệt giữa trộm phương tiện, tài sản mang theo, hàng hóa lớn và hành vi trong nhà. Điện thoại hoặc ví được gộp vào `Other` vì quá nhỏ. |
| Vandalism | `Vandalism Vehicle`, `Vandalism Glass`, `Vandalism Graffiti`, `Vandalism Building`, `Vandalism Structure`, `Vandalism Group`, `Vandalism Other` | Phân chia theo đối tượng bị phá hoại và số người. `Structure` bao gồm biển báo, hàng rào và công trình nhỏ, còn `Building` chỉ mặt ngoài hoặc bộ phận của tòa nhà. |

Tổng cộng có 78 subclass abnormal. `Normal` được giữ dưới dạng một prompt riêng.

## 5. Vì sao không cố định số subclass

Số lượng dữ liệu và độ đa dạng giữa các class rất khác nhau. `RoadAccidents` có 106 description train và nhiều loại tác nhân giao thông, trong khi `Explosion` chỉ có 7 description. Ép cả hai class dùng cùng số subclass sẽ dẫn đến một trong hai vấn đề:

- Class đa dạng bị gộp quá mức và mất các dạng thị giác quan trọng.
- Class ít dữ liệu bị chia quá nhỏ, khiến mỗi subclass chỉ dựa trên một vài video và dễ overfit.

Do đó bank sử dụng số lượng biến thiên từ 4 đến 9 subclass cho các class abnormal.

## 6. Giới hạn và cách kiểm tra sau này

- UCF-Crime chỉ có nhãn class ở cấp video, không có nhãn subclass. Model phải tự lựa chọn subclass thông qua MIL.
- Một số description có thể chưa phản ánh đầy đủ nội dung video hoặc chứa nhiễu từ annotation gốc.
- Các subclass cùng class vẫn có thể chồng lấn. Ví dụ, một vụ `Robbery Armed` có thể đồng thời xảy ra tại quầy.
- `Other` có thể trở thành lựa chọn quá phổ biến. Cần theo dõi tỷ lệ subclass được chọn trong train và test.
- Nếu một subclass gần như không được chọn, nên kiểm tra lại embedding hoặc gộp nó vào `Other`.
- Nếu `Other` chiếm tỷ lệ quá cao, cần xem lại các video thuộc nhóm này trước khi bổ sung subclass mới.

## 7. Lưu ý khi đưa vào pipeline hiện tại

Bank này cố ý có số subclass khác nhau giữa các class và chỉ giữ một prompt `Normal`. Pipeline subclass đã được mở rộng bằng padding và mask để xử lý các nhóm có độ dài khác nhau mà không tạo subclass giả.

Thứ tự xử lý được chọn để gần VadCLIP gốc:

1. Tại mỗi snippet, gộp các alignment score của subclass cùng class bằng Max hoặc Soft aggregation.
2. Thu được một chuỗi score theo thời gian cho mỗi class gốc.
3. Temporal Top-K chạy trên chuỗi class score này, giống cách VadCLIP gốc chạy Top-K trên `logits2`.

Như vậy các snippet mạnh có thể được đóng góp bởi những subclass khác nhau. Pipeline không chọn Top-K độc lập cho từng subclass rồi mới gộp ở cấp video.

Pipeline training thực hiện:

1. Pad các nhóm subclass về số lượng lớn nhất trong batch cấu hình.
2. Tạo mask đánh dấu subclass thật và vị trí padding.
3. Loại padding khỏi Temporal Top-K, Max aggregation và Soft aggregation.
4. Giữ riêng `Normal` với đúng một prompt hợp lệ.
5. Ghi lại tần suất lựa chọn từng subclass để kiểm tra `Other` có chi phối kết quả hay không.
