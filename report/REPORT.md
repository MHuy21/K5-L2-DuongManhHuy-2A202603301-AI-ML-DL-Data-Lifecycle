# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 11/06/2026**

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Nói về **cả bức ảnh**. Record không có `bbox`/`polygon` hay bất kỳ tọa độ nào chỉ ra vị trí của một
   vật thể cụ thể — nó chỉ gắn một `class_name`/`score` duy nhất cho toàn bộ `sample_id = traffic`. Đây
   đúng là đặc điểm của classification: một nhãn cho toàn ảnh, không phải nhãn cho từng vật thể.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Đi kèm taxonomy/checkpoint. Record ghi rõ `taxonomy_name = "ImageNet-1K"` và `model_file = "yolo11n-cls.pt"` — model chỉ xếp hạng trong danh sách 1000 lớp cố định của ImageNet-1K đi kèm
   checkpoint này, không tự bịa ra tên lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   `class_id` là số định danh cố định, tránh nhầm lẫn khi các taxonomy khác nhau có thể trùng tên lớp
   hoặc đổi cách viết tên lớp. `class_name` giúp người đọc hiểu nhanh mà không cần tra bảng ID. `taxonomy_name`
   cho biết `class_id`/`class_name` này thuộc bộ danh sách lớp nào (ở đây là ImageNet-1K, khác với
   COCO-80 dùng cho detection/segmentation) — thiếu trường này dễ tra nhầm nguồn khi hệ thống dùng nhiều
   taxonomy song song.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Ảnh `traffic` có rất nhiều loại xe khác nhau (xe buýt xanh, taxi, xe cảnh sát, xe con...), và top-5
   của model cũng cho thấy điều đó (`cab` 0.51, `minibus` 0.16, `police_van` 0.09, `recreational_vehicle`
   0.05, `streetcar` 0.05 — không có lớp nào áp đảo). Guideline cần quy định rõ **tiêu chí ưu tiên** khi
   ảnh có nhiều chủ đề, ví dụ: chọn vật chiếm diện tích/tiền cảnh lớn nhất, hoặc vật nằm gần tâm ảnh, hoặc
   vật là chủ thể chính theo ý đồ chụp — để mọi annotator chọn cùng một lớp nhất quán thay vì tùy ý.

6. Vì sao không được lấy `score` của model làm ground truth?

   `score` chỉ là độ tin của model khi xếp hạng, không phải xác nhận của con người theo guideline. Bằng
   chứng cụ thể: `score` của top-1 (`cab`) chỉ là 0.51 — model không chắc chắn, và nhìn ảnh thật thì vật
   nổi bật nhất lại là các xe buýt xanh chứ không rõ ràng là "cab" (taxi). Nếu lấy thẳng top-1 làm ground
   truth, một prediction sai hoặc không chắc chắn sẽ bị coi là nhãn đúng mà không ai kiểm tra lại.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Góc trên-trái của box nằm ở pixel (385.33, 69.24) — tức cách mép trái ảnh 385px, cách mép trên 69px.
   Góc dưới-phải nằm ở pixel (498.92, 348.92). Box rộng 113.58px và cao 279.68px — một hình chữ nhật đứng
   (cao hơn rộng nhiều), ôm trọn người đang đứng quay lưng trong ảnh `kitchen`.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Theo output ô so sánh ngưỡng trong notebook (sample `kitchen`):
   - Ngưỡng 0.20 → **17** prediction: `['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']`
   - Ngưỡng 0.60 → **6** prediction: `['person', 'bowl', 'bowl', 'oven', 'oven', 'person']`
   - (Đối chiếu: ngưỡng 0.35 dùng để tạo `detection_predictions.json` → 11 prediction, đúng bằng số
     record của sample `kitchen` trong file JSON.)

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Khi hạ từ 0.60 xuống 0.20, số box tăng từ 6 lên 17 — thêm 11 box. Trong đó có 5 box score thấp của lớp
   đã quen thuộc (`bowl`, `cup`), và 6 box thuộc lớp mới chỉ xuất hiện ở ngưỡng rất thấp (`spoon` ×3,
   `potted plant`, `dining table`, `bottle`) — riêng `dining table` xuất hiện trong ảnh bếp (`kitchen`)
   và nhiều `spoon` cùng lúc là khá đáng ngờ, nhiều khả năng là false positive vì score quá thấp. Reviewer
   phải kiểm tra từng box mới này, loại bỏ box sai lớp/trùng lặp, nên tốn nhiều công QC hơn hẳn so với
   ngưỡng cao.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Box phải ôm sát toàn bộ phần **nhìn thấy được** của vật thể — không cắt vào vật và không thừa nhiều nền
   xung quanh; nếu vật bị vật khác che một phần, box chỉ bao quanh phần còn nhìn thấy, không ước lượng
   phần bị che.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Record `oven` thứ nhất có `bbox_xyxy = [0.11, 187.54, 195.96, 292.81]` — `x_min` gần bằng 0, tức vật bị
   cắt sát mép trái của ảnh (nhìn trên PNG cũng thấy lò nướng bên trái bị khung ảnh cắt mất một phần).
   Đây là tình huống đủ **phổ biến** (vật nằm sát biên ảnh) nên viết thành **rule chung** trong guideline,
   ví dụ: "nếu vật bị cắt bởi mép ảnh, vẽ box đến đúng mép ảnh, không ước lượng phần nằm ngoài khung hình" —
   không cần escalation vì đây không phải ca hiếm hay mơ hồ cần hỏi thêm.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735744 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`=[53,344], [52,345], [50,345], [49,346], [48,346]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Box chỉ cho một hình chữ nhật bao quanh (4 số `x_min,y_min,x_max,y_max`), không nói gì về hình dạng
   bên trong. Polygon gồm hàng chục điểm (`polygon_point_count=67` cho record trên) đi sát theo **đường
   biên thật** của vật thể — mô tả được phần lồi lõm, góc nghiêng, hình dạng bất quy tắc của cái bát, mà
   box không thể hiện được.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   `instance_id` dùng để **phân biệt từng vật thể riêng lẻ** trong output của bài lab, dạng
   `<sample_id>-NNN` tăng dần theo thứ tự phát hiện. Bằng chứng: sample `kitchen` có 4 instance khác nhau
   cùng `class_id=45` ("bowl"): `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010`, và 2 instance
   cùng `class_id=0` ("person"): `kitchen-001`, `kitchen-009` — chứng tỏ nhiều instance có thể trùng
   `class_id` nhưng luôn khác `instance_id`, nên nó **không phải class ID**. Nó cũng **không phải tracking
   ID** vì notebook chỉ xử lý từng ảnh tĩnh độc lập, không có chuỗi nhiều khung hình/video để theo dõi một
   vật thể qua thời gian — `instance_id` chỉ có ý nghĩa trong phạm vi một ảnh, một lần chạy của bài lab.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Biên polygon phải đi sát theo rìa thật của vật thể (không nới rộng vào nền, không cắt hụt vào vật); nếu
   vật bị vật khác che một phần, chỉ vẽ phần còn nhìn thấy được, không ước lượng phần bị che khuất.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Trên `segmentation_prediction.png`, cụm 3 cái bát ở góc dưới-trái bàn (`bowl 0.74`, `bowl 0.70`, `bowl 0.53`) có mask chạm/chồng lên nhau rất sát, khó phân định ranh giới chính xác giữa từng cái. Đây là
   tình huống đủ **phổ biến** (vật thể cùng loại xếp sát nhau) nên viết thành **rule chung** trong
   guideline (ví dụ: "khi hai vật cùng lớp chạm nhau, vẽ biên theo điểm tiếp xúc thật quan sát được, nếu
   không phân biệt được ranh giới thì đánh dấu để reviewer kiểm tra lại"), không cần escalation vì không
   phải ca hiếm/đặc biệt.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                                             | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                                                                   | Annotator làm gì?                                                                                                           | Reviewer xem gì?                                                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một `class_id`/`class_name` duy nhất cho toàn bộ ảnh, theo taxonomy cố định (ImageNet-1K) | Ảnh `traffic` có nhiều loại xe (bus, minibus, police_van...), top-1 model chỉ đạt score 0.51 và chọn "cab" dù xe buýt xanh mới là vật nổi bật nhất trong ảnh — khó quyết định đâu là "chủ thể chính" | Chọn đúng **một** lớp theo tiêu chí ưu tiên trong guideline (vd. vật chiếm diện tích/tiền cảnh lớn nhất), không tự suy diễn | Kiểm tra lớp được chọn có đúng taxonomy và đúng tiêu chí ưu tiên không; không dùng `score` của model làm căn cứ chất lượng               |
| Phát hiện vật thể     | Một `class_name` + một `bbox_xyxy` (pixel) cho mỗi vật thể riêng lẻ                       | `oven` bên trái ảnh `kitchen` bị cắt sát mép ảnh (`x_min=0.11`); khi hạ ngưỡng xuống 0.20 xuất hiện thêm các lớp đáng ngờ (`dining table`, 3×`spoon`) nhiều khả năng là false positive               | Vẽ box ôm sát phần nhìn thấy được của từng vật, kể cả vật bị cắt mép (vẽ tới đúng mép ảnh, không ước lượng)                 | Kiểm tra box có ôm sát không, rà soát các prediction score thấp để loại false positive/trùng lặp, xem có bỏ sót vật không                |
| Instance segmentation | Một `class_name` + `bbox` + `polygon_xy` cho mỗi `instance_id` riêng, kể cả khi trùng lớp | Cụm 3 cái bát ở góc dưới-trái bàn (`bowl 0.74/0.70/0.53`) có mask chạm/chồng lên nhau, khó phân định ranh giới chính xác giữa từng cái                                                               | Vẽ polygon sát biên thật của từng instance, tách riêng từng vật dù cùng lớp và nằm sát nhau, không gộp hai vật thành một    | Kiểm tra polygon có đúng biên không, các instance chồng nhau có bị gộp nhầm thành một không, `instance_id` có gán đúng/không trùng không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Chỉ dùng 3 ảnh COCO công khai đã được notebook cố định và kiểm tra checksum (SHA-256), không tải lên
   Colab/GitHub bất kỳ ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ/khách hàng nào.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Dừng ngay việc chạy/tải dữ liệu đó và báo cho **Lab Coach/GV** qua kênh hỗ trợ của lớp, không tự ý xử
   lý hay tiếp tục sử dụng dữ liệu khi chưa có xác nhận.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
