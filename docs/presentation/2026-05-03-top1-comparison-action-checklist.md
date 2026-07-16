# 2026-05-03 Top1 Comparison and Action Checklist

## Mục tiêu

Tài liệu này tổng hợp:

- bài học rút ra từ deck top 1 năm trước [REFINE-XRAY.pdf](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/REFINE-XRAY.pdf)
- nhận xét trực tiếp của thầy Hoàng về cách làm nổi bật contribution
- action checklist rất cụ thể theo từng slide cho deck hiện tại [Flow is All You Need for WiFi.pdf](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/Flow%20is%20All%20You%20Need%20for%20WiFi.pdf)

Mục tiêu sau cùng:

> Biến deck WiFi từ một bài có nhiều ý hay thành một bài mà BGK nhìn vào là thấy rõ `M0-M4` khác nhau ở đâu, contribution nằm ở đâu, và vì sao nó xứng đáng điểm cao.

---

## 1. Bài học rút ra từ đội top 1

### 1.1. Điểm mạnh của deck top 1

1. Họ mở bài bằng pain point đời thật, không mở bằng mô hình.
2. Họ cho BGK “map để chấm” từ rất sớm: agenda rõ, contribution rõ, result rõ.
3. Họ tạo cảm giác nghiên cứu chín:
   - related work
   - training strategy
   - SOTA comparison
   - ablation
   - demo
   - limitations và future work
4. Họ kết bài theo kiểu trưởng thành:
   - chúng tôi đã tích hợp gì
   - đạt được gì
   - giới hạn ở đâu
   - sẽ đi tiếp như thế nào
5. Visual của họ giúp BGK dễ theo:
   - section divider rõ
   - hình sạch
   - ý tưởng mới được nhấn bằng visual, không chỉ bằng lời nói

### 1.2. Tiêu chí BGK có khả năng đã chấm cao

- Bài toán có cấp thiết thật
- Contribution dễ nhìn thấy và dễ chấm
- Kết quả đáng tin
- Tư duy nghiên cứu trưởng thành
- Team biết mình giới hạn ở đâu

---

## 2. Thông điệp của thầy Hoàng

Nội dung cốt lõi của thầy Hoàng:

> Chủ yếu là so ra được sự khác nhau của `M0-M4` bằng công thức, feature, hoặc hình ảnh. Như vậy mới làm nổi bật được cái mình contribute. Nhiều nhóm vào vòng trong làm gì cũng có nhưng đều qua qua, không đánh trúng trọng tâm, không có công thức và visual đủ sức để BGK hiểu.

Ý nghĩa trực tiếp:

- Không được kể chung chung “thêm module A, module B”.
- Phải cho BGK thấy `M0`, `M1`, `M2`, `M3`, `M4` khác nhau thế nào.
- Mỗi thay đổi quan trọng phải được “chứng minh” bằng ít nhất một trong ba cách:
  - công thức
  - feature / pipeline
  - hình ảnh / qualitative visual

---

## 3. Quy ra 5 nguyên tắc sửa slide từ nhận xét của thầy Hoàng

### Nguyên tắc 1. Mỗi stage phải có “delta” rõ ràng so với stage trước

BGK không nên phải tự nói trong đầu:

- `M1` khác `M0` chỗ nào?
- `M2` khác `M1` chỗ nào?
- `M3/M4` khác direct regression chỗ nào?

Mỗi slide contribution phải tự trả lời câu hỏi đó.

Áp dụng:

- Slide 10 phải định nghĩa `M0`
- Slide 13 phải tóm tắt delta `M0 -> M4`
- Slide 14 phải chỉ rõ `M1` đổi representation
- Slide 15 phải chỉ rõ `M2` đổi sequence modeling
- Slide 16/24 phải chỉ rõ `M3/M4` đổi prediction objective

### Nguyên tắc 2. Mỗi contribution phải có một “bằng chứng nhìn thấy được”

Với mỗi contribution, phải có ít nhất một loại bằng chứng:

- công thức
- feature/pipeline
- hình ảnh before/after

Áp dụng:

- Spectral Tokenizer: feature pipeline + “emphasize motion, suppress static noise”
- WiMamba: công thức / complexity `O(L^2) -> O(L)`
- Rectified Flow: công thức path / velocity / flow loss
- Result: hình qualitative + bảng số liệu

### Nguyên tắc 3. Main deck phải đánh trúng trọng tâm, không kể ngang hàng mọi thứ

Deck không được rơi vào tình trạng:

> Cái gì nhóm cũng làm, cái gì cũng nhắc, nhưng không có cái nào đập vào mắt BGK.

Phải ưu tiên:

- baseline
- 3 thay đổi chính
- 2 kết luận result chính

Áp dụng:

- giảm nói chi tiết low-level ở slide 8 và 18
- tăng lực cho slide 13, 15, 16, 22, 24

### Nguyên tắc 4. Công thức toán và visual phải giải thích contribution, không phải để trang trí

Công thức chỉ có giá trị nếu nó giúp BGK hiểu:

- cái mới là gì
- tại sao hợp lý
- tại sao tốt hơn baseline

Áp dụng:

- không đưa công thức FFT dài dòng nếu không phục vụ thông điệp
- ưu tiên công thức nào cho thấy “cơ chế thay đổi”
- ưu tiên visual before/after hơn là text dài

### Nguyên tắc 5. Appendix phải là bộ phòng thủ kỹ thuật, không phải kho chứa slide thừa

Appendix của mình phải giúp trả lời ngay các câu hỏi:

- Tại sao Mamba mà không phải Transformer?
- Tại sao Flow tốt hơn Direct Regression?
- Cái này đã customize cho WiFi như thế nào?
- Tại sao model này hợp edge deployment?

Appendix phải được thiết kế như “bộ phòng thủ contribution”, không chỉ là backup ngăn xếp.

---

## 4. Nguyên tắc cập nhật deck WiFi

### 4.1. Không làm

- Không biến deck thành bản sao của top 1.
- Không thêm một section related work dài vào main deck.
- Không nhét nhiều công thức vào main talk nếu không làm rõ contribution hơn.
- Không mở rộng main deck nếu timing bị vỡ.

### 4.2. Nên làm

- Tăng lực mở bài bằng tính cấp thiết thật hơn.
- Đưa contribution lên dễ chấm hơn.
- Làm nổi sự khác nhau `M0-M4` bằng công thức, feature, hoặc hình.
- Tăng cảm giác research maturity ở kết bài.
- Chuẩn bị appendix phòng thủ kỹ thuật mạnh.

---

## 5. P0 - Việc bắt buộc phải làm trước khi dùng deck

1. Re-export PDF sạch
   - đổi toàn bộ `MO` thành `M0`
   - xóa slide duplicate `Why flow helps beyond direct regression`

2. Sửa copy gây mất điểm
   - `In summarize` -> `In summary`
   - `MO-M4` -> `M0-M4`

3. Đồng bộ file export với docs
   - [phase4_full_slide_detail.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase4_full_slide_detail.md)
   - [phase5_speaker_script.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase5_speaker_script.md)

---

## 6. Action Checklist Theo Từng Slide

Ghi chú:

- `P0` = bắt buộc sửa
- `P1` = rất nên sửa
- `P2` = nếu còn thời gian

### Main Deck

| Slide | Tình trạng hiện tại | Sửa gì | Theo 5 nguyên tắc của thầy Hoàng | Priority |
|---|---|---|---|---|
| 1 | Title slide ổn | Giữ full title `Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures`; nếu cần subtitle nhỏ, dùng `Two-step pose flow refinement from WiFi CSI` | Chốt thesis ngay từ đầu | P2 |
| 2 | Agenda rõ | Thêm 1 dòng nhỏ hoặc speaker line: `3 contributions: spectral representation, efficient sequence modeling, flow refinement` | Cho BGK “map để chấm” sớm | P1 |
| 3 | Divider ổn | Giữ | Nhịp section rõ | Keep |
| 4 | Motivation tốt nhưng còn tổng quát | Thêm 1 pain point thật hơn: bedroom, rehab, elderly care, private indoor sensing | Tăng urgency, đánh vào bài toán thật | P1 |
| 5 | Research question tốt | Giữ. Nếu cần, giảm text để câu hỏi nổi bật hơn | Hook rõ hơn | P2 |
| 6 | Application framing ổn | Thêm 1 câu kết: `not replacing cameras everywhere, but enabling sensing where cameras are inappropriate` | Tăng maturity và cân bằng narrative | P1 |
| 7 | Divider ổn | Giữ | Chuyển nhóm ý sạch | Keep |
| 8 | System overview mạnh nhưng dày | Khi sửa, giảm nói low-level. Trong lúc thuyết trình, chỉ nói 3 tầng: sensing -> model -> application | Main deck đánh trúng trọng tâm | P1 |
| 9 | Divider ổn | Giữ | Chuyển sang baseline | Keep |
| 10 | PDF vẫn còn `MO` | Sửa `MO` -> `M0`. Thêm callout `strong PETR-style baseline` | Định nghĩa baseline thật rõ để về sau so delta | P0 |
| 11 | Nội dung đúng hướng | Giữ trục `final coordinates, not correction trajectory`. Nếu sửa text, làm rõ matching là assignment step | Làm rõ M0 học cái gì | P2 |
| 12 | Divider ổn | Giữ | Đánh dấu contribution block | Keep |
| 13 | Ladder rất quan trọng nhưng vẫn còn `MO-M4` | Sửa `M0-M4`. Tô đậm 3 trụ: `representation`, `efficiency`, `refinement`. Nếu đủ chỗ, thêm cột `What changed from previous stage?` | Đây là slide trung tâm để so ra sự khác nhau của M0-M4 | P0 |
| 14 | Spectral Tokenizer slide ổn | Thêm 1 dòng nhấn ý nghĩa: `emphasize motion, suppress static noise`. Nếu có thể, thêm hình before/after token flow | Contribution phải có feature/pipeline bằng chứng | P1 |
| 15 | WiMamba slide tốt | Thêm callout lớn hơn: `Transformer O(L^2) vs WiMamba O(L)`. Thêm label `edge-friendly inference` | Contribution phải có công thức / complexity để thấy ngay tại sao khác | P1 |
| 16 | Flow intuition ổn | Thêm nhãn rõ `draft pose`, `correction velocity`, `refined pose` | Chỉ rõ M3/M4 đổi prediction process, không chỉ thêm head | P1 |
| 17 | M4 pipeline mạnh nhưng dày | Highlight novelty path bằng màu khác. Đảm bảo BGK nhìn 3 giây là biết “phần mới nằm ở đây” | Pipeline phải giúp nhìn thấy contribution | P1 |
| 18 | Training pipeline rất mạnh nhưng nặng | Giảm clutter nếu có thể. Thêm rule đọc slide: `upper = forward architecture, lower = training and losses` | Công thức/visual phải giải thích, không được gây quá tải | P1 |
| 19 | Divider ổn | Giữ | Mở block kết quả | Keep |
| 20 | Dataset slide cần kiểm tra lại độ rõ | Đảm bảo có 3 ý rõ: source, 1p/2p/3p, annotation | Tăng độ chín nghiên cứu | P1 |
| 21 | MPJPE slide đúng | Rút text nếu dài. Làm rõ `lower is better` | Metric để BGK đọc kết quả nhanh | P2 |
| 22 | Quant slide là slide quan trọng nhất | Giữ trục `M3 = accuracy winner`, `M4 = deployment winner`. Bỏ wording có mùi “thắng mọi thứ”. Nếu cần, thêm note `preliminary local benchmark` | Đây là slide kết luận contribution bằng số liệu | P1 |
| 23 | Qualitative slide tốt | Khoanh đúng 1-2 failure case của `M0` được sửa bởi `M3/M4` | Contribution phải có bằng chứng bằng hình ảnh | P1 |
| 24 | Slide intuition cực mạnh | Giữ. Thêm 1 dòng `learn correction path, not one-shot jump` nếu chưa nổi bật | Đây là slide formula + intuition để BGK nhớ contribution | P1 |
| 25 | Duplicate slide | Xóa hẳn | Tránh mất điểm polish | P0 |
| 26 | Summary hiện là metric recap | Đổi title thành `In summary` hoặc `Three takeaways`. Chia 3 ý: `what changed`, `what improved`, `what remains next` | Kết bài theo kiểu research maturity như top 1 | P0 |
| 27 | Thanks slide ổn | Giữ | Kết gọn | Keep |
| 28 | Appendix divider | Giữ | Tách main và backup | Keep |
| 29 | Appendix chưa dùng đúng mục đích | Đổi thành `Why Mamba, not Transformer?` | Appendix = bộ phòng thủ kỹ thuật | P1 |
| 30 | Mega technical backup | Giữ làm `full technical backup` | Dùng cho Q&A sau khi phòng thủ xong 2 slide chính | P2 |
| 31 | Appendix chưa phát huy hết | Đổi thành `Why Rectified Flow, not Direct Regression?` | Đây là slide phòng thủ quan trọng nhất của contribution | P1 |
| 32 | Appendix cuối | Đổi thành `Edge deployment / Limitations / Future Work` nếu có thể | Tăng research maturity và practical impact | P1 |

---

## 7. Các slide nên thêm hoặc thay ở appendix

### 7.1. Bắt buộc nên có

1. `Why Mamba, not Transformer?`
   - `Attn(Q,K,V) = softmax(QK^T / sqrt(d))V`
   - `QK^T` là ma trận `L x L`
   - suy ra compute và memory tăng theo `O(L^2)`
   - contrast với SSM:
     - `h_t = A_t h_(t-1) + B_t x_t`
     - `y_t = C_t h_t + D x_t`
   - chốt:
     - Transformer = pairwise quadratic interactions
     - Mamba = linear recurrent state updates
     - phù hợp edge deployment hơn

2. `Why Rectified Flow, not Direct Regression?`
   - direct regression:
     - `x_hat = f_theta(c)`
   - flow path:
     - `x_t = (1 - t)x_0 + tx_1`
     - `v_theta(x_t, t, c) ~= x_1 - x_0`
     - `L_flow = E ||v_theta - (x_1 - x_0)||^2`
   - nhấn mạnh:
     - không generate từ pure image noise
     - bắt đầu từ `draft pose`
     - học `correction trajectory`
     - customize cho WiFi bằng `CSI-conditioned pose refinement`

3. `Edge Deployment / Limitations / Future Work`
   - tại sao `O(L)` quan trọng với edge
   - tại sao low memory quan trọng với router / gateway
   - limitations:
     - local benchmark snapshot
     - cần thêm validation
   - next:
     - stronger edge deployment
     - more robust multi-person scenes

### 7.2. Có thể thêm nếu còn thời gian

4. `Related Work Positioning`
   - PETR-style direct regression baselines
   - efficient sequence modeling
   - flow-based refinement

5. `Why We Trust the Redesign`
   - controlled ablation
   - qualitative matched poses
   - draft-to-refine intuition

---

## 8. Thứ tự thực thi để sửa deck

### P0 - Làm ngay

1. Re-export PDF sạch
2. Sửa `M0`
3. Xóa duplicate slide 25
4. Sửa `In summarize`
5. Sửa slide 13 cho rõ `M0-M4`
6. Sửa slide 26 theo kiểu conclusion trưởng thành hơn

### P1 - Làm tiếp nếu còn 1 buổi chỉnh

7. Tăng urgency ở slide 4 hoặc 6
8. Tăng contribution map ở slide 2 hoặc 13
9. Tối ưu slide 8, 18, 20, 22, 23, 24
10. Thêm backup slide `Why Mamba?`
11. Thêm backup slide `Why Rectified Flow?`
12. Thêm backup slide `Edge deployment / limitations / future work`

### P2 - Nếu còn thời gian polish

13. Thêm `Related Work Positioning`
14. Thêm `Why we trust the redesign`
15. Tinh visual highlight cho novelty path ở slide 17
16. Rút text ở slide 21

---

## 9. Kết luận chốt cho cả nhóm

Điều thầy Hoàng nhấn mạnh và đội top 1 năm trước làm tốt nhất là:

- contribution phải dễ nhìn thấy
- sự khác nhau giữa các stage phải dễ so ra
- công thức và visual phải phục vụ việc giải thích contribution
- deck phải đánh trúng trọng tâm, không nói ngang hàng quá nhiều thứ

Deck WiFi của chúng ta đã có lõi kỹ thuật mạnh hơn ở:

- `M0-M4`
- `WiMamba`
- `Rectified Flow`

Để tiến gần giải nhất, việc cần làm tiếp theo không phải đổi toàn bộ câu chuyện, mà là:

- tăng lực mở bài
- làm rõ delta `M0-M4`
- biến mỗi contribution thành thứ BGK nhìn thấy được
- tăng sức phòng thủ kỹ thuật ở appendix
- kết bài theo kiểu nghiên cứu trưởng thành hơn
