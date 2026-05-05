# Kế hoạch bổ sung slide cho WiMamba: Temporal Mamba + Bi-Mamba

## 1. Mục tiêu của lần bổ sung này

Mục tiêu là sửa phần kể chuyện của `M2` để ban giám khảo nhìn ra rõ hơn:

1. Nhóm không chỉ "thay Transformer bằng Mamba".
2. Nhóm thực sự đề xuất một kiến trúc `Factorized WiMamba` gồm hai pha riêng:
   - `Temporal Mamba`
   - `Bidirectional Spatial Mamba (Bi-Mamba)`
3. Phần `Bi-Mamba` phải có một slide riêng giải thích rõ:
   - input tensor là gì
   - output tensor là gì
   - vì sao phải chạy hai chiều
   - vì sao phù hợp với dữ liệu WiFi CSI hơn attention toàn cục

Nói ngắn gọn: `M2` phải được trình bày như một contribution có thiết kế riêng cho WiFi, chứ không phải chỉ là "thử Mamba".

---

## 2. Kết quả đọc code

### 2.1. Contribution nằm ở đâu trong code

Phần contribution của `M2` nằm chủ yếu ở:

- `opera/models/backbones/wimamba.py`
- `configs/wifi/petr_wifi_mamba.py`
- `opera/models/dense_heads/wi_tidar_head.py`

### 2.2. Điều code thực sự đang làm

Trong [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py), mỗi `FactorizedWiMambaBlock` không phải một Mamba duy nhất, mà gồm hai giai đoạn:

1. `Temporal Mamba`
2. `Bidirectional Spatial Mamba`

Ở mức code:

- `mamba_t` được khai báo ở [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:29)
- `mamba_s_fwd` và `mamba_s_bwd` được khai báo ở [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:37)

Về logic forward:

- Input block có shape `x in R^(B x S x T x C)` tại [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:50)
- Temporal pass reshape thành `(B*S, T, C)` tại [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:55)
- Spatial pass transpose và reshape thành `(B*T, S, C)` tại [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:64)
- Bi-Mamba spatial chạy:
  - forward direction ở [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:67)
  - backward direction bằng `flip -> Mamba -> flip back` ở [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:69)
- Hai nhánh spatial được cộng lại tại [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:73)

### 2.3. Mô tả chính xác contribution

Từ code, cách gọi đúng nhất là:

> Chúng tôi đề xuất một `Factorized WiMamba encoder` cho WiFi CSI token sequence, trong đó mỗi block tách riêng:
> `Temporal Mamba` để mô hình hóa động học theo thời gian,
> và `Bidirectional Spatial Mamba` để mô hình hóa quan hệ không gian hai chiều giữa các nhóm anten tại mỗi thời điểm.

Điểm quan trọng:

- đây không phải một Mamba "nguyên khối"
- đây cũng không phải Bi-Mamba cho toàn bộ chuỗi `L=180`
- đây là Bi-Mamba chỉ trên trục không gian `S=9`, sau khi đã tách temporal và spatial

Đây chính là chỗ nên nhấn mạnh là `đề xuất của nhóm`.

### 2.4. Input/output của toàn encoder

Trong [WiTiDARHead.forward](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/dense_heads/wi_tidar_head.py:149), head nhận `feat` có shape `(B, L, C)` rồi:

1. permute thành `(L, B, C)`
2. truyền vào `WiMambaEncoder`
3. bên trong encoder permute/reshape lại thành `(B, S, T, C)`
4. sau nhiều block, reshape về `(B, L, C)`
5. final output của encoder trả về `(L, B, C)`
6. head permute lại thành `memory in R^(B x L x C)`

Vì vậy nếu vẽ slide ở mức hệ thống:

- `Encoder input`: `X in R^(B x 180 x 256)`
- `Factorized view`: `X in R^(B x 9 x 20 x 256)`
- `Encoder output`: `M in R^(B x 180 x 256)`

Đây là điểm rất tốt để giải thích "input feature -> factorized processing -> output feature".

---

## 3. Kết luận narrative nên dùng trên slide

### 3.1. Câu cũ chưa đủ mạnh

Nếu chỉ nói:

> We replace Transformer with WiMamba for linear complexity.

thì vẫn đúng, nhưng quá yếu. Câu này làm contribution bị hiểu thành:

- thay mô hình backbone
- tối ưu tốc độ

trong khi code cho thấy contribution thật mạnh hơn:

- nhóm tự thiết kế một khối factorized theo cấu trúc của dữ liệu WiFi
- temporal và spatial được xử lý tách rời
- spatial còn chạy hai chiều

### 3.2. Câu nên dùng

Nên đổi sang:

> Instead of full pairwise attention, we propose a factorized WiMamba encoder for WiFi CSI: a `Temporal Mamba` for motion dynamics and a `Bidirectional Spatial Mamba` for two-way antenna-group interaction.

Nếu cần bản ngắn hơn:

> Our `M2` contribution is not just Mamba replacement, but a WiFi-specific factorization: Temporal Mamba + Bi-Mamba Spatial pass.

---

## 4. Đề xuất cấu trúc slide

## Phương án khuyến nghị

Nên tách phần `M2` thành **3 lớp trình bày**:

1. `Slide A`: overview của `Stage 2`
2. `Slide B`: `Temporal Mamba`
3. `Slide C`: `Bi-Mamba` với input/output chi tiết

Lý do:

- `Slide A` giúp BGK hiểu vì sao bỏ Transformer
- `Slide B` giúp BGK thấy đây không phải một khối đen "Mamba"
- `Slide C` giúp chốt contribution riêng của nhóm ở phần không gian hai chiều

Nếu thời lượng rất căng, có thể rút còn 2 slide:

1. `Overview + Temporal`
2. `Bi-Mamba chi tiết`

Nhưng phương án 3 lớp vẫn tốt nhất nếu mục tiêu là "đánh trúng contribution".

---

## 5. Slide A - Stage 2 overview

### Mục tiêu

Cho BGK thấy:

- `M2` thay gì so với `M1`
- thay đổi này phục vụ mục tiêu gì
- tại sao phù hợp cho edge deployment

### Title gợi ý

`Stage 2: Factorized WiMamba replaces quadratic attention`

### Nội dung chính

Trái:

- `Transformer`
- full pairwise attention
- `O(L^2)` memory/compute scaling

Phải:

- `Factorized WiMamba`
- temporal pass + bidirectional spatial pass
- `O(L)` sequence scaling

### 1 câu chốt dưới slide

> `M2` changes the sequence model, not the prediction objective: direct regression is still kept, while the encoder is redesigned for linear-time WiFi sequence modeling.

### Câu speaker note nên nói

> Ở giai đoạn này, contribution không phải nằm ở output head mà nằm ở encoder. Chúng em thay self-attention toàn cục bằng một WiMamba factorized, để vừa giữ khả năng mô hình hóa chuỗi, vừa hướng đến triển khai edge AI.

---

## 6. Slide B - Temporal Mamba

### Mục tiêu

Chỉ ra rõ:

- phần temporal là một mô-đun riêng
- nó chạy trên trục thời gian `T`
- mỗi spatial group được xử lý độc lập theo thời gian

### Title gợi ý

`Stage 2a: Temporal Mamba models motion along time`

### Điều cần vẽ

Nên vẽ tensor:

```text
Input feature map: X in R^(B x S x T x C)
Reshape temporal view: X_t in R^((B*S) x T x C)
Temporal Mamba -> H_t in R^((B*S) x T x C)
Reshape back -> R^(B x S x T x C)
```

### Công thức nên dùng

```latex
\tilde{X}_t = \operatorname{reshape}(\operatorname{LN}(X), (B S)\times T \times C)
```

```latex
H_t = \operatorname{Mamba}_t(\tilde{X}_t)
```

```latex
X' = X + \operatorname{reshape}(H_t, B \times S \times T \times C)
```

### Ý nghĩa nên ghi rất ngắn trên slide

- scan theo thời gian cho từng nhóm anten
- học motion dynamics thay vì attention toàn cục
- giữ nguyên output shape để nối sang spatial stage

### Câu speaker note

> Temporal Mamba của chúng em không trộn lẫn toàn bộ token ngay từ đầu. Nó trước hết xem mỗi nhóm anten như một chuỗi thời gian riêng, rồi học động học chuyển động theo trục thời gian.

---

## 7. Slide C - Bi-Mamba chi tiết

### Đây là slide bắt buộc nên thêm

Đây là slide quan trọng nhất của phần bổ sung này.

### Mục tiêu

Làm rõ:

1. `Bi-Mamba` là đề xuất của nhóm
2. input cho spatial pass là gì
3. output của spatial pass là gì
4. vì sao phải chạy hai chiều

### Title gợi ý

`Stage 2b: Bidirectional Spatial Mamba for two-way antenna interaction`

### Bố cục khuyến nghị

#### Cột trái: input/output tensor

Hiển thị thật rõ:

```text
Input to Bi-Mamba:
X' in R^(B x S x T x C)

After transpose/reshape:
X_s in R^((B*T) x S x C)
```

Giải thích ngay bên dưới:

- mỗi thời điểm `t` trở thành một chuỗi theo trục không gian
- chuỗi này chạy qua `S = 9` nhóm Rx-Tx

Sau đó hiển thị:

```text
Output of Bi-Mamba:
H_s in R^((B*T) x S x C)
Reshape back:
Y in R^(B x S x T x C)
```

#### Cột giữa: sơ đồ forward/backward

Vẽ 2 nhánh:

```text
Forward spatial pass:  left -> right
Backward spatial pass: right -> left
```

Rồi cộng:

```text
H_s = H_s^fwd + H_s^bwd
```

#### Cột phải: trực giác

3 bullet ngắn:

- captures left-to-right and right-to-left antenna dependencies
- avoids order bias of one-way scanning
- preserves linear sequence modeling

### Công thức nên dùng

```latex
\tilde{X}_s = \operatorname{reshape}\!\left(\operatorname{transpose}_{S,T}(\operatorname{LN}(X')), (B T)\times S \times C\right)
```

```latex
H_s^{\rightarrow} = \operatorname{Mamba}_{s,\mathrm{fwd}}(\tilde{X}_s)
```

```latex
H_s^{\leftarrow} = \operatorname{flip}\!\left(\operatorname{Mamba}_{s,\mathrm{bwd}}(\operatorname{flip}(\tilde{X}_s))\right)
```

```latex
Y = X' + \operatorname{reshape}\!\left(H_s^{\rightarrow} + H_s^{\leftarrow}, B \times S \times T \times C\right)
```

### Câu insight nên để trong box nổi bật

> Bi-Mamba is applied on the spatial axis after temporal encoding, so each time step aggregates two-way antenna-group interactions without constructing an all-pairs attention matrix.

### Câu speaker note

> Đây là điểm quan trọng nhất. Sau khi đã học động học theo thời gian, chúng em không dùng attention toàn cục trên toàn bộ 180 token, mà tái tổ chức đặc trưng theo từng thời điểm để chạy Mamba hai chiều trên 9 nhóm không gian. Nhờ vậy mô hình vừa thấy được tương tác hai chiều giữa các nhóm anten, vừa giữ được độ phức tạp tuyến tính.

---

## 8. Cách nối Bi-Mamba với phần còn lại của pipeline

Sau slide `Bi-Mamba`, nên có một câu pivot rất ngắn:

> After factorized WiMamba, the encoder returns the same feature interface `M in R^(B x L x C)`, so the downstream query decoder and prediction head remain compatible.

Ý này bám sát code:

- output encoder quay lại `(L, B, C)` ở [wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py:158)
- head permute lại thành `memory in R^(B x L x C)` ở [wi_tidar_head.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/dense_heads/wi_tidar_head.py:152)

Điểm này tốt vì giúp BGK thấy:

- encoder được thay
- interface với decoder vẫn nhất quán

---

## 9. Điều nên nhấn mạnh là "đề xuất của nhóm"

Đây là 3 claim nên lặp lại thống nhất:

1. `Temporal Mamba` là thành phần đầu tiên trong factorized WiMamba của nhóm.
2. `Bi-Mamba Spatial pass` là thành phần thứ hai, chạy hai chiều trên trục không gian.
3. Hai phần này được thiết kế theo đúng cấu trúc token WiFi `S x T`, không phải bê nguyên một khối Mamba mặc định.

Nếu muốn nói mạnh hơn nhưng vẫn an toàn:

> Our contribution is a WiFi-structured Mamba factorization, not merely a plug-in replacement of Transformer.

---

## 10. Điều không nên nói

Không nên nói:

- "Mamba xử lý đồng thời cả temporal và spatial trong một bước"
- "Bi-Mamba chạy trên toàn bộ 180 token"
- "Đây là bidirectional attention"
- "Chúng em dùng Mamba nguyên bản cho WiFi"

Vì các câu này không khớp hoàn toàn với code.

---

## 11. Đề xuất chèn vào deck hiện tại

Nếu map vào deck đang có, mình khuyên:

1. Giữ slide overview `Transformer vs WiMamba`
2. Thêm ngay sau đó slide `Temporal Mamba`
3. Thêm tiếp slide `Bi-Mamba input/output`

Trình tự tốt sẽ là:

1. `Why replace Transformer?`
2. `Temporal Mamba`
3. `Bi-Mamba for spatial interaction`
4. `M2 mainly targets deployability for edge AI`

Như vậy BGK sẽ thấy:

- vì sao đổi
- đổi bằng cái gì
- từng phần bên trong làm gì
- contribution nào là của nhóm

---

## 12. Kết luận thực thi

Kết luận sau khi đọc code:

- Có, nên bổ sung hẳn phần `Temporal Mamba` như một contribution được gọi tên rõ ràng.
- Có, rất nên có một slide riêng cho `Bi-Mamba`.
- Slide `Bi-Mamba` phải có:
  - input tensor
  - reshape
  - forward/backward passes
  - output tensor
  - residual return

Nếu chỉ còn thời gian sửa tối thiểu, ưu tiên mạnh nhất là:

1. Giữ 1 slide overview `Transformer -> Factorized WiMamba`
2. Thêm 1 slide riêng `Bi-Mamba input/output`

Nhưng nếu muốn phần `M2` thực sự đủ lực để bảo vệ contribution, phương án tốt nhất vẫn là bộ 3 slide:

1. overview
2. temporal
3. Bi-Mamba

---

## 13. Bộ 3 slide chi tiết để dựng thật

### 13.1. Cách chèn vào deck hiện tại

Nếu áp dụng vào deck hiện tại, nên thay phần `Stage 2` cũ bằng 3 slide liên tiếp:

1. `Stage 2 Overview: Why Factorized WiMamba?`
2. `Stage 2a: Temporal Mamba`
3. `Stage 2b: Bidirectional Spatial Mamba`

Nếu đánh số theo deck hiện tại, cách ổn nhất là:

- giữ `Slide 14 = M1`
- thay `Slide 15` hiện tại bằng `Stage 2 Overview`
- chèn thêm `Stage 2a: Temporal Mamba`
- chèn thêm `Stage 2b: Bi-Mamba`
- các slide sau lùi xuống tương ứng

Như vậy phần `M2` sẽ đủ lực để bảo vệ contribution thay vì chỉ đi qua trong 1 slide.

### 13.2. Visual style thống nhất cho cả 3 slide

Ba slide này nên dùng cùng một ngôn ngữ hình ảnh:

- nền trắng sạch
- thanh header xanh navy
- màu nhấn cyan/xanh dương sáng cho các block Mamba
- đường tensor và mũi tên màu đen/xám đậm
- dùng box bo góc, ít chữ, nhiều sơ đồ
- không dùng dark theme vì sẽ lệch khỏi phần còn lại của deck

Formatting goal nên thống nhất:

> clean scientific slide, white background, navy header bars, cyan technical modules, black tensor arrows, minimal text, strong visual hierarchy, no decorative clutter, consistent with the existing ResFes WiFi deck

---

## 14. Slide 1 trong bộ 3: Overview

### 14.1. Mục tiêu

Slide này trả lời câu hỏi:

- vì sao phải bỏ Transformer
- `M2` thay đổi cái gì
- đóng góp của nhóm không chỉ là “chạy Mamba”

### 14.2. Title đề xuất

> Stage 2 (`M2`): Replace quadratic attention with factorized WiMamba

### 14.3. Key message một dòng

> We replace full pairwise attention with a WiFi-specific factorization: Temporal Mamba + Bidirectional Spatial Mamba.

### 14.4. Layout đề xuất

#### Nửa trái

Card `Transformer`

- Full pairwise attention
- Global token-token interactions
- `O(L^2)` memory/compute scaling

Vẽ biểu tượng mạng dày đặc giữa các token.

#### Nửa phải

Card `Factorized WiMamba`

- Temporal pass
- Bidirectional spatial pass
- `O(L)` sequence scaling

Vẽ 2 tầng:

- tầng trên: các đường ngang cho temporal
- tầng dưới: hai chiều trái-phải và phải-trái cho spatial

#### Dải kết luận cuối slide

> `M2` changes the encoder, not the prediction head. Direct regression is still kept at this stage.

### 14.5. Text đặt trực tiếp lên slide

```text
Transformer
- Full pairwise interactions
- O(L^2) memory/compute
- Heavy growth for long CSI sequences

Factorized WiMamba
- Temporal Mamba
- Bidirectional Spatial Mamba
- O(L) sequence scaling

M2 targets efficient WiFi sequence modeling while keeping the prediction objective unchanged.
```

### 14.6. Công thức nên đặt

Ở phần trên hoặc góc phải nhỏ:

```text
Transformer:
Attn(Q,K,V) = softmax(QK^T / sqrt(d))V
=> O(L^2)

WiMamba:
Temporal state updates + spatial state updates
=> O(L)
```

### 14.7. Speaker cue ngắn

> Ở `M2`, chúng em chưa đổi objective dự đoán. Thay đổi nằm ở sequence encoder: thay attention toàn cục bằng một WiMamba factorized dành riêng cho token WiFi.

### 14.8. Speaker script 20-25 giây

> In `M2`, we replace heavy quadratic attention with factorized WiMamba. The key point is that this is not just a generic Mamba swap. We explicitly split WiFi sequence modeling into two parts: Temporal Mamba for motion over time, and Bidirectional Spatial Mamba for two-way interaction across antenna groups.

### 14.9. Prompt tạo full slide

```text
Create a scientific presentation slide titled "Stage 2 (M2): Replace quadratic attention with factorized WiMamba".

Goal:
This slide compares the old Transformer encoder against our proposed factorized WiMamba encoder for WiFi CSI.

Layout:
- Two large side-by-side comparison cards.
- Left card title: "Transformer"
- Right card title: "Factorized WiMamba"
- White background, navy header bars, cyan accents, clean technical academic style.

Left card content:
- Show a dense all-to-all token interaction diagram.
- Add three bullets:
  1) Full pairwise interactions
  2) O(L^2) memory/compute
  3) Heavy growth for long CSI sequences

Right card content:
- Show a factorized structure with two levels:
  - top level labeled "Temporal Mamba"
  - bottom level labeled "Bidirectional Spatial Mamba"
- Add three bullets:
  1) Temporal pass + spatial pass
  2) O(L) sequence scaling
  3) Edge-friendly inference

Add a small formula box near the top:
Transformer: Attn(Q,K,V) = softmax(QK^T / sqrt(d))V -> O(L^2)
WiMamba: temporal state updates + spatial state updates -> O(L)

Bottom takeaway bar:
"M2 changes the encoder, not the prediction head. Direct regression is still kept at this stage."

Formatting goal:
clean scientific slide, white background, navy header bars, cyan technical modules, black tensor arrows, minimal text, strong visual hierarchy, no decorative clutter, consistent with the existing ResFes WiFi deck
```

### 14.10. Prompt visual-only nếu dựng tay trong PowerPoint

```text
Create a clean technical comparison diagram on white background for a research slide.
Left side: dense Transformer all-to-all token interaction network labeled "Transformer", with a note O(L^2).
Right side: factorized WiMamba diagram labeled "Factorized WiMamba", with two stacked parts labeled "Temporal Mamba" and "Bidirectional Spatial Mamba", and a note O(L).
Use navy header bars, cyan modules, black arrows, minimal clutter, publication-quality academic style.
```

---

## 15. Slide 2 trong bộ 3: Temporal Mamba

### 15.1. Mục tiêu

Slide này phải làm rõ:

- temporal là một mô-đun riêng
- nó chạy trên trục thời gian `T`
- mỗi nhóm không gian được xử lý như một chuỗi thời gian độc lập

### 15.2. Title đề xuất

> Stage 2a: Temporal Mamba models motion along time

### 15.3. Key message một dòng

> We first process each spatial group as its own temporal sequence, so motion dynamics are learned before spatial aggregation.

### 15.4. Layout đề xuất

#### Cột trái: tensor flow

```text
Input:
X in R^(B x S x T x C)

Temporal reshape:
X_t in R^((B*S) x T x C)
```

Vẽ 9 nhóm không gian, mỗi nhóm là một dải theo thời gian.

#### Cột giữa: Temporal Mamba block

Một block cyan lớn:

- `LayerNorm`
- `Mamba_t`
- `Dropout`
- `Residual Add`

#### Cột phải: output

```text
Output:
H_t in R^((B*S) x T x C)

Reshape back:
X' in R^(B x S x T x C)
```

#### Dải kết luận cuối slide

> Temporal Mamba learns motion evolution along time while preserving the WiFi spatial grouping.

### 15.5. Text đặt trực tiếp lên slide

```text
Input view:
X in R^(B x S x T x C)

Temporal view:
reshape -> (B*S) x T x C

Temporal Mamba:
- one temporal scan per spatial group
- learns motion dynamics along time
- preserves output shape for the next spatial stage
```

### 15.6. Công thức nên đặt

```latex
\tilde{X}_t = \operatorname{reshape}(\operatorname{LN}(X), (B S)\times T \times C)
```

```latex
H_t = \operatorname{Mamba}_t(\tilde{X}_t)
```

```latex
X' = X + \operatorname{reshape}(H_t, B \times S \times T \times C)
```

### 15.7. Speaker cue ngắn

> Ở bước đầu tiên, chúng em chưa trộn không gian với thời gian. Thay vào đó, mỗi nhóm anten được xem như một chuỗi thời gian riêng để học motion dynamics trước.

### 15.8. Speaker script 20-25 giây

> In the first WiMamba stage, each spatial group is treated as its own temporal sequence. After reshaping into `B times S` sequences of length `T`, Temporal Mamba scans along time to learn motion dynamics, then reshapes the features back without breaking the WiFi spatial grouping.

### 15.9. Prompt tạo full slide

```text
Create a scientific presentation slide titled "Stage 2a: Temporal Mamba models motion along time".

Goal:
Explain the temporal component inside our factorized WiMamba block for WiFi CSI tokens.

Layout:
- Three-column layout on white background with navy and cyan scientific styling.

Left column:
- Show tensor input labeled:
  X in R^(B x S x T x C)
- Show a reshape arrow to:
  X_t in R^((B*S) x T x C)
- Visualize each spatial group as an independent temporal sequence.

Middle column:
- Large cyan module labeled "Temporal Mamba"
- Small internal labels:
  LayerNorm -> Mamba_t -> Dropout -> Residual Add
- Use horizontal arrows to indicate temporal scanning.

Right column:
- Show output labeled:
  H_t in R^((B*S) x T x C)
- Then reshape back to:
  X' in R^(B x S x T x C)

Add equations in a neat formula box:
X_t_tilde = reshape(LN(X), (BS) x T x C)
H_t = Mamba_t(X_t_tilde)
X' = X + reshape(H_t, B x S x T x C)

Bottom takeaway bar:
"Temporal Mamba learns motion evolution along time while preserving the WiFi spatial grouping."

Formatting goal:
clean scientific slide, white background, navy header bars, cyan technical modules, black tensor arrows, minimal text, strong visual hierarchy, no decorative clutter, consistent with the existing ResFes WiFi deck
```

### 15.10. Prompt visual-only nếu dựng tay trong PowerPoint

```text
Create a clean tensor-processing diagram on white background for a research slide.
Show an input tensor X in B x S x T x C, reshape it into (B*S) x T x C, pass it through a cyan block labeled "Temporal Mamba", then reshape back to B x S x T x C.
Use navy labels, cyan modules, black arrows, and a minimal academic visual style.
```

---

## 16. Slide 3 trong bộ 3: Bi-Mamba

### 16.1. Mục tiêu

Đây là slide quan trọng nhất. Nó phải làm cho BGK thấy ngay:

- `Bi-Mamba` là phần đề xuất có chủ đích
- spatial pass chạy trên trục `S`
- mỗi thời điểm `t` được xử lý như một chuỗi không gian dài `S=9`
- mô hình dùng cả chiều thuận và chiều nghịch để tránh bias một chiều

### 16.2. Title đề xuất

> Stage 2b: Bidirectional Spatial Mamba for two-way antenna-group interaction

### 16.3. Key message một dòng

> After temporal encoding, each time step is reorganized as a short spatial sequence and processed in both directions to model two-way antenna-group interaction.

### 16.4. Layout đề xuất

#### Cột trái: input/output tensor

```text
Input to spatial stage:
X' in R^(B x S x T x C)

Transpose + reshape:
X_s in R^((B*T) x S x C)
```

Ghi chú nhỏ:

- one spatial sequence per time step
- `S = 9` antenna-group tokens

Phần dưới cột trái:

```text
Spatial output:
H_s in R^((B*T) x S x C)

Reshape back:
Y in R^(B x S x T x C)
```

#### Cột giữa: sơ đồ hai nhánh

Nhánh 1:

```text
Forward spatial pass
X_s -> Mamba_s_fwd -> H_s^fwd
```

Nhánh 2:

```text
Backward spatial pass
flip(X_s) -> Mamba_s_bwd -> flip back -> H_s^bwd
```

Ở giữa hoặc dưới:

```text
H_s = H_s^fwd + H_s^bwd
Y = X' + reshape(H_s)
```

#### Cột phải: insight

3 bullet nổi bật:

- captures left-to-right and right-to-left spatial dependencies
- reduces one-way scan bias
- keeps linear-time sequence modeling

### 16.5. Text đặt trực tiếp lên slide

```text
Spatial input:
X' in R^(B x S x T x C)

Spatial view:
transpose + reshape -> (B*T) x S x C

Bidirectional Spatial Mamba:
- forward pass on spatial order
- backward pass on reversed spatial order
- fuse both directions before residual return

Output:
Y in R^(B x S x T x C)
```

### 16.6. Công thức nên đặt

```latex
\tilde{X}_s = \operatorname{reshape}\!\left(\operatorname{transpose}_{S,T}(\operatorname{LN}(X')), (B T)\times S \times C\right)
```

```latex
H_s^{\rightarrow} = \operatorname{Mamba}_{s,\mathrm{fwd}}(\tilde{X}_s)
```

```latex
H_s^{\leftarrow} = \operatorname{flip}\!\left(\operatorname{Mamba}_{s,\mathrm{bwd}}(\operatorname{flip}(\tilde{X}_s))\right)
```

```latex
Y = X' + \operatorname{reshape}\!\left(H_s^{\rightarrow} + H_s^{\leftarrow}, B \times S \times T \times C\right)
```

### 16.7. Box insight bắt buộc

> Bi-Mamba is applied only on the spatial axis after temporal encoding, not on the full 180-token sequence.

### 16.8. Speaker cue ngắn

> Đây là điểm khác biệt quan trọng nhất. Sau temporal pass, chúng em tái tổ chức đặc trưng theo từng thời điểm để chạy Mamba hai chiều trên 9 nhóm không gian. Nhờ đó mô hình thấy được tương tác hai chiều giữa các nhóm anten mà không cần attention toàn cục.

### 16.9. Speaker script 20-25 giây

> This is the most important detail of `M2`. After temporal encoding, each time step becomes a short spatial sequence over the `S = 9` antenna groups. We run one spatial Mamba forward and one backward, fuse both directions, and reshape back. So Bi-Mamba is applied only on the spatial axis, not on the full 180-token sequence.

### 16.10. Prompt tạo full slide

```text
Create a scientific presentation slide titled "Stage 2b: Bidirectional Spatial Mamba for two-way antenna-group interaction".

Goal:
Explain in detail how our Bi-Mamba spatial stage works, with explicit input tensor, reshape, forward/backward passes, and output tensor.

Layout:
- Three-column technical slide on white background with navy header bars and cyan modules.

Left column:
- Show input tensor:
  X' in R^(B x S x T x C)
- Show transpose + reshape:
  X_s in R^((B*T) x S x C)
- Add a short note:
  "one spatial sequence per time step"
  "S = 9 antenna-group tokens"
- Show output:
  H_s in R^((B*T) x S x C)
- Show reshape back:
  Y in R^(B x S x T x C)

Middle column:
- Draw two parallel cyan processing branches.
- Top branch label:
  Forward spatial pass
  X_s -> Mamba_s_fwd -> H_s^fwd
- Bottom branch label:
  Backward spatial pass
  flip(X_s) -> Mamba_s_bwd -> flip back -> H_s^bwd
- At the bottom of the branches show:
  H_s = H_s^fwd + H_s^bwd
  Y = X' + reshape(H_s)

Right column:
- Three bullets:
  1) captures left-to-right and right-to-left spatial dependencies
  2) reduces one-way scan bias
  3) keeps linear-time sequence modeling

Equations box:
X_s_tilde = reshape(transpose(LN(X')), (BT) x S x C)
H_s_right = Mamba_s_fwd(X_s_tilde)
H_s_left = flip(Mamba_s_bwd(flip(X_s_tilde)))
Y = X' + reshape(H_s_right + H_s_left, B x S x T x C)

Add one highlighted callout:
"Bi-Mamba is applied only on the spatial axis after temporal encoding, not on the full 180-token sequence."

Formatting goal:
clean scientific slide, white background, navy header bars, cyan technical modules, black tensor arrows, minimal text, strong visual hierarchy, no decorative clutter, consistent with the existing ResFes WiFi deck
```

### 16.11. Prompt visual-only nếu dựng tay trong PowerPoint

```text
Create a publication-style tensor diagram on white background for a research presentation.
Show a spatial processing pipeline:
input X' in B x S x T x C,
transpose and reshape to (B*T) x S x C,
then split into two branches:
forward spatial Mamba and backward spatial Mamba with flip/reverse,
then fuse both outputs and reshape back to B x S x T x C.
Use navy labels, cyan modules, black arrows, and an academic technical style with clear tensor annotations.
```

---

## 17. Kịch bản rút gọn nếu chỉ còn thời gian làm nhanh

Nếu nhóm chỉ kịp dựng 2 slide thay vì 3, nên ưu tiên:

1. `Overview`
2. `Bi-Mamba chi tiết`

Khi đó:

- phần `Temporal Mamba` được thu gọn thành một khối nhỏ nằm trong slide overview
- còn slide chi tiết duy nhất phải dành cho `Bi-Mamba`, vì đây là phần dễ ghi điểm nhất về contribution

Tuy nhiên, nếu còn đủ thời gian chỉnh trong một buổi nữa, bộ 3 slide ở trên vẫn là phương án nên làm.
