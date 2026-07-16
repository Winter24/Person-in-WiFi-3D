# 2026-05-03 Full Slide Content V2

## Mục đích

Đây là **bản nội dung full slide hoàn chỉnh** cho deck WiFi sau khi áp dụng toàn bộ các quyết định sửa từ:

- `P0 - Làm ngay`
- `P1 - Làm tiếp nếu còn 1 buổi chỉnh`
- `P2 - Nếu còn thời gian polish`

File này được tạo **riêng**, không ghi đè lên [phase4_full_slide_detail.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase4_full_slide_detail.md).

Vai trò của file:

- làm bản dựng nội dung cuối để nhóm chỉnh slide thật
- chốt thứ tự slide
- chốt text chính, visual chính, và điểm nhấn contribution
- chốt các slide phòng thủ kỹ thuật ở appendix

Lưu ý quan trọng:

- Trong workspace hiện tại **không có file source PPT/PPTX/Canva editable**, nên mình chưa thể re-export PDF trực tiếp từ đây.
- File này vì vậy đóng vai trò là **bản nội dung triển khai cuối** để nhóm cập nhật lại deck gốc và xuất PDF sạch.

---

## Snapshot deck sau khi áp dụng P0/P1/P2

### Main deck

- `26` slide
- đã xóa slide duplicate `Why flow helps beyond direct regression`
- đã đổi `MO -> M0`
- đã đổi `In summarize -> In summary`
- đã làm rõ `M0-M4`
- đã tăng urgency, clarity, maturity, và khả năng phòng thủ contribution

### Appendix / backup

- `9` slide
- gồm:
  - `Related Work Positioning`
  - `Why We Trust the Redesign`
  - `Why Mamba, not Transformer?`
  - `Why Rectified Flow, not Direct Regression?`
  - `Edge Deployment / Limitations / Future Work`
  - `Full Technical Backup`
  - `M4 Backbone and Pose Head`
  - `Detailed Qualitative Backup`

Tổng số slide đề xuất:

- `35` slide gồm cả appendix

---

## Main Deck

### Slide 1 - Title

**Title**

> Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures

**Subtitle**

> From Direct Regression to Two-Step Pose Flow Refinement

**Footer**

- Presented by Group NTN
- Nguyễn Vinh Nghi
- Lê Ngọc Anh Thư
- Nguyễn Ngô Nhật Nam

**Mục tiêu slide**

- Chốt thesis ngay từ đầu
- Gợi đúng contribution chính: `Flow` và `Rectify`

**Visual**

- Giữ title clean
- Không thêm quá nhiều text
- Có thể dùng hình nền rất mờ kiểu sóng WiFi hoặc skeleton 3D tối giản

---

### Slide 2 - Content

**Title**

> Content

**Nội dung chính**

```text
01 Problem & Motivation
02 Proposed System
03 Baseline Method
04 What We Propose: A Staged Redesign
05 Experiments & Results
06 Appendix
```

**Contribution map nhỏ ở cuối slide**

> 3 contributions:
> 1) Motion-aware spectral representation  
> 2) Efficient WiMamba sequence modeling  
> 3) Draft-to-refine Rectified Flow for pose correction

**Mục tiêu slide**

- Cho BGK “map để chấm” từ rất sớm

---

### Slide 3 - Section Divider

**Title**

> Problem & Motivation

**Mục tiêu slide**

- Tạo nhịp section rõ
- Không cần thêm nội dung khác

---

### Slide 4 - Why camera-based sensing is not always acceptable

**Title**

> Why camera-based sensing is not always acceptable

**Câu chốt trên slide**

> We need a sensing method that understands posture  
> without capturing appearance.

**So sánh 3 cột**

`Camera`
- Captures appearance
- Raises privacy concerns
- Inappropriate for sensitive spaces

`Wearable`
- Requires device
- User compliance
- Often inconvenient

`WiFi`
- Passive sensing
- No visual identity captured
- Ready for indoor monitoring

**Callout urgency**

> Privacy-sensitive spaces:
> bedroom, rehabilitation room, elderly care, in-home monitoring

**Mục tiêu slide**

- Tăng urgency thật hơn
- Khung vấn đề phải là “bài toán đáng làm”, không chỉ “bài toán thú vị”

---

### Slide 5 - Can WiFi understand human posture without cameras and wearables?

**Title**

> Can WiFi understand human posture  
> without cameras and wearables?

**Text chính**

> No camera.  
> No wearable device.  
> Just WiFi signals.

**Mục tiêu slide**

- Chốt research question
- Làm hook nhận diện của bài

**Visual**

- Text lớn
- Có thể thêm 3 icon nhỏ: camera x, wearable x, WiFi check

---

### Slide 6 - Toward privacy-preserving indoor human sensing

**Title**

> Toward privacy-preserving indoor human sensing

**Nội dung ngắn**

- Smart home monitoring
- Rehabilitation support
- Elderly care facilities
- Fall and activity monitoring

**Câu kết nổi bật**

> This is not about replacing cameras everywhere.  
> It is about enabling sensing where cameras are inappropriate.

**Mục tiêu slide**

- Kết thúc block motivation theo hướng trưởng thành hơn
- Gắn ứng dụng thực tế

---

### Slide 7 - Section Divider

**Title**

> Proposed System

---

### Slide 8 - Proposed System

**Title**

> System overview: from WiFi CSI to 3D pose

**Nội dung nên nói bằng visual**

`Layer 1 - Sensing`
- 1 TX / 3 RX
- CSI acquisition
- Indoor multipath reflections

`Layer 2 - Model pipeline`
- Sequential adapter / tokenizer
- Sequence modeling
- Query decoding
- Pose refinement

`Layer 3 - Output / application`
- 1-person
- 2-person
- 3-person
- Indoor deployment scenarios

**Câu kết nhỏ**

> We first show the whole system,  
> then define the baseline before explaining our redesign.

**Mục tiêu slide**

- Đưa global picture
- Tránh sa vào low-level

**Ghi chú khi dựng**

- Làm nổi 3 tầng bằng bố cục
- Không để TX/RX low-level chiếm toàn bộ spotlight

---

### Slide 9 - Section Divider

**Title**

> Baseline Method

---

### Slide 10 - `M0` baseline: Linear projection + Transformer + direct regression

**Title**

> `M0` baseline: Linear projection + Transformer + direct regression

**Câu chốt nổi bật**

> `M0` is a strong PETR-style baseline  
> that predicts the final 3D skeleton in one shot.

**Pipeline trên slide**

```text
WiFi CSI input
-> Linear projection
-> Transformer encoder
-> Query decoder
-> Classification + keypoint refinement
-> Final 3D pose
```

**Callout nhỏ**

- Strong baseline
- Query-based set prediction
- One-shot final pose prediction

**Mục tiêu slide**

- Định nghĩa baseline thật rõ
- Tạo nền so sánh công bằng

---

### Slide 11 - What `M0` optimizes

**Title**

> What `M0` optimizes

**Công thức trên slide**

```text
L_M0 = L_cls + L_kpt + L_oks + L_refine
```

**Lưu ý nhỏ bên dưới**

> Hungarian matching is the assignment step  
> before these losses are computed.

**Giải thích ngắn trên slide**

- `L_cls`: person classification
- `L_kpt`: direct keypoint coordinate regression
- `L_oks`: structure-aware consistency
- `L_refine`: decoder-side refinement loss

**Câu kết nổi bật**

> The target is still the final coordinates themselves,  
> not a correction trajectory.

**Mục tiêu slide**

- Làm rõ `M0` học cái gì
- Chuẩn bị cho cú nhảy sang Flow

---

### Slide 12 - Section Divider

**Title**

> What We Propose: A Staged Redesign

---

### Slide 13 - A controlled `M0-M4` ablation ladder

**Title**

> A controlled `M0-M4` ablation ladder

**Bảng đề xuất mới**

| Model | What changed from previous stage? | Representation | Sequence Modeling | Prediction Objective | Main purpose |
|---|---|---|---|---|---|
| `M0` | Baseline | Linear projection | Transformer | Direct regression | Baseline |
| `M1` | Better input representation | Spectral tokenizer | Transformer | Direct regression | Test representation |
| `M2` | Better efficiency | Spectral tokenizer | WiMamba | Direct regression | Test efficiency |
| `M3` | Better prediction target | Spectral tokenizer | Transformer | Rectified Flow | Test flow accuracy |
| `M4` | Combine flow + efficiency | Spectral tokenizer | WiMamba | Rectified Flow | Final deployment trade-off |

**3 trụ tô đậm**

- Representation
- Efficiency
- Refinement

**Câu chốt dưới slide**

> Each stage isolates one hypothesis,  
> so our contribution is interpretable and measurable.

**Mục tiêu slide**

- Đây là slide trung tâm để BGK so ra `M0-M4`
- Phải nhìn 5 giây là hiểu logic contribution

---

### Slide 14 - Stage 1: motion-aware spectral tokenization

**Title**

> Stage 1: motion-aware spectral tokenization

**Pipeline trực quan**

```text
Raw CSI
-> Temporal smoothing
-> Doppler profile via FFT / RFFT
-> Frequency gate
-> Dynamic motion-enhanced tokens
```

**Câu ý nghĩa nổi bật**

> Emphasize motion.  
> Suppress static noise.

**Nếu còn chỗ, thêm mini before/after**

- Before: raw tokens
- After: motion-aware tokens

**Mục tiêu slide**

- Chỉ rõ `M1` khác `M0` ở representation
- Dùng feature/pipeline làm bằng chứng

---

### Slide 15 - Stage 2 (`M2`): Replace quadratic attention with factorized WiMamba

**Title**

> Stage 2 (`M2`): Replace quadratic attention with factorized WiMamba

**Công thức / ý chính**

```text
Transformer attention:
Attn(Q,K,V) = softmax(QK^T / sqrt(d))V
-> O(L^2)

WiMamba:
Temporal state updates + spatial state updates
-> O(L)
```

**Hai hộp nổi bật**

`Transformer`
- Full pairwise interactions
- `O(L^2)`
- Heavy memory growth

`WiMamba`
- Factorized temporal + spatial modeling
- `O(L)`
- Edge-friendly inference

**Câu chốt**

> We do not replace Transformer because it is weak.  
> We replace it because WiFi edge deployment needs linear scaling.

**Mục tiêu slide**

- Làm rõ `M2` khác `M1` ở sequence modeling
- Gài sẵn phần phòng thủ Mamba

---

### Slide 16 - Stage 3: From direct regression to draft-to-refine

**Title**

> Stage 3: From direct regression to draft-to-refine

**Công thức ngắn**

```text
X_t = (1 - t)X_0 + tX_1
v_theta = v_theta(X_t, t, c)
X_1_hat = X_0 + v_theta
```

**Nhãn bắt buộc**

- `X_0`: draft pose
- `v_theta`: correction velocity
- `X_1_hat`: refined pose
- `c`: encoded WiFi feature

**Câu chốt**

> The key change is not just a new head.  
> The model now learns how to correct a draft pose.

**Mục tiêu slide**

- Chỉ rõ `M3/M4` đổi prediction process

---

### Slide 17 - The proposed `M4` pipeline

**Title**

> The proposed `M4` pipeline

**Yêu cầu visual**

- Highlight novelty path bằng màu riêng:
  - spectral tokenizer
  - factorized WiMamba encoder
  - top-k query selection
  - one-step rectified-flow refinement

**Tag ngắn trên slide**

- Motion-aware tokens
- Linear sequence modeling
- Draft-to-refine correction

**Câu chốt**

> `M4` is not a random mix of modules.  
> It is the final integration of all stage-wise improvements.

**Mục tiêu slide**

- Giúp BGK nhìn ra contribution nằm ở đâu trong pipeline cuối

---

### Slide 18 - `M4` Training Pipeline

**Title**

> `M4` Training Pipeline

**Luật trình bày trên slide**

`Upper tier`
- forward architecture path
- tokenization -> sequence modeling -> query decoding -> draft pose -> refined pose

`Lower tier`
- matching
- regression losses
- flow loss
- backpropagation path

**Câu nhắc trên slide**

> Architecture path above.  
> Training and losses below.

**Mục tiêu slide**

- Giữ depth kỹ thuật
- Giảm cảm giác “wall of boxes”

---

### Slide 19 - Section Divider

**Title**

> Experiments & Results

---

### Slide 20 - Dataset Overview

**Title**

> Dataset overview

**3 ý bắt buộc trên slide**

1. **Source**
   - WiFi CSI indoor dataset
2. **Setting**
   - 1-person
   - 2-person
   - 3-person
3. **Annotation**
   - 3D pose labels from Azure Kinect Body Tracking SDK annotations

**Câu chốt**

> Scene difficulty increases from 1-person to 3-person settings.

**Mục tiêu slide**

- Tăng độ chín nghiên cứu
- Chuẩn bị cho result interpretation

---

### Slide 21 - Evaluation Metric: Mean Per Joint Position Error (MPJPE)

**Title**

> Evaluation Metric: Mean Per Joint Position Error (MPJPE)

**Text rút gọn**

> MPJPE computes the mean 3D Euclidean distance  
> between predicted joints and ground-truth joints.

**Callout lớn**

> Lower MPJPE = better 3D pose accuracy

**Mục tiêu slide**

- Giảm text
- Cho BGK hiểu metric ngay lập tức

---

### Slide 22 - Flow improves accuracy. Mamba improves deployability.

**Title**

> Flow improves accuracy. Mamba improves deployability.

**Bảng chính**

| Model | MPJPE ↓ | FPS ↑ | Params ↓ | Peak Memory ↓ |
|---|---:|---:|---:|---:|
| `M0` | 169.34 | 45.69 | 13.13M | 155.60 MB |
| `M3` | 151.99 | 138.79 | 7.06M | 38.49 MB |
| `M4` | 159.00 | 159.85 | 5.83M | 27.02 MB |

**Hai kết luận nổi bật**

- `M3` is the accuracy winner.
- `M4` is the deployment winner.

**Wording an toàn**

> Preliminary local benchmark snapshot

**Không dùng wording**

- “wins everything”
- “best in every metric”

**Mục tiêu slide**

- Đây là slide kết luận contribution bằng số liệu
- Vừa mạnh vừa trung thực

---

### Slide 23 - More coherent matched poses in challenging scenes

**Title**

> More coherent matched poses in challenging scenes

**Bố cục**

- Ground Truth
- `M0` Baseline
- `M3` (Accuracy)
- `M4` (Efficiency)

**Phải highlight**

- 1-2 failure case cụ thể của `M0`
- case đó được sửa bởi `M3/M4`

**Caption**

> Flow-based variants correct structural collapse  
> under severe multipath ambiguity.

**Mục tiêu slide**

- Contribution phải có bằng chứng bằng hình ảnh

---

### Slide 24 - Why flow helps beyond direct regression

**Title**

> Why flow helps beyond direct regression

**Hai cột**

`The old paradigm`
- Direct Regression (`M0`)
- Model learns a single jump to final coordinates

`The new paradigm`
- Trajectory Refinement (`M3/M4`)
- Model learns the correction path from draft pose to target pose

**Công thức ngắn**

```text
L_flow = E || v_theta(X_t, t, c) - (X_1 - X_0) ||^2
```

**Câu chốt lớn**

> Learn a correction path,  
> not a one-shot jump.

**Mục tiêu slide**

- Đây là slide intuition + formula để BGK nhớ contribution

---

### Slide 25 - In summary

**Title**

> In summary

**3 takeaway boxes**

`What changed`
- `M0 -> M4` is a staged redesign
- Better representation
- Better efficiency
- Better prediction objective

`What improved`
- Flow drives accuracy
- WiMamba drives efficiency
- `M4` gives the strongest deployment trade-off

`What remains next`
- Broader validation
- Stronger edge deployment
- Harder multi-person robustness

**Mục tiêu slide**

- Kết bài theo kiểu nghiên cứu trưởng thành hơn
- Không chỉ recap metric

---

### Slide 26 - Thanks

**Title**

> Thanks For Your Listening

**Subtitle nhỏ**

> Questions and discussion

---

## Appendix

### Slide 27 - Appendix Divider

**Title**

> Appendix

---

### Slide 28 - Related Work Positioning

**Title**

> Related Work Positioning

**3 cụm positioning**

1. PETR-style direct regression baselines
2. Efficient sequence modeling for long token sequences
3. Flow-based refinement and correction objectives

**Câu chốt**

> Our work sits at the intersection of efficient sequence modeling  
> and trajectory-based pose refinement for WiFi sensing.

**Mục tiêu slide**

- Dùng khi BGK hỏi “bài này đứng ở đâu trong literature?”

---

### Slide 29 - Why We Trust the Redesign

**Title**

> Why We Trust the Redesign

**3 bằng chứng**

1. Controlled `M0-M4` ablation ladder
2. Quantitative gain: `M3` and `M4` improve over `M0`
3. Qualitative gain: more coherent matched poses

**Câu chốt**

> We do not ask the audience to trust a black box.  
> We show controlled changes and their effects.

---

### Slide 30 - Why Mamba, not Transformer?

**Title**

> Why Mamba, not Transformer?

**Công thức**

```text
Transformer:
Attn(Q,K,V) = softmax(QK^T / sqrt(d))V
QK^T in R^(L x L)
=> compute O(L^2), memory O(L^2)
```

```text
Selective State Space Model:
h_t = A_t h_(t-1) + B_t x_t
y_t = C_t h_t + D x_t
=> recurrent state updates
=> compute O(L)
```

**Hai kết luận ngắn**

- Transformer builds full pairwise interactions
- Mamba updates a fixed-size state over the sequence

**Callout WiFi-specific**

> We further factorize temporal and spatial modeling  
> for WiFi token structure.

**Callout edge-specific**

> Linear scaling matters for edge AI deployment  
> such as WiFi routers and gateways.

---

### Slide 31 - Why Rectified Flow, not Direct Regression?

**Title**

> Why Rectified Flow, not Direct Regression?

**Direct regression**

```text
X_hat = f_theta(c)
L_reg = ||X_hat - X_1||^2
```

**Rectified Flow**

```text
X_t = (1 - t)X_0 + tX_1
v_theta(X_t, t, c) ~= X_1 - X_0
L_flow = E || v_theta(X_t, t, c) - (X_1 - X_0) ||^2
```

**Câu chốt**

> Direct regression predicts the final pose in one shot.  
> Rectified Flow learns how to correct a draft pose.

**Callout customization**

- not pure image-noise generation
- conditioned on encoded WiFi features
- starts from draft pose queries
- one-step pose refinement for WiFi 3D skeletons

---

### Slide 32 - Edge Deployment / Limitations / Future Work

**Title**

> Edge Deployment, Limitations, and Future Work

**Edge deployment**

- `O(L)` scaling is more practical than `O(L^2)`
- lower memory suits WiFi router / gateway deployment
- `M4` is the deployment-oriented model

**Limitations**

- current benchmark is a local snapshot
- broader validation is still needed
- harder multi-person scenes remain challenging

**Future work**

- stronger on-device deployment
- broader test environments
- more robust multi-person correction

---

### Slide 33 - Full Technical Backup

**Title**

> Full Technical Backup

**Nội dung**

- full end-to-end architecture
- matching
- losses
- refinement path
- training overview

**Mục tiêu slide**

- Dùng khi BGK muốn “xem tất cả trên một màn hình”

---

### Slide 34 - `M4` Backbone and Pose Head

**Title**

> `M4` Backbone and Pose Head

**Nội dung**

- spectral tokenizer internals
- factorized WiMamba encoder
- lightweight query decoder
- draft pose + flow update head

---

### Slide 35 - Detailed Qualitative Backup

**Title**

> Detailed Qualitative Backup

**Nội dung**

- thêm nhiều case 1-person / 2-person / 3-person
- failure modes
- ambiguous scenes
- matched vs cluttered predictions

---

## Checklist triển khai thực tế từ file này

### P0 - Làm ngay

1. Re-export deck sạch với `M0`
2. Xóa slide duplicate cũ
3. Đổi `In summarize -> In summary`
4. Cập nhật ladder `M0-M4`
5. Cập nhật summary slide theo 3 takeaway boxes

### P1 - Làm tiếp trong 1 buổi chỉnh

6. Tăng urgency ở slide 4/6
7. Tăng contribution map ở slide 2/13
8. Tối ưu slide 8, 18, 20, 22, 23, 24
9. Dựng 3 slide backup:
   - Why Mamba?
   - Why Rectified Flow?
   - Edge Deployment / Limitations / Future Work

### P2 - Nếu còn thời gian polish

10. Thêm Related Work Positioning
11. Thêm Why We Trust the Redesign
12. Tinh highlight novelty path ở slide 17
13. Rút text slide 21

---

## Kết luận

Phiên bản full slide content này đã thực thi đúng tinh thần:

- contribution phải dễ nhìn thấy
- `M0-M4` phải dễ so ra
- công thức và visual phải phục vụ giải thích contribution
- appendix phải là bộ phòng thủ kỹ thuật

Từ file này, nhóm có thể dựng lại deck thật và xuất ra bản PDF sạch mà không cần sửa tiếp trong [phase4_full_slide_detail.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase4_full_slide_detail.md).
