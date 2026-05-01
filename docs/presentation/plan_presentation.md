# Plan Presentation — ResFes 2026

## Mục tiêu tổng thể

Dự án của nhóm đã qua vòng sơ khảo và bước vào vòng chấm điểm thuyết trình. Mục tiêu của kế hoạch này là chuẩn bị một phần trình bày **ấn tượng, rõ ràng, có chiều sâu nghiên cứu, kiểm soát tốt Q&A, và tối đa hóa cơ hội đạt giải Nhất**.

Thông điệp trung tâm của bài thuyết trình:

> **We turn WiFi pose estimation from one-shot guessing into efficient draft-to-refine trajectory refinement.**

Ba điều hội đồng cần nhớ sau phần trình bày:

1. **Problem:** Camera-based pose estimation chính xác nhưng không phù hợp trong nhiều không gian riêng tư.
2. **Innovation:** Nhóm thay direct one-shot regression bằng **Draft-to-Refine Rectified Flow**.
3. **Evidence:** M3 đạt accuracy tốt nhất; M4 đạt trade-off triển khai tốt nhất với tốc độ cao, ít tham số và memory thấp.

---

## Giai đoạn 1 — Chốt chiến lược nội dung và thông điệp chính

**Thời gian đề xuất:** Ngày 1  
**Mục tiêu:** Cả nhóm thống nhất bài thuyết trình sẽ kể câu chuyện gì, tránh lan man kỹ thuật.

### 1. Chốt tên trình bày

Tên proposal hiện tại:

> **Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures.**

Subtitle:

> **From One-Shot Regression to Draft-to-Refine Rectified Flow**

### 2. Chốt 3 thông điệp chính

**Message 1 — Problem**  
Camera-based pose estimation chính xác nhưng có privacy risk và không phù hợp trong các không gian nhạy cảm như phòng ngủ, bệnh viện, nhà người già hoặc smart home.

**Message 2 — Innovation**  
Thay vì bắt mô hình đoán pose cuối cùng một lần từ CSI nhiễu, nhóm tạo một draft pose trước, sau đó dùng Rectified Flow để học cách tinh chỉnh draft đó thành skeleton 3D ổn định hơn.

**Message 3 — Evidence**  
M3 là model tốt nhất về accuracy; M4 là model tốt nhất về deployment trade-off với FPS cao, params thấp và memory thấp.

### 3. Chốt vai trò từng thành viên

Nếu 3 người cùng trình bày:

| Người | Phần phụ trách | Mục tiêu |
|---|---|---|
| Speaker 1 | Hook, problem, motivation | Tạo hứng thú |
| Speaker 2 | Method, pipeline, ablation | Thể hiện chiều sâu kỹ thuật |
| Speaker 3 | Results, impact, conclusion | Chứng minh giá trị và chốt bài |

Nếu 2 người trình bày:

| Người | Phần phụ trách |
|---|---|
| Speaker 1 | Hook → problem → key idea |
| Speaker 2 | Method → results → impact → conclusion |

### Output của giai đoạn 1

- Tên bài thuyết trình chính thức.
- 3 thông điệp chính.
- Phân công người nói.
- Outline 10–11 slide.
- Quyết định bài nói dùng tiếng Anh hoàn toàn hay tiếng Anh có giải thích nội bộ bằng tiếng Việt khi luyện.

---

## Giai đoạn 2 — Chuẩn hóa số liệu, bằng chứng và nội dung nghiên cứu

**Thời gian đề xuất:** Ngày 2–3  
**Mục tiêu:** Đảm bảo mọi số liệu, claim và kết luận trong slide đều chính xác, nhất quán với proposal.

### 1. Xác nhận lại bảng kết quả chính

Bảng benchmark hiện có:

| Model | MPJPE ↓ | FPS ↑ | Params ↓ | Peak Memory ↓ |
|---|---:|---:|---:|---:|
| M0 | 169.34 | 45.69 | 13.13M | 155.60 MB |
| M1 | 164.60 | 49.36 | 13.20M | 155.86 MB |
| M2 | 169.41 | 51.48 | 11.97M | 151.16 MB |
| M3 | 151.99 | 138.79 | 7.06M | 38.49 MB |
| M4 | 159.00 | 159.85 | 5.83M | 27.02 MB |

Cần kiểm tra:

- Các số này có phải từ cùng một môi trường benchmark không?
- FPS đo với batch size bao nhiêu?
- Memory đo trên GPU nào?
- MPJPE đo trên test split nào?
- Có dùng cùng checkpoint/evaluation script không?
- Có cần ghi “preliminary frozen local snapshot” không?

Nếu chưa có multi-seed, trên slide nên dùng phrasing:

> **Preliminary benchmark results**

Không nên nói:

> “Final results prove…”

Nên nói:

> “These preliminary results indicate…”

### 2. Chốt cách giải thích M3 và M4

M3 có MPJPE tốt nhất: **151.99 mm**.  
M4 có tốc độ, params, memory tốt nhất: **159.85 FPS, 5.83M params, 27.02 MB**.

Cách nói đúng:

> **M3 is the accuracy winner. M4 is the practical deployment winner.**

Không nên nói:

> “M4 is the best model” một cách chung chung.

Nên nói:

> “M4 is selected as the final deployment-oriented model because it provides the best accuracy-efficiency trade-off.”

### 3. Chuẩn hóa research questions

Đưa vào slide dạng ngắn:

**RQ1:** Can Spectral Tokenization and WiMamba improve CSI representation and efficiency?  
**RQ2:** Can Rectified Flow improve pose accuracy and structural fidelity?  
**RQ3:** Which model provides the best accuracy-efficiency trade-off?

### 4. Chốt contribution

Trong slide chỉ nên có 3 contribution:

1. **Draft-to-Refine Rectified Flow** for WiFi-based 3D multi-person pose estimation.
2. **Efficient CSI encoding** using Spectral Tokenization and WiMamba.
3. **Controlled M0–M4 ablation** evaluating accuracy, FPS, parameters and memory.

### Output của giai đoạn 2

- Bảng số liệu đã xác nhận.
- Câu giải thích chính thức về M3 vs M4.
- 3 research questions.
- 3 contributions.
- Danh sách claim nào được nói mạnh, claim nào phải nói thận trọng.

---

## Giai đoạn 3 — Chuẩn bị hình ảnh, video và demo trực quan

**Thời gian đề xuất:** Ngày 4–6  
**Mục tiêu:** Tạo phần visual đủ mạnh để người không chuyên cũng hiểu và nhớ dự án.

Đây là giai đoạn cực kỳ quan trọng vì đề tài kỹ thuật sâu. Nếu chỉ nói Transformer, Mamba, CSI, Rectified Flow, hội đồng có thể bị quá tải. Visual phải làm nhiệm vụ “dịch kỹ thuật thành trực giác”.

### Trạng thái thực thi hiện tại

Đối chiếu với code và asset hiện có trong repo, giai đoạn 3 hiện được chia thành bốn nhóm:

- **Làm ngay được bằng asset hiện có:** Figure 1 overview pipeline, Figure 3 detailed M4 architecture, Figure 4 multi-metric comparison, Figure 5 qualitative comparison, bảng M0–M4 rút gọn, slide problem dạng bảng.
- **Làm ngay được bằng script hiện có:** video hook từ `tools/analysis/render_presentation_video.py`, Figure 5 từ `tools/analysis/render_qualitative_figure.py`, bubble/Pareto chart từ `tools/analysis/plot_teaser_figure.py`.
- **Làm ngay được nhưng chủ yếu là thao tác slide thủ công:** slide problem, direct-regression fail callout, draft-to-refine Morph animation, slide backup formula, slide Responsible AI.
- **Chỉ cần code bổ sung nếu muốn tự động hóa sâu hơn:** auto-annotate failure cases, auto-generate draft-to-refine vector animation, xuất riêng asset Figure 6 và Figure 7 từ proposal PDF cuối.

Nguyên tắc thực thi của giai đoạn này là: **ưu tiên tái sử dụng asset và script đã có, chỉ viết thêm code khi phần visual đó thực sự không thể hoàn thành đủ tốt bằng PowerPoint hoặc export thủ công.**

---

### 3.1. Video mở đầu

**Thời lượng:** 10–15 giây

Cấu trúc video:

| Cảnh | Nội dung | Text overlay |
|---|---|---|
| 1 | Người di chuyển trong phòng | “Human motion in indoor spaces” |
| 2 | Tín hiệu WiFi/CSI dạng wave hoặc heatmap | “Captured through WiFi signals” |
| 3 | Skeleton 3D xuất hiện | “3D pose without cameras” |

Câu chữ nên hiện cuối video:

> **No camera. No wearable. Just WiFi signals.**

Cách làm:

- Có thể dùng PowerPoint, Canva, CapCut hoặc Python animation.
- Nếu không đủ thời gian làm video thật, dùng animation slide:
  - ảnh RGB dataset;
  - hiệu ứng sóng WiFi;
  - skeleton 3D fade in;
  - text xuất hiện từng dòng.

**Khả năng thực thi hiện tại**

- **Làm ngay được bằng code:** `tools/analysis/render_presentation_video.py` đã hỗ trợ merge `train_data` + `test_data`, ghép video gốc `output.mkv`, đọc `time_list.txt`, render GT + M0 baseline + M3/M4 target.
- **Làm ngay được bằng slide thủ công:** nếu video render không ổn hoặc thiếu thời gian, vẫn có thể dùng PowerPoint/Canva/CapCut để dựng hook giả lập.
- **Không cần viết thêm code mới cho opening video:** chỉ cần render đúng mode và chọn đúng đoạn video.

**Mode video cần khóa**

- **Presentation mode:** không bật `--show-unmatched`, dùng `--score-thr 0.0`, `--match-quality-thr-mm 500`, và chọn test segment. Mode này tập trung vào chất lượng skeleton matched, tránh để FP/duplicate queries làm rối slide mở đầu.
- **Audit mode:** bật `--show-unmatched` để nhìn rõ FP/duplicate predictions. Mode này chỉ dùng cho backup hoặc khi cần kiểm tra kỹ thuật, không dùng làm hook chính.

**Lệnh render khuyến nghị trên Colab**

```bash
python /content/Person-in-WiFi-3D/tools/analysis/render_presentation_video.py \
  --video-id S23_12 \
  --config /content/Person-in-WiFi-3D/configs/wifi/wi_tidir_wifi_transformer.py \
  --checkpoint /content/drive/MyDrive/RESEARCH/RESFES2026/Ablation_20e_RTX3090/M1_5/M3/epoch_20.pth \
  --model M3 \
  --baseline-model M0 \
  --baseline-config /content/Person-in-WiFi-3D/configs/wifi/petr_wifi.py \
  --baseline-checkpoint /content/drive/MyDrive/RESEARCH/RESFES2026/Ablation_20e_RTX3090/M1_5/M0/epoch_20.pth \
  --source-video /content/drive/MyDrive/RESEARCH/RESFES2026/S23_12/output.mkv \
  --source-time-list /content/drive/MyDrive/RESEARCH/RESFES2026/S23_12/time_list.txt \
  --start-frame 301 \
  --end-frame 349 \
  --score-thr 0.0 \
  --match-quality-thr-mm 500 \
  --fps 12 \
  --output /content/drive/MyDrive/RESEARCH/RESFES2026/S23_12/presentation_S23_12_M0_vs_M3_presentation.mp4
```

**Khuyến nghị thực thi**

- Ưu tiên render **presentation mode** trước, dài 10–15 giây.
- Nếu video còn rối vì FP hoặc jitter, dùng video như backup và làm slide animation hook gọn hơn.
- Không dùng audit mode trên slide mở đầu vì hội đồng dễ hiểu nhầm FP là pose quality tệ hơn, trong khi benchmark chính là matched MPJPE.

---

### 3.2. Hình “Problem”

Chuẩn bị một slide so sánh:

| Camera | Wearable | WiFi |
|---|---|---|
| Captures appearance | Requires device | No visual identity |
| Privacy concern | User compliance | Passive sensing |
| Sensitive spaces issue | Inconvenient | Suitable for indoor monitoring |

Câu nói kèm:

> “Cameras can estimate pose, but they also capture identity. WiFi signals do not capture visual appearance, which makes them promising for privacy-sensitive indoor sensing.”

**Khả năng thực thi hiện tại**

- **Làm ngay được:** hoàn toàn không cần code.
- Chỉ cần dựng một slide bảng hoặc icon comparison trong PowerPoint.

**Khuyến nghị thực thi**

- Chốt đây là một **sub-process không cần code**.
- Thời gian nên ưu tiên cho visual method/results thay vì làm đẹp quá mức slide này.

---

### 3.3. Hình “Direct regression fails”

Chuẩn bị một case 2-person hoặc 3-person:

- Ground Truth;
- M0 baseline;
- M3;
- M4.

Cần khoanh đỏ skeleton M0 bị lệch/méo/collapsed.

Mục tiêu:

> Hội đồng thấy baseline direct regression dễ tạo skeleton lệch/méo trong một số cảnh đông người, còn flow-based refinement cải thiện chất lượng pose matched theo MPJPE.

**Khả năng thực thi hiện tại**

- **Làm ngay được:** đã có sẵn asset qualitative tại `work_dirs/paper_M1-5/figures/figure4_qualitative.png` (legacy filename, dùng như **Figure 5** trong proposal cuối) và các frame tham chiếu như `S11_06_319.jpg`, `S23_12_337.jpg`, `S52_40_322.jpg`.
- `render_qualitative_figure.py` đã tồn tại và có thể render lại Figure 5 khi cần.
- **Không cần code bổ sung** nếu chỉ cần chọn 1 case đẹp và khoanh đỏ thủ công trên slide.

**Chỉ cần code bổ sung nếu**

- nhóm muốn auto-highlight lỗi của `M0` bằng box, arrow, hay callout được sinh tự động.

**Khuyến nghị thực thi**

- Dùng ngay `Figure 5` làm nguồn.
- Khoanh đỏ thủ công phần skeleton lỗi của `M0` trong PowerPoint để nhanh và kiểm soát tốt hơn.
- Không claim rằng `M3` luôn ít FP hơn `M0`. Nếu nói về video, cần phân biệt rõ: `M3` tốt hơn theo matched MPJPE, nhưng có thể có nhiều duplicate/FP hơn do confidence/query calibration.

---

### 3.4. Hình pipeline chính

Slide method chính nên **ưu tiên dùng Figure 1 trong proposal cuối** vì đây là hình overview đã khớp trực tiếp với `NTN_IT_CT.pdf`. Không nên tự tạo một pipeline khác nếu nó làm lệch cách kể chuyện với proposal. Trong trường hợp cần một bản đơn giản hơn để nói miệng, có thể rút Figure 1 về đúng 5 block lớn sau:

```text
WiFi CSI
   ↓
Spectral Tokenizer
   ↓
WiMamba Encoder
   ↓
Draft Pose
   ↓
Rectified Flow Refinement
   ↓
3D Multi-person Skeleton
```

Dưới mỗi block chỉ có 3–5 từ:

- Spectral Tokenizer: “motion-aware denoising”
- WiMamba: “linear-time encoding”
- Draft Pose: “coarse skeleton”
- Rectified Flow: “trajectory correction”
- Output: “refined 3D pose”

**Khả năng thực thi hiện tại**

- **Làm ngay được:** `Figure 1` overview và `Figure 3` detailed M4 architecture đã có sẵn trong proposal cuối và thư mục `work_dirs/paper_M1-5/figures`.
- **Không cần code bổ sung** nếu nhóm dùng trực tiếp `Figure 1` làm slide method chính.

**Chỉ cần code bổ sung nếu**

- nhóm muốn một bản simplified pipeline mới được render tự động thay vì rút gọn thủ công từ Figure 1.

**Khuyến nghị thực thi**

- Slide chính dùng **Figure 1**.
- Nếu cần bản ít chữ hơn, chỉnh trực tiếp trên slide thay vì mở nhánh code riêng.

---

### 3.5. Animation “Draft-to-Refine”

Đây là visual nên ưu tiên cao nhất.

Mục tiêu của visual này là giải thích trực giác:

```text
Draft Pose X0 + predicted velocity v -> Refined Pose X1
```

Điểm cần giữ nhất quán: skeleton trong visual là **keypoint graph của pose estimation**, không phải bộ xương giải phẫu người thật. Vì vậy mọi hình/video phải dùng các điểm tròn và đoạn nối đơn giản, tránh skull/ribs/spine/anatomical skeleton.

**Bố cục đã chốt**

Tạo 3 frame:

1. `Draft Pose X0`: keypoint graph màu xám, hơi lệch nhưng vẫn plausible.
2. `Learned/Predicted Velocity v`: keypoint graph mờ, các mũi tên cyan biểu diễn local keypoint correction.
3. `Refined Pose X1`: keypoint graph màu xanh, ổn định và cân đối hơn.

Text:

> **Not one-shot guessing — trajectory refinement.**

**Cách làm ưu tiên**

- Dùng ảnh 3-stage đã tạo làm base image.
- Dùng Veo 3 để tạo animation nhẹ: pulse ở `X0`, mũi tên velocity di chuyển đúng hướng, `X1` glow/stabilize.
- Nếu Veo tạo chữ lỗi hoặc biến dạng skeleton, fallback sang PowerPoint Morph/CapCut với ảnh tĩnh.

**Prompt Veo 3 đã chốt**

```text
Use the provided image as the exact visual reference. Create an 8-second scientific diagram animation in 16:9.

Important conceptual constraint:
The skeletons are abstract keypoint graphs for pose estimation, not realistic human bones. They must be rendered only as spherical keypoints connected by simple cylindrical or line segments. Do not make them look anatomical, biological, medical, or like real human skeleton bones.

Keep the same three-stage layout:
Left: Draft Pose X0, gray keypoint graph.
Middle: Learned Velocity v, semi-transparent gray keypoint graph with cyan velocity arrows.
Right: Refined Pose X1, blue keypoint graph.

Animation:
0-2 seconds:
Highlight the gray draft keypoint graph on the left. Slightly pulse the gray keypoints to indicate an uncertain coarse prediction.

2-5 seconds:
Animate the cyan velocity arrows in the middle. Each keypoint should move exactly in the direction indicated by its own cyan arrow. The motion must follow the arrow direction, not randomly drift. The arrows represent local joint/keypoint displacement vectors. The movement should be short, local, and smooth.

5-8 seconds:
The corrected keypoint positions settle into the blue refined keypoint graph on the right. The blue refined keypoints glow softly and stabilize.

Strict motion rules:
- Velocity vectors must move from the tail of each arrow toward the arrowhead.
- Do not reverse arrow directions.
- Do not create long horizontal movement across the whole image.
- Do not move the entire skeleton as one object.
- Only animate local keypoint corrections along the cyan arrows.
- Keep the camera locked and the three panels fixed.
- Keep all labels static and readable.

Visual style:
Clean academic ML visualization, white background, light-blue 3D grid, subtle CSI/WiFi wave lines, gray for draft, cyan for velocity, blue for refined. Smooth, minimal, precise.

Do not add:
realistic human body, face, skin, clothing, anatomical bones, medical skeleton, muscles, organs, horror distortion, random text, logos, watermarks, fake equations, UI elements.
```

Negative prompt nếu Veo hỗ trợ:

```text
real human skeleton, anatomical bones, skull, ribs, spine, face, skin, clothing, medical illustration, horror, random motion, reversed arrows, camera movement, extra text, watermark, logo
```

**Fallback PowerPoint Morph**

- Slide A: skeleton draft.
- Slide B: skeleton refined.
- Dùng Morph để chuyển động.

Nếu không có animation, dùng 3 hình đặt ngang:

```text
Draft pose → Learned velocity field → Refined pose
```

**Khả năng thực thi hiện tại**

- **Làm ngay được:** dùng ảnh 3-stage hiện có làm slide tĩnh hoặc đưa vào Veo 3 để sinh video 8 giây.
- **Không cần code repo:** đây là visual minh họa khái niệm, không cần lấy vector thật từ model.
- **Fallback chắc chắn:** PowerPoint Morph hoặc ba frame tĩnh nếu Veo output bị lỗi chữ/keypoint.

**Chỉ cần code bổ sung nếu**

- nhóm muốn animation này được sinh tự động từ output mô hình hoặc từ vector correction thật.

**Khuyến nghị thực thi**

- Ưu tiên tạo video Veo 3 8 giây từ ảnh 3-stage.
- Nếu Veo làm sai hướng mũi tên hoặc biến keypoint graph thành anatomical skeleton, không dùng video đó; chuyển về ảnh tĩnh hoặc Morph.
- Khi trình bày, nói rõ đây là **conceptual animation**, không phải frame inference thật.

**Câu nói trên slide**

> “Instead of directly guessing the final pose, the model starts from a coarse draft `X0`, predicts a local velocity field `v`, and performs a one-step correction toward `X1`.”

**Tiêu chí nghiệm thu**

- Mũi tên velocity di chuyển đúng chiều từ tail đến arrowhead.
- Không có hình người thật, mặt, da, quần áo, hoặc bộ xương giải phẫu.
- `X0`, `v`, `X1` đọc được trong 2 giây.
- Animation không làm camera rung, không đổi layout ba stage.
- Video có thể nhúng PowerPoint và mở được offline.

---

### 3.6. Benchmark chart và figure định lượng

Slide định lượng chính nên **ưu tiên dùng Figure 4 trong proposal cuối** vì đây là hình multi-metric comparison đã khớp với `NTN_IT_CT.pdf`. Figure 2 trong proposal cuối vẫn rất hữu ích, nhưng nên xem như một trade-off visual bổ sung hoặc backup visual để nhấn mạnh frontier `accuracy vs. efficiency`.

Nếu cần thêm một chart phụ dễ nhìn khi nói, nên dùng **bubble chart/Pareto chart**, không dùng radar chart nếu sợ khó đọc.

Trục:

- X: FPS, càng phải càng tốt.
- Y: MPJPE, càng thấp càng tốt.
- Bubble size: Params hoặc Memory.

Highlight:

- M0: baseline, chậm và sai hơn.
- M3: best accuracy.
- M4: best trade-off.

Câu nói kèm:

> “M3 tells us that Rectified Flow improves accuracy. M4 tells us that this improvement can be made lightweight and fast.”

**Khả năng thực thi hiện tại**

- **Làm ngay được:** `Figure 4` đã có sẵn trong proposal cuối, và `figure1_teaser.*` đã có trong `work_dirs/paper_M1-5/figures`.
- `plot_teaser_figure.py` đã sẵn để render lại bubble chart nếu nhóm cần điều chỉnh nhỏ.
- **Không cần code bổ sung** cho slide định lượng chính.

**Chỉ cần code bổ sung nếu**

- nhóm muốn thêm một chart phụ hoàn toàn mới ngoài Figure 2 và Figure 4.

**Khuyến nghị thực thi**

- Slide định lượng chính dùng **Figure 4**.
- Figure 2 chỉ là visual phụ hoặc backup.

---

### 3.7. Backup visuals

Chuẩn bị các slide backup:

1. Dataset details: Person-in-WiFi 3D.
2. Full M0–M4 table.
3. Rectified Flow formula.
4. Spectral Tokenization formula.
5. Baseline architecture Person-in-WiFi 3D.
6. Responsible AI statement.

**Khả năng thực thi hiện tại**

- **Làm ngay được:** full M0–M4 table, Rectified Flow formula, Spectral Tokenization formula, Responsible AI statement.
- **Có thể làm ngay nhưng cần export thủ công từ PDF cuối:** dataset figure và baseline architecture figure nếu muốn giữ đúng hình trong appendix proposal.

**Chỉ cần code bổ sung nếu**

- nhóm muốn trích xuất tự động từng hình appendix từ proposal PDF thành asset riêng.

**Khuyến nghị thực thi**

- Không viết code mới cho backup visual ở giai đoạn này.
- Export thủ công `Figure 6` và `Figure 7` từ proposal PDF nếu thực sự cần lên backup slides.

### 3.8. Mapping figure từ proposal cuối sang slide

Để tránh lệch với `NTN_IT_CT.pdf`, cả nhóm nên khóa mapping như sau:

- **Slide 6 — Proposed Method:** dùng **Figure 1** làm overview pipeline chính.
- **Slide 7 hoặc backup method slide:** dùng **Figure 3** để giải thích chi tiết kiến trúc `M4`.
- **Slide 8 — Quantitative Results:** dùng **Figure 4** làm visual định lượng chính; **Figure 2** chỉ dùng nếu muốn nhấn mạnh trade-off frontier.
- **Slide 9 — Qualitative Results:** dùng **Figure 5** làm visual qualitative chính.

Nguyên tắc: `Figure 1` để người nghe hiểu hệ thống, `Figure 3` để đi sâu kỹ thuật, `Figure 4` để chứng minh định lượng, và `Figure 5` để chứng minh hành vi thị giác của mô hình trong cảnh khó.

### Output của giai đoạn 3

- Video hook `S23_12` presentation mode hoặc slide hook fallback.
- Problem visual Camera/Wearable/WiFi.
- Method slide dùng Figure 1 overview.
- Technical/backup method slide dùng Figure 3 detailed M4 architecture.
- Draft-to-refine visual: ảnh tĩnh hoặc Veo 3 animation 8 giây.
- Quantitative slide dùng Figure 4.
- Qualitative slide dùng Figure 5.
- Backup visuals: M0-M4 table, formulas, Responsible AI, optional dataset/baseline appendix figures.

### Deliverables khóa theo mức ưu tiên

**Bắt buộc phải hoàn thành trong giai đoạn 3**

1. Hook visual: ưu tiên video `S23_12` presentation mode; fallback là slide animation.
2. Slide problem visual.
3. Slide method dùng `Figure 1`.
4. Slide kỹ thuật/back-up method dùng `Figure 3`.
5. Slide định lượng dùng `Figure 4`.
6. Slide qualitative dùng `Figure 5`.
7. Draft-to-refine visual tối thiểu dạng ảnh tĩnh 3-stage.

**Nên hoàn thành nếu còn thời gian**

1. Draft-to-refine Veo 3 animation hoặc PowerPoint Morph.
2. Bubble chart phụ từ `figure1_teaser`.
3. Dataset backup slide.
4. Baseline architecture backup slide.
5. Audit video có `--show-unmatched` để giải thích FP/duplicate nếu bị hỏi.

**Chưa cần viết thêm code ở giai đoạn này**

1. Auto-annotated failure highlight.
2. Auto-generated draft-to-refine vector animation.
3. Auto-export appendix figures from PDF.

### 3.9. Checklist thực thi ngay

1. Render video hook `S23_12` presentation mode bằng lệnh ở mục 3.1.
2. Render thêm bản audit mode nếu cần debug FP:

```bash
python /content/Person-in-WiFi-3D/tools/analysis/render_presentation_video.py \
  --video-id S23_12 \
  --config /content/Person-in-WiFi-3D/configs/wifi/wi_tidir_wifi_transformer.py \
  --checkpoint /content/drive/MyDrive/RESEARCH/RESFES2026/Ablation_20e_RTX3090/M1_5/M3/epoch_20.pth \
  --model M3 \
  --baseline-model M0 \
  --baseline-config /content/Person-in-WiFi-3D/configs/wifi/petr_wifi.py \
  --baseline-checkpoint /content/drive/MyDrive/RESEARCH/RESFES2026/Ablation_20e_RTX3090/M1_5/M0/epoch_20.pth \
  --source-video /content/drive/MyDrive/RESEARCH/RESFES2026/S23_12/output.mkv \
  --source-time-list /content/drive/MyDrive/RESEARCH/RESFES2026/S23_12/time_list.txt \
  --start-frame 301 \
  --end-frame 349 \
  --score-thr 0.0 \
  --match-quality-thr-mm 500 \
  --show-unmatched \
  --fps 12 \
  --output /content/drive/MyDrive/RESEARCH/RESFES2026/S23_12/presentation_S23_12_M0_vs_M3_audit.mp4
```

3. Dựng slide problem bằng bảng Camera/Wearable/WiFi.
4. Gắn Figure 1 vào slide method chính.
5. Gắn Figure 3 vào backup/technical method slide.
6. Gắn Figure 4 vào slide quantitative results.
7. Gắn Figure 5 vào slide qualitative results.
8. Tạo draft-to-refine video bằng Veo 3 từ ảnh 3-stage; nếu không đạt tiêu chí nghiệm thu, dùng ảnh tĩnh hoặc Morph.
9. Kiểm tra video mở được trên Windows/PowerPoint; nếu lỗi codec, re-encode bằng H.264 `yuv420p`.
10. Chốt một câu giải thích FP: `M3` có thể nhiều duplicate/FP hơn `M0`, nhưng benchmark chính là matched MPJPE, không phải FP count trong video.
11. Gom toàn bộ visual vào một thư mục dùng cho slide deck: video hook, draft-to-refine animation/image, Figure 1, Figure 3, Figure 4, Figure 5, teaser chart.

### 3.10. Tiêu chí hoàn thành giai đoạn 3

Giai đoạn 3 được xem là hoàn thành khi có đủ các asset sau:

| Asset | Bắt buộc | Nguồn | Ghi chú |
|---|---|---|---|
| Hook video hoặc hook animation | Có | `render_presentation_video.py` hoặc PowerPoint | Dùng presentation mode, không dùng audit mode trên slide mở đầu |
| Problem visual | Có | PowerPoint | Camera vs Wearable vs WiFi |
| Figure 1 method overview | Có | Proposal/figures | Slide method chính |
| Figure 3 detailed architecture | Có | Proposal/figures | Backup hoặc technical method |
| Draft-to-refine visual | Có | Veo 3 hoặc ảnh tĩnh | Conceptual, keypoint graph, không phải anatomical skeleton |
| Figure 4 quantitative | Có | Proposal/figures | Metric chính để chứng minh M3/M4 |
| Figure 5 qualitative | Có | Proposal/figures | Dùng caption trung thực về matched pose quality |
| Audit video | Không | `--show-unmatched` | Chỉ dùng backup/Q&A |

Không chuyển sang Giai đoạn 4 nếu còn thiếu `Figure 1`, `Figure 4`, hoặc hook visual tối thiểu.

---

## Giai đoạn 4 — Xây dựng slide deck bản chính

**Thời gian đề xuất:** Ngày 7–8  
**Mục tiêu:** Hoàn thành bản slide đầu tiên có thể tập nói được.

---

### Slide 1 — Hook

**Title:**  
**Can WiFi understand human posture without cameras?**

Nội dung:

- video mở đầu;
- một câu hỏi lớn;
- tên nhóm.

Mục tiêu: gây tò mò ngay.

---

### Slide 2 — Motivation

**Title:**  
**Why camera-based sensing is not always acceptable**

Nội dung:

- Camera: accurate but privacy-sensitive.
- Wearable: intrusive.
- WiFi: already available, privacy-preserving.

Câu chốt:

> “We need a sensing method that understands posture without capturing appearance.”

---

### Slide 3 — Challenge

**Title:**  
**WiFi CSI is not an image**

Nội dung:

- CSI noisy;
- low-resolution;
- multipath;
- multi-person ambiguity.

Visual:

```text
Human motion → WiFi multipath → noisy CSI → ambiguous pose
```

---

### Slide 4 — Research Gap

**Title:**  
**Existing models guess the skeleton in one shot**

Nội dung:

- Direct regression is unstable.
- Transformer is computationally heavy.
- Multi-person scenes increase ambiguity.

Câu chốt:

> “The problem is not only prediction accuracy, but also structural stability and deployability.”

---

### Slide 5 — Key Idea

**Title:**  
**From one-shot guessing to draft-to-refine**

Nội dung:

```text
Old: CSI → Final pose
Ours: CSI → Draft pose → Flow refinement → Final pose
```

Dùng animation draft-to-refine.

Đây là slide “linh hồn” của bài.

---

### Slide 6 — Proposed Method

**Title:**  
**Proposed M4 Pipeline**

Nội dung:

```text
WiFi CSI → Spectral Tokenizer → WiMamba → Draft Pose → Rectified Flow → 3D Pose
```

Visual chính: **Figure 1** trong proposal cuối.  
Mỗi block giải thích 1 câu. Nếu còn thời gian hoặc bị hỏi sâu hơn, chuyển sang **Figure 3** ở backup slide để giải thích chi tiết M4.

---

### Slide 7 — Research Design

**Title:**  
**A controlled M0–M4 ablation ladder**

Nội dung:

| Model | Purpose |
|---|---|
| M0 | Baseline |
| M1 | Test spectral representation |
| M2 | Test efficient sequence modeling |
| M3 | Test flow-based refinement |
| M4 | Final deployment-oriented model |

Gắn với RBL:

> “Each stage tests one hypothesis.”

---

### Slide 8 — Quantitative Results

**Title:**  
**Flow improves accuracy. Mamba improves deployability.**

Nội dung bảng rút gọn:

| Model | MPJPE ↓ | FPS ↑ | Params ↓ | Memory ↓ |
|---|---:|---:|---:|---:|
| M0 | 169.34 | 45.69 | 13.13M | 155.60 MB |
| M3 | **151.99** | 138.79 | 7.06M | 38.49 MB |
| M4 | 159.00 | **159.85** | **5.83M** | **27.02 MB** |

Visual chính: **Figure 4** trong proposal cuối.  
Visual bổ sung nếu cần: **Figure 2** để nhấn mạnh `M3 = best accuracy`, `M4 = best trade-off`.

---

### Slide 9 — Qualitative Results

**Title:**  
**More stable skeletons in crowded scenes**

Nội dung:

- Ground Truth;
- M0;
- M3;
- M4.

Dùng case 3-person nếu rõ nhất.

Visual chính: **Figure 5** trong proposal cuối.

Câu nói:

> “M0 often produces distorted skeletons in crowded settings, while flow-based models produce more coherent body structures.”

---

### Slide 10 — Impact

**Title:**  
**Toward privacy-preserving indoor human sensing**

Nội dung:

Ứng dụng:

- smart home;
- elderly care;
- rehabilitation;
- fall/activity monitoring;
- privacy-sensitive spaces.

Câu chốt:

> “This is not about replacing cameras everywhere. It is about enabling sensing where cameras are not acceptable.”

---

### Slide 11 — Conclusion

**Title:**  
**What we learned**

Nội dung:

```text
We asked: Can WiFi estimate 3D human poses without cameras?
We proposed: Draft-to-Refine Rectified Flow with WiMamba.
We showed: Better baseline accuracy, higher speed, lower model cost.
```

Câu kết:

> “Our work brings WiFi-based 3D pose estimation closer to real-time, privacy-preserving deployment.”

---

### Nguyên tắc thiết kế slide

- Mỗi slide chỉ có **1 thông điệp chính**.
- Không để quá 40 chữ trên slide.
- Số liệu quan trọng phải được highlight.
- Không copy nguyên đoạn từ proposal.
- Không nhồi công thức vào slide chính.
- Công thức để backup slide.
- Hình phải lớn, dễ nhìn từ xa.
- Baseline M0 nên dùng màu xám/đỏ.
- Proposed M4 nên dùng màu xanh.
- M3 nên dùng màu vàng/cam để thể hiện “accuracy upper bound”.

### Output của giai đoạn 4

- Slide deck bản 1.
- Backup slide deck.
- File video/animation đã nhúng.
- Bảng số liệu đã highlight.
- Flow trình bày 10 phút.

---

## Giai đoạn 5 — Viết script và speaker notes

**Thời gian đề xuất:** Ngày 9  
**Mục tiêu:** Mỗi người biết chính xác mình phải nói gì, chuyển ý ra sao, không nói quá thời gian.

### Cách viết script

Không nên viết script quá dài để học thuộc từng chữ. Nên viết theo dạng:

- câu mở slide;
- 2–3 ý chính;
- câu chuyển sang slide tiếp theo.

Ví dụ:

#### Slide 1 script

> “A camera can estimate human pose, but it also captures human identity. Our project asks a different question: can we estimate 3D human poses using only WiFi signals?”

#### Slide 5 script

> “The key idea is simple. Existing models try to predict the final skeleton in one step. But CSI is noisy and ambiguous, so one-shot regression often becomes unstable. We instead generate a coarse draft pose first, then use Rectified Flow to learn how to correct it toward a realistic skeleton.”

#### Slide 8 script

> “The results show two important findings. First, M3 achieves the lowest MPJPE, reducing the error from 169.34 mm to 151.99 mm. This confirms the benefit of Rectified Flow. Second, M4 achieves the best deployment trade-off, reaching 159.85 FPS with only 5.83 million parameters and 27.02 MB peak memory.”

### Chia thời gian nói

| Phần | Slide | Thời gian |
|---|---|---:|
| Hook + motivation | 1–2 | 1:30 |
| Challenge + gap | 3–4 | 1:30 |
| Key idea + method | 5–6 | 2:00 |
| Research design | 7 | 1:00 |
| Results | 8–9 | 2:00 |
| Impact + conclusion | 10–11 | 2:00 |
| Tổng | 11 slides | 10:00 |

Mục tiêu khi tập là **9:30**, để lúc thi có dư 30 giây.

### Output của giai đoạn 5

- Script từng slide.
- Người nói từng slide.
- Câu chuyển giữa các speaker.
- Bản timing 9:30–10:00.
- Câu mở bài và câu kết bài đã thuộc.

---

## Giai đoạn 6 — Luyện trình bày và tối ưu delivery

**Thời gian đề xuất:** Ngày 10–12  
**Mục tiêu:** Biến nội dung tốt thành phần trình bày tự tin, mạch lạc, không đọc slide.

### Buổi luyện 1 — Hiểu flow

Mỗi người nói phần của mình, chưa cần đúng thời gian tuyệt đối.

Sau buổi này cần trả lời:

- Slide nào đang khó hiểu?
- Người nghe có hiểu Rectified Flow không?
- Có slide nào quá nhiều chữ không?
- Có chỗ nào chuyển ý gượng không?

### Buổi luyện 2 — Canh thời gian

Bấm giờ nghiêm túc.

Mục tiêu:

- không vượt 10 phút;
- không nói quá nhanh;
- không bỏ slide quan trọng.

Nếu vượt thời gian, cắt bớt:

- literature review;
- công thức;
- chi tiết M1/M2;
- giải thích quá sâu về Mamba.

Không được cắt:

- hook;
- key idea;
- M0–M4;
- benchmark;
- qualitative result;
- conclusion.

### Buổi luyện 3 — Luyện chuyển speaker

Câu chuyển Speaker 1 sang Speaker 2:

> “Now that we understand why WiFi pose estimation is difficult, let me hand over to my teammate to explain our draft-to-refine solution.”

Câu chuyển Speaker 2 sang Speaker 3:

> “After designing the method, the next question is whether each component really works. My teammate will present our ablation results.”

### Buổi luyện 4 — Quay video

Quay lại toàn bộ bài trình bày.

Kiểm tra:

- Có nhìn màn hình quá nhiều không?
- Tay có bị cứng không?
- Giọng có đều đều không?
- Có nói quá nhanh ở phần kỹ thuật không?
- Có giải thích số liệu rõ không?
- Câu kết có đủ mạnh không?

### Buổi luyện 5 — Mock presentation trước người ngoài nhóm

Mời 1–2 bạn không quá hiểu dự án nghe thử.

Hỏi họ 5 câu:

1. Bạn có hiểu dự án làm gì không?
2. Bạn có hiểu vì sao dùng WiFi không?
3. Bạn có hiểu khác biệt giữa M0 và M4 không?
4. Bạn nhớ con số nào nhất?
5. Phần nào khó hiểu nhất?

Nếu người ngoài nhóm không hiểu, hội đồng cũng có thể không hiểu.

### Output của giai đoạn 6

- Bản trình bày ổn định dưới 10 phút.
- Mỗi speaker nói tự nhiên, không đọc.
- Slide đã cắt gọn.
- Câu chuyển mượt.
- Video rehearsal để tự đánh giá.

---

## Giai đoạn 7 — Chuẩn bị Q&A chiến lược

**Thời gian đề xuất:** Ngày 13–14  
**Mục tiêu:** Chuẩn bị câu trả lời cho các câu hỏi khó, biến Q&A thành điểm mạnh.

---

### Nhóm câu hỏi 1 — Về motivation

#### Q1. Why WiFi instead of cameras?

> Cameras provide rich visual information, but they also capture identity and appearance. WiFi CSI does not contain visual identity, so it is more suitable for privacy-sensitive indoor environments such as elderly care, bedrooms, hospitals, or smart homes.

#### Q2. Does WiFi fully replace cameras?

> No. Our goal is not to replace cameras in all scenarios. Our goal is to provide an alternative for spaces where cameras are not acceptable due to privacy or occlusion concerns.

---

### Nhóm câu hỏi 2 — Về method

#### Q3. Why Rectified Flow?

> CSI is noisy and ambiguous. Direct regression forces the model to jump from signal to final pose in one step. Rectified Flow allows the model to start from a coarse draft and learn a correction trajectory toward the ground-truth pose, which is more suitable for ambiguous observations.

#### Q4. How is Rectified Flow different from Diffusion?

> Diffusion usually starts from random noise and may require many denoising steps. Our draft-to-refine Rectified Flow starts from a conditional draft pose, so it can use fewer inference steps and remain efficient.

#### Q5. Why Mamba?

> Transformer self-attention has quadratic complexity with sequence length. WiFi CSI can form long token sequences, so WiMamba is used as a linear-time sequence modeling alternative to improve efficiency.

---

### Nhóm câu hỏi 3 — Về kết quả

#### Q6. Why is M3 more accurate than M4?

> M3 uses the Transformer-based flow variant, which preserves stronger representation capacity, so it achieves the lowest MPJPE. M4 uses WiMamba to reduce computation, so it sacrifices a small amount of accuracy but gains much higher speed and lower memory.

#### Q7. If M3 has better MPJPE, why choose M4?

> M3 is the accuracy upper bound. M4 is the deployment-oriented model. M4 is selected because it offers the best accuracy-efficiency trade-off: 159.85 FPS, 5.83M parameters, and 27.02 MB memory, while still improving over the M0 baseline.

#### Q8. Why does M2 not improve MPJPE?

> This is an important finding from our ablation. Mamba alone mainly improves efficiency, but it does not fully resolve CSI ambiguity. The accuracy gain comes from Rectified Flow. The final M4 combines Flow for refinement and Mamba for efficiency.

---

### Nhóm câu hỏi 4 — Về research validity

#### Q9. How do you ensure fair comparison?

> All variants are evaluated under the same Person-in-WiFi 3D protocol, same train/test split, and same benchmark pipeline. The ablation ladder changes one major component at a time to isolate its effect.

#### Q10. Is the result final?

> The current table is a preliminary frozen local snapshot. For the final report, we plan to strengthen reliability through repeated evaluation and additional structural metrics such as PA-MPJPE and Bone Length Consistency.

#### Q11. What are the limitations?

> The current study uses an existing benchmark dataset and does not yet evaluate cross-device or cross-environment generalization extensively. It also focuses on up to three-person scenarios. These are the next steps for future work.

---

### Nhóm câu hỏi 5 — Về đóng góp của nhóm

#### Q12. What exactly did your team contribute?

> We built a controlled M0–M4 research pipeline, introduced the draft-to-refine Rectified Flow formulation, integrated it with Spectral Tokenization and WiMamba, and evaluated the trade-off between accuracy and deployability through ablation.

#### Q13. What is inherited from prior work?

> The task definition and benchmark protocol come from Person-in-WiFi 3D. Our contribution is the architectural redesign and ablation-based evaluation around spectral representation, WiMamba sequence modeling, and Rectified Flow refinement.

---

### Output của giai đoạn 7

- Q&A bank ít nhất 30 câu.
- Mỗi câu có câu trả lời 30 giây và 60 giây.
- Phân công ai trả lời nhóm câu hỏi nào.
- Backup slide tương ứng với từng nhóm câu hỏi.
- Luyện mock Q&A ít nhất 2 lần.

---

## Giai đoạn 8 — Hoàn thiện bản cuối và chuẩn bị ngày thi

**Thời gian đề xuất:** 1–2 ngày cuối  
**Mục tiêu:** Không để lỗi kỹ thuật làm hỏng phần trình bày.

### Checklist slide

- Không lỗi font.
- Không lỗi video.
- Có bản PPTX.
- Có bản PDF backup.
- Có video tách riêng nếu PowerPoint lỗi.
- Hình đủ nét.
- Số liệu khớp proposal.
- Không có typo ở title, tên nhóm, tên tác giả.
- Không có câu overclaim như “prove absolutely”, “perfect”, “all you need” nếu không giải thích được.

### Checklist thiết bị

- Laptop chính.
- Laptop backup hoặc USB backup.
- Sạc laptop.
- HDMI adapter.
- Clicker.
- File offline.
- Video offline.
- PDF backup.
- Internet không được xem là bắt buộc.

### Checklist người trình bày

- Mỗi người thuộc câu mở phần mình.
- Mỗi người biết slide trước và sau phần mình.
- Mỗi người biết khi nào dừng.
- Một người chịu trách nhiệm điều khiển slide.
- Một người chịu trách nhiệm Q&A kỹ thuật.
- Một người chịu trách nhiệm Q&A impact/motivation.

### Checklist ngày thi

Trước khi lên trình bày:

- mở sẵn file;
- test video;
- test âm thanh nếu có;
- test clicker;
- uống nước;
- không sửa slide phút cuối trừ lỗi nghiêm trọng;
- thống nhất người trả lời câu hỏi đầu tiên.

---

## Timeline thực hiện mẫu trong 14 ngày

| Ngày | Giai đoạn | Công việc chính | Output |
|---:|---|---|---|
| 1 | Chiến lược | Chốt title, message, phân công | Outline + message |
| 2 | Nội dung | Chuẩn hóa RQ, contribution, claim | Nội dung nghiên cứu |
| 3 | Số liệu | Kiểm tra bảng M0–M4, giải thích M3/M4 | Benchmark final |
| 4 | Visual | Làm video hook, problem visual | Video/ảnh mở đầu |
| 5 | Visual | Làm pipeline, draft-to-refine animation | Method visual |
| 6 | Visual | Làm chart, qualitative slide | Results visual |
| 7 | Slide | Làm deck bản 1 | PPT v1 |
| 8 | Slide | Chỉnh design, thêm backup slides | PPT v2 |
| 9 | Script | Viết speaker notes | Script 10 phút |
| 10 | Rehearsal | Tập lần 1–2, canh thời gian | Bản nói ổn |
| 11 | Rehearsal | Quay video, chỉnh delivery | Bản nói tốt |
| 12 | Mock | Trình bày cho người ngoài nhóm | Feedback |
| 13 | Q&A | Chuẩn bị Q&A bank, backup slides | Q&A bank |
| 14 | Final | Tổng duyệt, xuất file, kiểm tra thiết bị | Final deck |

Nếu chỉ còn ít ngày, rút gọn như sau:

| Thời gian còn lại | Ưu tiên |
|---|---|
| 3 ngày | Slide chính + script + Q&A top 15 |
| 5 ngày | Thêm visual/animation + rehearsal |
| 7 ngày | Có mock presentation + backup slide |
| 14 ngày | Làm đủ toàn bộ kế hoạch |

---

## Phân công công việc đề xuất cho nhóm 3 người

### Nguyen Vinh Nghi — Leader

Phụ trách:

- chốt narrative tổng thể;
- giải thích M4, Mamba, efficiency;
- kiểm soát timing;
- trả lời Q&A về architecture và trade-off.

Deliverables:

- slide method;
- slide M0–M4;
- câu trả lời M3 vs M4;
- rehearsal coordination.

### Le Ngoc Anh Thu

Phụ trách:

- motivation, problem, application;
- thiết kế slide;
- visual/video hook;
- chỉnh ngôn ngữ thuyết trình.

Deliverables:

- slide 1–3;
- video mở đầu;
- slide impact;
- script mở bài.

### Nguyen Ngo Nhat Nam

Phụ trách:

- Rectified Flow;
- benchmark/results;
- qualitative comparison;
- Q&A kỹ thuật về flow và evaluation.

Deliverables:

- slide key idea;
- slide quantitative results;
- slide qualitative results;
- backup công thức Rectified Flow.

---

## Mức độ ưu tiên công việc

### Bắt buộc phải có

1. Slide hook thật ấn tượng.
2. Slide problem rõ.
3. Slide key idea draft-to-refine.
4. Slide pipeline sạch.
5. Slide M0–M4.
6. Slide benchmark M0/M3/M4.
7. Slide qualitative 3D skeleton.
8. Script 10 phút.
9. Q&A top 15.
10. PDF backup.

### Nên có

1. Video opening.
2. Draft-to-refine animation.
3. Bubble chart.
4. Backup formula slides.
5. Mock presentation với người ngoài nhóm.

### Có thì rất tốt

1. Demo interactive.
2. 3D rotating skeleton video.
3. Multi-seed result.
4. Failure case slide.
5. Edge deployment estimation.

---

## Bản kế hoạch hành động ngay từ hôm nay

### Trong 24 giờ tới

Cả nhóm cần hoàn thành:

1. Chốt title trình bày.
2. Chốt 11 slide.
3. Chốt speaker.
4. Gom tất cả hình từ proposal:
   - Figure 1 overview pipeline;
   - Figure 3 detailed M4 architecture;
   - Figure 4 multi-metric comparison;
   - Figure 5 qualitative comparison;
   - Appendix A1–A3;
   - dataset Figure 6;
   - baseline Figure 7.
5. Tạo bảng benchmark rút gọn M0/M3/M4.
6. Viết bản script thô.

### Trong 48 giờ tới

Cần có:

1. Slide deck bản 1.
2. Slide 6 đã gắn Figure 1 và backup Figure 3.
3. Slide 8 đã gắn Figure 4, Slide 9 đã gắn Figure 5.
4. Video hook hoặc animation mock.
5. Script từng speaker.
6. Tập thử lần đầu.

### Trong 72 giờ tới

Cần có:

1. Deck bản 2.
2. Bài nói dưới 10 phút.
3. Q&A top 15.
4. Backup slides.
5. Một lần mock presentation.

---

## Kết luận

Kế hoạch này không chỉ nhằm “làm slide đẹp”. Mục tiêu là biến dự án thành một phần trình bày có logic thắng giải:

> **Vấn đề thật → insight thông minh → phương pháp rõ → thí nghiệm có kiểm soát → số liệu mạnh → ứng dụng thực tế → trả lời Q&A chắc.**

Nếu nhóm làm đúng theo các giai đoạn trên, phần trình bày sẽ không bị rơi vào kiểu “trình bày model kỹ thuật”, mà trở thành một câu chuyện nghiên cứu hoàn chỉnh:

> **WiFi can estimate 3D human poses without cameras, and our draft-to-refine flow model makes it accurate, lightweight, and practical.**
