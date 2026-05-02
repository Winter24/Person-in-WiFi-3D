# Plan Presentation — ResFes 2026

## Mục tiêu tổng thể

Dự án của nhóm đã qua vòng sơ khảo và bước vào vòng chấm điểm thuyết trình. Mục tiêu của kế hoạch này là chuẩn bị một phần trình bày **ấn tượng, rõ ràng, có chiều sâu nghiên cứu, kiểm soát tốt Q&A, và tối đa hóa cơ hội đạt giải Nhất**.

Thông điệp trung tâm của bài thuyết trình:

> **We turn WiFi pose estimation from one-shot guessing into efficient draft-to-refine trajectory refinement.**

Ba điều hội đồng cần nhớ sau phần trình bày:

1. **Problem:** Camera-based pose estimation chính xác nhưng không phù hợp trong nhiều không gian riêng tư.
2. **Innovation:** Nhóm thay direct one-shot regression bằng **Draft-to-Refine Rectified Flow**, và giải thích rõ sự thay đổi này so với baseline `M0`.
3. **Evidence:** `M3` đạt accuracy tốt nhất; `M4` đạt trade-off triển khai tốt nhất với tốc độ cao, ít tham số và memory thấp.

---

## Narrative khóa cho toàn bộ deck

Deck chính thức được khóa theo narrative:

```text
M0 baseline
-> M1 spectral representation
-> M2 efficient sequence modeling
-> M3/M4 draft-to-refine flow refinement
-> quantitative + qualitative evidence
```

Nguyên tắc quan trọng:

- Main deck dùng **13 slides**, không còn bám cấu trúc 11-slide cũ.
- Công thức và loss không để toàn bộ ở backup nữa.
- Main deck chỉ kể câu chuyện `M0-M4`.
- Repository có thể có nhánh `BoneLengthLoss`, nhưng **deck này coi snapshot thuyết trình là runtime no-bone-loss cho M0-M4**.
- Không nói `M4` thắng mọi metric.
- Không đồng nhất false positive trong video với matched MPJPE chính thức.

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
Thay vì bắt mô hình đoán pose cuối cùng một lần từ CSI nhiễu, nhóm tách bài toán thành các giai đoạn: baseline direct regression, cải thiện representation, cải thiện sequence modeling, và cuối cùng đổi objective sang trajectory refinement bằng Rectified Flow.

**Message 3 — Evidence**  
`M3` là model tốt nhất về accuracy; `M4` là model tốt nhất về deployment trade-off với FPS cao, params thấp và memory thấp.

### 3. Chốt vai trò từng thành viên

Nếu 3 người cùng trình bày:

| Người | Phần phụ trách | Mục tiêu |
|---|---|---|
| Speaker 1 | Hook, problem, baseline framing | Tạo hứng thú và dựng baseline |
| Speaker 2 | Stage upgrades: loss, spectral, WiMamba, flow | Thể hiện chiều sâu kỹ thuật |
| Speaker 3 | Results, impact, conclusion, Q&A close | Chứng minh giá trị và chốt bài |

Nếu 2 người trình bày:

| Người | Phần phụ trách |
|---|---|
| Speaker 1 | Hook → motivation → challenge → `M0` baseline |
| Speaker 2 | Stage upgrades → results → conclusion |

### Output của giai đoạn 1

- Tên bài thuyết trình chính thức.
- 3 thông điệp chính.
- Phân công người nói.
- Outline **13 slide**.
- Quyết định bài nói dùng tiếng Anh hoàn toàn hay tiếng Anh có giải thích nội bộ bằng tiếng Việt khi luyện.

---

## Giai đoạn 2 — Chuẩn hóa số liệu, bằng chứng và nội dung nghiên cứu

**Thời gian đề xuất:** Ngày 2–3  
**Mục tiêu:** Đảm bảo mọi số liệu, claim và kết luận trong slide đều chính xác, nhất quán với proposal và code hiện tại.

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
- Có dùng cùng checkpoint và evaluation script không?
- Có cần ghi “preliminary frozen local snapshot” không?

Nếu chưa có multi-seed, trên slide nên dùng phrasing:

> **Preliminary benchmark results**

Không nên nói:

> “Final results prove…”

Nên nói:

> “These preliminary results indicate…”

### 2. Chốt cách giải thích `M3` và `M4`

`M3` có MPJPE tốt nhất: **151.99 mm**.  
`M4` có tốc độ, params, memory tốt nhất: **159.85 FPS, 5.83M params, 27.02 MB**.

Cách nói đúng:

> **M3 is the accuracy winner. M4 is the practical deployment winner.**

Không nên nói:

> “M4 is the best model” một cách chung chung.

Nên nói:

> “M4 is selected as the final deployment-oriented model because it provides the best accuracy-efficiency trade-off.”

### 3. Chuẩn hóa research questions

Đưa vào slide dạng ngắn:

**RQ1:** Can Spectral Tokenization improve CSI representation?  
**RQ2:** Can WiMamba improve efficiency without collapsing the pose pipeline?  
**RQ3:** Can Rectified Flow improve pose accuracy and structural fidelity beyond direct regression?

### 4. Chốt contribution

Trong slide chỉ nên có 3 contribution:

1. **A clear `M0-M4` controlled evidence ladder** for WiFi-based 3D multi-person pose estimation.
2. **Efficient CSI encoding** using Spectral Tokenization and WiMamba.
3. **Draft-to-Refine Rectified Flow** that changes the learning target from direct coordinates to trajectory refinement.

### Output của giai đoạn 2

- Bảng số liệu đã xác nhận.
- Câu giải thích chính thức về `M3` vs `M4`.
- 3 research questions.
- 3 contributions.
- Danh sách claim nào được nói mạnh, claim nào phải nói thận trọng.

---

## Giai đoạn 3 — Chuẩn bị hình ảnh, video và visual kỹ thuật

**Thời gian đề xuất:** Ngày 4–6  
**Mục tiêu:** Tạo phần visual đủ mạnh để người không chuyên vẫn hiểu, nhưng không đánh rơi phần kỹ thuật cốt lõi.

Đây là giai đoạn cực kỳ quan trọng vì đề tài kỹ thuật sâu. Nếu chỉ nói Transformer, Mamba, CSI, Rectified Flow, hội đồng có thể bị quá tải. Visual phải làm nhiệm vụ “dịch kỹ thuật thành trực giác”, đồng thời support đúng 13-slide narrative.

### Trạng thái thực thi hiện tại

Đối chiếu với code và asset hiện có trong repo, giai đoạn 3 hiện được chia thành bốn nhóm:

- **Làm ngay được bằng asset hiện có:** Figure 1 overview pipeline, Figure 3 detailed M4 architecture, Figure 4 multi-metric comparison, Figure 5 qualitative comparison, bảng `M0-M4` rút gọn, slide problem dạng bảng.
- **Làm ngay được bằng script hiện có:** video hook từ `tools/analysis/render_presentation_video.py`, Figure 5 từ `tools/analysis/render_qualitative_figure.py`, bubble/Pareto chart từ `tools/analysis/plot_teaser_figure.py`.
- **Làm ngay được nhưng chủ yếu là thao tác slide thủ công:** slide problem, direct-regression fail callout, draft-to-refine Morph animation, slide baseline architecture, slide equations, slide Responsible AI.
- **Chỉ cần code bổ sung nếu muốn tự động hóa sâu hơn:** auto-annotate failure cases, auto-generate draft-to-refine vector animation, redraw clean baseline architecture figure từ appendix proposal.

Nguyên tắc thực thi:

- Ưu tiên tái sử dụng asset và script đã có.
- Chỉ viết thêm code khi phần visual đó thực sự không thể hoàn thành đủ tốt bằng PowerPoint hoặc export thủ công.
- Công thức `M0 loss`, `flow update`, `flow matching loss` phải được thiết kế **dễ đọc**, không biến thành slide giấy trắng đầy ký hiệu.

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

Mode video cần khóa:

- **Presentation mode:** không bật `--show-unmatched`, dùng `--score-thr 0.0`, `--match-quality-thr-mm 500`, và chọn test segment.
- **Audit mode:** bật `--show-unmatched` để nhìn rõ FP và duplicate predictions; chỉ dùng cho backup và Q&A.

### 3.2. Hình “Problem”

Chuẩn bị một slide so sánh:

| Camera | Wearable | WiFi |
|---|---|---|
| Captures appearance | Requires device | No visual identity |
| Privacy concern | User compliance | Passive sensing |
| Sensitive spaces issue | Inconvenient | Suitable for indoor monitoring |

### 3.3. Hình “Direct regression fails”

Chuẩn bị một case 2-person hoặc 3-person:

- Ground Truth
- M0 baseline
- M3
- M4

Cần khoanh đỏ skeleton `M0` bị lệch, méo, collapsed.

Mục tiêu:

> Hội đồng thấy baseline direct regression dễ tạo skeleton méo trong cảnh đông người, còn flow-based refinement cải thiện matched pose quality.

### 3.4. Hình pipeline chính

Figure chính vẫn nên bám proposal:

- **Figure 1**: overview pipeline
- **Figure 3**: detailed `M4` architecture
- **Figure 4**: quantitative evidence
- **Figure 5**: qualitative evidence

Nhưng narrative 13-slide có thêm các visual kỹ thuật bắt buộc:

- một hình baseline `M0` architecture đơn giản cho Slide 4
- một slide công thức `M0` loss cho Slide 5
- một slide công thức Spectral Tokenizer cho Slide 6
- một slide công thức WiMamba + complexity cho Slide 7
- một slide flow update cho Slide 8
- một slide flow matching loss cho Slide 9

### 3.5. Animation “Draft-to-Refine”

Visual này vẫn ưu tiên cao nhất.

Mục tiêu trực giác:

```text
Draft Pose X0 + predicted velocity v -> Refined Pose X1
```

Điểm cần giữ nhất quán:

- skeleton là **keypoint graph**, không phải bộ xương giải phẫu
- màu xám cho draft
- cyan cho velocity
- xanh cho refined pose

### 3.6. Visual định lượng

Slide định lượng chính vẫn nên ưu tiên:

- **Figure 4** trong proposal cuối

Nhưng vì deck mới nhấn theo giai đoạn, slide kết quả nên có thêm:

- callout `M3 = best MPJPE`
- callout `M4 = best trade-off`
- nếu còn chỗ: chênh lệch so với `M0`

### 3.7. Visual backup

Backup slides nên còn:

1. Detailed `M4` architecture
2. Full `M0-M4` variant summary
3. False positive and duplicate query explanation
4. Runtime note on bone-loss-off snapshot
5. Responsible AI statement

Lưu ý:

- `Rectified Flow formula` và `Spectral Tokenization formula` không còn chỉ là backup. Chúng đã đi lên main deck.

### Output của giai đoạn 3

- Hook video hoặc hook animation
- Problem slide visual
- `M0` baseline architecture visual
- Draft-to-refine visual
- Figure 4 quantitative
- Figure 5 qualitative
- Backup visual set cho Q&A

---

## Giai đoạn 4 — Xây dựng slide deck bản chính

**Thời gian đề xuất:** Ngày 7–9  
**Mục tiêu:** Hoàn thành bản slide đầu tiên có thể tập nói được.

### Outline chính thức: 13-slide deck

| Slide | Tên | Vai trò chính |
|---|---|---|
| 1 | Hook | Mở bài bằng privacy-preserving sensing |
| 2 | Motivation | Camera vs Wearable vs WiFi |
| 3 | Challenge + Gap | CSI ambiguity + brittleness của one-shot regression |
| 4 | Baseline `M0` Architecture | Xác định baseline rõ ràng |
| 5 | Baseline `M0` Loss | Xác định objective direct regression |
| 6 | Stage 1: Spectral Tokenizer | Contribution về representation |
| 7 | Stage 2: WiMamba | Contribution về efficiency |
| 8 | Stage 3: Flow Head | Contribution về draft-to-refine mechanism |
| 9 | Stage 3: Flow Matching Loss | Contribution về objective |
| 10 | `M0-M4` Stage Summary | Controlled ablation ladder |
| 11 | Quantitative Results | Figure 4 + M3/M4 interpretation |
| 12 | Qualitative Results | Figure 5 + matched pose quality |
| 13 | Conclusion + Impact | 3 takeaways + application context |

### Nguyên tắc thiết kế deck

- Một slide, một thông điệp.
- Không nhồi cả architecture và loss vào cùng một slide.
- Main deck phải trả lời được:
  - baseline `M0` là gì?
  - `M1` thay đổi representation như thế nào?
  - `M2` thay đổi sequence modeling như thế nào?
  - `M3/M4` thay đổi objective như thế nào?
- `M0-M4` là câu chuyện chính; không kéo bone loss vào main narrative.

### Build order khuyến nghị

1. Tạo Slide 11 và Slide 12 trước để khóa evidence.
2. Tạo Slide 4 và Slide 5 để khóa baseline.
3. Tạo Slide 6 đến Slide 9 để khóa các contribution kỹ thuật.
4. Tạo Slide 10 để tóm tắt ladder.
5. Tạo Slide 1 đến Slide 3 để framing.
6. Tạo Slide 13 để chốt bài.
7. Thêm backup slides.
8. Chạy timing pass 10 phút.

### Output của giai đoạn 4

- Slide deck bản 1
- Backup slide deck
- Asset folder cho toàn bộ bài nói

---

## Giai đoạn 5 — Viết script và speaker notes

**Thời gian đề xuất:** Ngày 10–11  
**Mục tiêu:** Biến deck tốt thành bài nói mạch lạc, không đọc slide.

Timing mục tiêu:

| Phần | Slides | Thời gian |
|---|---|---:|
| Hook + motivation | 1-2 | 1:10 |
| Problem framing | 3-5 | 1:50 |
| Architectural stages | 6-10 | 4:05 |
| Results | 11-12 | 2:10 |
| Conclusion | 13 | 0:35 |
| Tổng | 13 slides | 10:00 |

### Nội dung script bắt buộc phải rõ

- Slide 4: `M0` không phải strawman; đó là baseline nghiêm túc
- Slide 5: `M0` học direct coordinates sau Hungarian matching
- Slide 6: Spectral Tokenizer cải thiện representation chứ chưa đổi objective
- Slide 7: WiMamba chủ yếu giải quyết efficiency
- Slide 8: flow head đổi prediction process
- Slide 9: flow loss đổi learning target
- Slide 11: `M3` là accuracy winner, `M4` là trade-off winner
- Slide 12: qualitative chỉ là selected challenging cases, không phải universal proof

### Output của giai đoạn 5

- Script từng slide
- Speaker notes từng slide
- Handoff lines giữa các speaker
- Bản rút gọn nếu bị quá giờ

---

## Giai đoạn 6 — Rehearsal và Q&A

**Thời gian đề xuất:** Ngày 12–14  
**Mục tiêu:** Biến nội dung tốt thành phần trình bày tự tin, mạch lạc, không đọc slide.

### Checklist rehearsal

- Có slide nào quá nhiều chữ không?
- Có slide nào công thức khó đọc không?
- Người nghe có hiểu sự khác nhau giữa `M0` và `M4` không?
- Người nghe có hiểu vì sao cần cả Slide 8 và Slide 9 không?
- Có đang lỡ nói `M4` là model tốt nhất mọi mặt không?

### Q&A phải chuẩn bị sẵn

1. Why not keep the 11-slide simpler story?
2. Why do we need a separate baseline loss slide?
3. Why is `M3` more accurate than `M4`?
4. If `M3` is more accurate, why choose `M4`?
5. What is the difference between direct regression and flow matching?
6. Why can some videos show duplicate predictions?
7. Is bone loss part of the main model?

### Output của giai đoạn 6

- Mock presentation ổn định
- Q&A bank
- Backup slides sẵn sàng
- PDF backup

---

## Mapping tài liệu hiện tại

Sau khi đồng bộ, ba file sau phải nhất quán với nhau:

- [phase4_full_slide_detail.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase4_full_slide_detail.md): blueprint chi tiết của 13-slide deck
- [phase5_speaker_script.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase5_speaker_script.md): script nói miệng cho 13 slide
- [2026-05-01-technical-slide-augmentation-plan.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/2026-05-01-technical-slide-augmentation-plan.md): lý do kỹ thuật phải kéo formula và loss lên main story

---

## Một câu chốt cho cả nhóm

Kế hoạch này không chỉ nhằm “làm slide đẹp”. Mục tiêu là biến dự án thành một phần trình bày có logic thắng giải:

- baseline rõ
- contribution tách theo giai đoạn
- công thức đủ để thuyết phục
- evidence đủ để chốt `M3` và `M4`

Nếu hội đồng nhớ được rằng **chúng ta không chỉ thêm module mới, mà đã đổi bài toán từ direct regression sang trajectory refinement**, thì narrative này đã thành công.
