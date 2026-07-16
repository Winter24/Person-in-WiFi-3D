# Academic Paper Review Report

**Paper:** *Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures*
**File:** `paper_assets/manuscript_latex/resfes2026_witidar/main.tex`
**Authors:** Nguyen Vinh Nghi, Le Ngoc Anh Thu, Nguyen Ngo Nhat Nam, Tran Ngoc Hoang (FPT University, HCM)
**Target venue:** ResFes 2026 (student research festival, workshop tier)
**Review date:** 2026-06-18
**Review framework:** Multi-perspective panel (academic-paper-reviewer v1.10.0 workflow, manual execution)

---

## PHASE 0 — Field Analysis & Reviewer Panel Configuration

### Paper Profile

| Aspect | Identification |
|---|---|
| Primary discipline | WiFi/RF sensing & wireless perception |
| Secondary discipline | Efficient deep sequence modeling (Mamba SSM) + generative refinement (rectified flow) |
| Methodology type | Empirical ablation / architecture engineering on a single existing benchmark setup |
| Target venue tier | Workshop / student festival |
| Maturity | Pre-submission draft; internal 20-epoch ablation; single-seed; no external baselines |
| References | 6 entries (very thin) |
| Tables / Figures | 2 tables, 4 figures (2 imported from slide deck PDF) |

### Reviewer Panel (5 personas)

1. **R0 — Editor-in-Chief** — IEEE consumer-electronics / sensing workshop editor. Focus: venue fit, claim discipline, manuscript polish.
2. **R1 — Methodology Reviewer** — empirical DL methodology specialist. Focus: variance, seeds, ablation design, latency protocol.
3. **R2 — Domain Reviewer** — WiFi/CSI sensing expert (Doppler, multipath, antenna processing). Focus: signal-processing claims, WiFi-pose literature.
4. **R3 — Perspective Reviewer** — generative modeling expert (flow matching, diffusion refinement). Focus: rectified flow framing.
5. **R4 — Devil's Advocate** — adversarial reviewer. Focus: single-seed risk, test-time tuning, title vs claim, attribution.

---

## PHASE 1 — Five Independent Review Reports

### R0 · Editor-in-Chief Report

**Summary recommendation:** **Major Revision** (workshop scope: borderline accept after revision).

**Venue fit:** Phù hợp với workshop/student venue chủ đề WiFi sensing hoặc efficient deep learning. **Không** phù hợp với hội nghị tier cao (CVPR/NeurIPS/MobiCom) ở dạng hiện tại do (i) không có external baseline, (ii) single-seed, (iii) reference rất nghèo.

**Originality & significance:** Đóng góp khái niệm là *kết hợp* (spectral tokenizer + Mamba2 + draft-to-refine rectified flow); mỗi block đều mượn từ literature có sẵn. Đóng góp khoa học chính thực ra là *quan sát*: "2-step Euler trên checkpoint M9 tốt hơn 1-step và 4-step" — observation đáng giá nhưng nhỏ, phù hợp tier student.

**Critical concerns**
1. **Title overclaim.** "Flow is All You Need" mâu thuẫn với abstract và §Discussion line 180: paper tự thừa nhận "the defensible claim is narrower". Flow không phải "all you need" — nó là một trong ba thành phần. **Phải sửa title.**
2. **Baseline number untested externally.** Người đọc không biết M0 = 172.540 mm có khớp với Person-in-WiFi 3D paper báo cáo không.
3. **Acronym "WiTiDAR" không được giải nghĩa** ở đâu trong paper.
4. **Figures import từ `flow_slides.pdf`** (page 8, page 20) — không IEEE-style.
5. **Reference list 6 entries** — tối thiểu cần 20–25 cho workshop.

**Strengths warranting acceptance after revision**
- Trình bày trung thực: nói rõ internal ablation, có Limitations section.
- Câu chuyện M0→M9_RF2 traceable, có counter-evidence (T_FW2 retrain *không* giúp).
- Hiệu năng có ý nghĩa: 72.45% giảm tham số, ~2× FPS (nếu kết quả không phải seed-noise).

---

### R1 · Methodology Reviewer Report

**Scores:** Rigor 4/10, Reproducibility 3/10.

**Strengths**
- Ablation path M0→M1→M7→M8→M9→M9_RF2 có structural control: mỗi bước thay đổi *một* thành phần.
- Counter-evidence được report (T_FW2_20e tệ hơn) — không cherry-pick một chiều.
- Tách "Matched MPJPE" và "Missed" cho thấy nhận thức về detection-refinement coupling.

**Critical methodological gaps**

1. **Single seed, no variance reporting.** 7.053 mm "improvement" báo cáo đến μm precision (165.487) nhưng không có ±σ. Với MPJPE scale 165–172 mm, một seed có thể dao động ±2–5 mm. Cần **tối thiểu 3 seed** cho M0, M9, M9_RF2. Nếu không, toàn bộ claim cải tiến có thể là noise.
2. **20-epoch protocol có thể là under-training.** Original Person-in-WiFi 3D có khả năng train >> 20 epoch. Cải tiến tại 20-epoch checkpoint không chứng minh cải tiến ở convergence. Cần một converge check (50 hoặc 100 epoch) cho M0 vs M9_RF2.
3. **Training hyper-parameters hoàn toàn thiếu.** Không có optimizer, LR schedule, batch size, weight decay, gradient clip, augmentation, hardware. Reproducibility không thể.
4. **Latency measurement protocol mơ hồ.** Không nói: warm-up, batch size, FP32/FP16, sync mode, GPU model. Spec rõ batch size, dtype, GPU, N warmup, N runs; báo cáo median±IQR.
5. **"Memory" column không có unit** (155.60 vs 25.29 — MB? GB? Per-sample? Peak?).
6. **Test-set tuning risk (CRITICAL).** M9_RF2 chọn 2-step solver bằng cách đánh giá nhiều solver step counts trên test set rồi chọn cái tốt nhất — test-set selection bias. Phải có validation set riêng.
7. **"Matched MPJPE" và "Missed" không được định nghĩa trong main text.** Hungarian matching threshold? OKS-based?
8. **Lỗ hổng giải thích solver behavior.** Paper nói "two steps provide a better discretization" nhưng không nghiên cứu *tại sao*. Không có RF3, RF5, không có analysis vector field magnitude.
9. **Gaps trong M0–M9 progression.** M2–M6 hoàn toàn vắng mặt trong paper, dù scripts có chúng. Cần đầy đủ trong appendix hoặc giải thích lý do bỏ qua.
10. **Significance reporting.** Không có statistical test (paired bootstrap, sign test) giữa M0 và M9_RF2.

**Methodology verdict:** Phải multi-seed + test-set discipline + training details + latency protocol mới đứng vững.

---

### R2 · Domain Reviewer Report (WiFi/CSI Sensing)

**Scores:** Domain contribution 5/10, Literature coverage 3/10.

**Strengths**
- Kế thừa setup từ Person-in-WiFi 3D, không tự "build benchmark riêng".
- Spectral tokenizer khai thác cấu trúc (9 spatial × 20 time × 60 subcarrier) hợp lý hơn linear projection thuần.
- Bone loss đã có sẵn cho anatomical constraint (qua codebase) — paper cần đề cập (hiện không nhắc).

**Critical domain concerns**

1. **"Doppler-like motion profile" claim chưa chuẩn.** §III-C áp RFFT lên embedded 256-d feature *sau khi đã linear projected* — đặc tính Doppler đã bị trộn lẫn. Doppler trong CSI thường lấy từ phase derivative hoặc FFT trên phase chuỗi sau phase sanitization.
   **Fix:** Rename "motion-aware spectral profile" hoặc áp RFFT trên amplitude/phase trực tiếp và visualize spectrogram-like.
2. **Phase handling không rõ.** WiFi CSI cần phase sanitization (CFO, SFO, PDD compensation). Paper không nói phase được xử lý thế nào.
3. **Antenna array assumption ngầm.** "Three transmit-receive groups" → 3 TX × 3 RX? 1 TX × 3 RX × 3 link? Không rõ topology.
4. **Reference list về WiFi sensing chỉ có 1 mục.** Thiếu *WiPose*, *Person-in-WiFi 2D*, *WiSPPN*, *EfficientFi*, *SignFi*, *WiDance*, phase sanitization (Tadayon, Sen).
5. **No external pose accuracy reference.** MPJPE 165 mm là *cao* (vision-based ~50–80 mm trên H36M). WiFi pose có thể phải vậy nhưng cần so với (i) Person-in-WiFi 3D paper báo cáo, (ii) camera-based trên cùng dataset.
6. **Multi-person evaluation underexplored.** 1P/2P/3P split tốt, nhưng không có association quality metric (precision/recall person detection, ID switch, OKS).
7. **Sinh hoạt thực tế.** Train/test cùng môi trường? Cross-room? Cross-subject? WiFi sensing cực kỳ environment-specific.

**Domain verdict:** Cần làm rõ signal-processing claims, mở rộng related work WiFi-sensing, bàn cross-environment generalization. Hiện tại ở mức incremental.

---

### R3 · Perspective Reviewer Report (Generative / Flow Matching)

**Scores:** Cross-disciplinary integration 6/10, Theoretical framing 4/10.

**Strengths**
- Áp dụng rectified flow như refinement operator là một hướng đẹp về khái niệm. Xa từ noise→data sang draft→target tận dụng tốt formulation của Liu et al. 2023.
- Trình bày toán học (eq. 1–4) sạch.
- Honest reporting T_FW2 retrain *không* giúp — bằng chứng quan trọng.

**Critical conceptual issues**

1. **Conceptual mismatch: 1-step training, K-step inference, non-monotonic.** Rectified flow training sample $t \sim U(0,1)$ học velocity field trên *toàn quỹ đạo*. Inference với $K$ Euler steps lý thuyết phải gần monotonic improvement với $K$ tăng. Việc *RF1 → RF2 → RF4* đi theo hướng 168.086 → 165.487 → 169.454 (**không** monotonic) chỉ ra trained velocity field *không straight*. Đây là dấu hiệu velocity field học chưa tốt, không phải là tính chất lý thuyết của rectified flow.
   **Implication:** Đóng góp thực sự gần như *test-time ensembling* hơn là flow integration đúng nghĩa.
2. **Không reflow (rectification step) thực hiện.** Rectified flow paper (Liu 2023) emphasize reflow để straighten trajectory. Paper này skip hoàn toàn → không nên gọi "rectified flow", chính xác hơn là "conditional flow matching head".
3. **Velocity network kiến trúc không mô tả trong paper.** Code có `VelocityMLP` (input=42, cond=256, hidden=512) nhưng manuscript không describe.
4. **Comparison với diffusion-based pose refinement vắng mặt.** DiffPose, PoseGen có cùng tinh thần draft-to-refine — cần ít nhất 1 đoạn trong Related Work.
5. **Conditioning analysis vắng.** Nếu set $c=0$ (unconditional flow), MPJPE thay đổi thế nào? Nếu vẫn cải tiến, flow đang học một correction prior chung chứ không phải WiFi-conditional.
6. **Step count selection on test set** (đồng quan điểm R1).

**Perspective verdict:** Framing có lỗi technical (gọi rectified flow nhưng không reflow; non-monotonic K-step cho thấy field không straight). Vẫn có giá trị nếu reframe đúng — coi đây là *learned refinement operator* parameterized dưới flow matching loss, 2-step là sweet spot empirical chứ không phải khẳng định lý thuyết.

---

### R4 · Devil's Advocate Report

**Strongest Counter-Argument**

Toàn bộ "cải tiến 7.053 mm" có thể là *artifact của (i) single seed + (ii) test-set tuning + (iii) under-training*, không phải bằng chứng cho thấy spectral+Mamba2+RF2 vượt trội PETR baseline.

Single seed dao động dễ ≥3 mm trên MPJPE scale này. Với 20 epoch chưa convergence, kiến trúc nhỏ hơn (Mamba2 3.6M params) tự nhiên fit nhanh hơn baseline lớn (PETR 13.1M params) — gap 4–5 mm tại early-stopping không tự động bền vững. Cộng thêm việc chọn solver step count (RF1 vs RF2 vs RF4) bằng cách quan sát test set, "improvement" của RF2 trên RF1 (2.599 mm) là test-set selection bias. Khi loại bỏ ba nguồn noise này, gap M0 vs M9_RF2 có thể co về <2 mm hoặc đảo chiều.

Paper *tự cung cấp* counter-evidence: M9 với 1-step RF (168.086) gần như *tệ hơn* M6 draft-only (167.573) — 1-step flow *làm hỏng* pose draft. Đến lúc 4 step (T_FW2_20e_RF4: 169.454) lại tệ hơn cả M0 spectral baseline. Cả ba kết quả này hợp nhất thành tín hiệu: velocity field trained không robust, "selected" RF2 là điểm rơi may mắn giữa "1-step không đủ" và "4-step over-shoot". Đóng góp khoa học chính của paper không phải kiến trúc mà là *quan sát rằng learned velocity field cần 2 Euler steps để hoạt động* — quan sát giá trị, nhưng không xứng title "Flow is All You Need".

**Issue List**

| Severity | Dimension | Issue | Location |
|---|---|---|---|
| **CRITICAL** | Statistical validity | Single seed; no variance; cannot conclude improvement | §V, Table 1, Table 2 |
| **CRITICAL** | Test-set tuning | RF step count chọn trên test set; biased point estimate | §IV-B, Table 2 |
| **CRITICAL** | Title overclaim | "Flow is All You Need" mâu thuẫn abstract & discussion | Title, §VI line 180 |
| MAJOR | Conceptual fidelity | "Rectified flow" không có reflow step | §III-D, §II-C |
| MAJOR | Under-training risk | 20 epoch chưa converge; gap có thể co tại convergence | §V Limitations |
| MAJOR | "Doppler-like" loose | RFFT trên embedded feature ≠ Doppler thực | §III-C |
| MAJOR | Missing M2–M6 | Ablation path bỏ qua 5 cấu hình | §IV-B, Table 1 |
| MAJOR | No external comparison | MPJPE 165 mm không calibrate được | §V |
| MINOR | Acronym | "WiTiDAR" chưa được định nghĩa | toàn paper |
| MINOR | Figure quality | 2 figure import từ slide deck PDF | Figs 1, 2 |
| MINOR | Reference depth | 6 refs, thiếu foundational works | §II |
| MINOR | Memory unit | Cột "Mem." không có unit | Table 1 |

**Ignored Alternative Explanations**
1. **Test-time averaging hypothesis:** RF2 = áp velocity twice = ensemble effect. Có thể đạt cùng lợi ích bằng cách trung bình 2 forward pass khác.
2. **Smaller-model-converges-faster hypothesis:** Mamba2 3.6M vs PETR 13.1M — ở 20 epoch, small model lợi thế.
3. **Spectral tokenizer alone hypothesis:** M1 (spectral + PETRHead) đã giảm MPJPE 3.3 mm. Phần lớn cải tiến có thể đến từ tokenizer chứ không phải flow.

**Missing Stakeholder Perspectives**
- **Deployment engineers:** không có ARM/Jetson latency, chỉ CUDA.
- **Privacy practitioners:** "privacy-preserving" claim không phân tích — CSI vẫn có thể identify subjects (gait fingerprint từ CSI đã được chứng minh).
- **Replicators:** Hyper-parameters, hardware, seed list — đều thiếu.

**"So what?" test:** Nếu kết quả đúng và bền vững, ý nghĩa thực tiễn: WiFi 3D pose chạy 209 FPS, ~3.6M params — đủ nhỏ để consider edge deployment. Contribution thực, *nếu* không phải seed noise.

**Decision impact:** Có ≥1 CRITICAL issue ⇒ theo Iron Rule #4, Editorial Decision **không thể là Accept**.

---

## PHASE 2 — Editorial Synthesis & Decision

### Cross-Reviewer Consensus

| Issue | R0 | R1 | R2 | R3 | R4 | Consensus |
|---|---|---|---|---|---|---|
| Title overclaim | ✓ | — | — | — | ✓✓ | **Strong** (2/5 explicit, abstract self-undermines) |
| Single seed, no variance | — | ✓✓ | — | — | ✓✓ | **Strong** (CRITICAL) |
| Test-set tuning of RF step | — | ✓✓ | — | ✓ | ✓✓ | **Strong** (CRITICAL) |
| Thin reference list | ✓ | — | ✓ | ✓ | ✓ | **Strong** |
| Conceptual flow framing (non-monotonic K) | — | ✓ | — | ✓✓ | ✓✓ | **Strong** |
| Training hyper-params missing | — | ✓✓ | ✓ | ✓ | ✓ | **Strong** |
| Latency protocol underspecified | ✓ | ✓✓ | — | — | — | Medium |
| Doppler terminology loose | — | — | ✓✓ | — | ✓ | Medium (domain-specific) |
| No external baseline number | ✓ | — | ✓✓ | — | ✓ | **Strong** |
| WiTiDAR acronym undefined | ✓ | — | — | — | ✓ | Medium |
| Figures imported from slide deck | ✓ | — | — | — | ✓ | Medium |
| Missing M2–M6 | — | ✓ | — | — | ✓ | Medium |
| Cross-environment generalization | — | — | ✓ | — | ✓ | Medium |
| Memory column unit | — | ✓ | — | — | ✓ | Minor |

**Disagreements:** None substantive. R2 và R3 nhìn vấn đề từ góc domain khác nhau nhưng không xung đột; cả hai bổ sung cho nhau.

### Editorial Decision

**Decision: MAJOR REVISION**

Lý do:
1. R4 đưa ra ≥3 CRITICAL issues (single-seed, test-set tuning, title overclaim) → theo Iron Rule #4, không thể Accept.
2. R1 đánh giá rigor 4/10 và reproducibility 3/10 — dưới ngưỡng publishable cho bất kỳ venue nào kể cả workshop.
3. R0 ghi nhận story có giá trị (efficient WiFi 3D pose ~210 FPS, 3.6M params) → đủ tiềm năng cho revise, không reject.
4. Bằng chứng counter-evidence trong paper (T_FW2 retrain không cải tiến, RF non-monotonic) cho thấy authors hành xử khoa học → đáng tin để revise đúng hướng.

Workshop tier sau revision: có thể Accept. Conference tier (CVPR/MobiCom): cần revision lớn hơn nhiều, hiện không khuyến khích.

---

## Revision Roadmap (Prioritized)

### TIER 1 — MUST FIX (blocking publication)

**1. Multi-seed evaluation [R1, R4]**
- Train M0, M1, M7, M8, M9 với ≥3 seed; nếu compute cho phép, 5 seed.
- Cập nhật Table 1, Table 2 với mean ± std (hoặc median + IQR).
- Thêm paired bootstrap test giữa M0 và M9_RF2; report p-value.
- Nếu seed-noise lớn hơn 7 mm gap → thừa nhận thẳng thắn và relax claim.

**2. Validation-based selection of flow step count [R1, R3, R4]**
- Tách validation set độc lập (ví dụ 15% từ train_data, sampled per subject/scene để tránh leakage).
- Sweep RF1, RF2, RF3, RF4, RF5 trên validation; chọn step count tốt nhất.
- Report final MPJPE trên test set chỉ với step count đã chọn.
- Nếu RF2 vẫn thắng → claim đứng vững. Nếu không → cập nhật M9_RFk.

**3. Title decision [R0, R4]**
- Official paper title:
  - *"Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures"*
- Keep the manuscript framing bounded to the internal ablation setting so the title reads as a project claim about the selected draft-to-refine recipe, not as a broad SOTA claim.

**4. Implementation details subsection [R1, R2]**
Thêm vào §IV-A:
- Optimizer (AdamW?), learning rate + schedule, batch size, weight decay, gradient clip.
- Augmentation pipeline (DWT denoise, phase normalization, masking).
- Hardware: GPU model, CUDA/PyTorch version.
- Random seed list dùng.
- Training time per config.

**5. Latency measurement protocol [R0, R1]**
- Spec rõ: batch size, dtype (FP32/FP16), warm-up runs, measurement runs, sync mode, GPU model, system load condition.
- Report median ± IQR thay vì point estimate.

### TIER 2 — SHOULD FIX (strengthens paper significantly)

**6. External baseline calibration [R0, R2, R4]**
- Add column hoặc footnote: "Person-in-WiFi 3D as reported in original paper: MPJPE = X mm". Để reader đối chiếu absolute scale.
- Tối thiểu 1 đoạn discussion trong §V đặt MPJPE 165 mm vào context.

**7. Define "Matched MPJPE" và "Missed person" [R1, R2]**
- Bổ sung formal definitions trước Table 2: Hungarian matching threshold, what counts as "missed".

**8. Define WiTiDAR acronym [R0, R4]**
- Lần xuất hiện đầu tiên (Abstract hoặc §III-D): expand acronym.

**9. Reframe "Rectified Flow" terminology [R3, R4]**
- Hoặc:
  - (a) Thực hiện reflow procedure → giữ tên "rectified flow", chứng minh trajectory straight hơn sau reflow, RF1 trở thành cạnh tranh.
  - (b) Rename thành "Conditional Flow Matching Refinement Head" và thừa nhận skip reflow trong Limitations.
- Add 1 đoạn giải thích *tại sao* RF2 > RF1 > RF4 không monotonic (velocity field curvature analysis hoặc plot).

**10. Expand reference list [R0, R2, R3]**
- WiFi sensing: WiPose, Person-in-WiFi 2D (Wang et al. ICCV 2019), WiSPPN, EfficientFi, mmWave pose works để so modality.
- Phase sanitization: Tadayon CFO/SFO, Sen et al. linear transform.
- Flow / diffusion pose: DiffPose, PoseGen, DPoser nếu có.
- DETR family: deformable DETR, group DETR nếu paper kế thừa PETR.
- Mục tiêu: ≥20 entries cho workshop, ≥35 cho conference tier.

**11. Fill in M2–M6 ablation gap [R1, R4]**
- Add appendix với đầy đủ M0–M9. Hoặc giải thích trong §IV-B vì sao chỉ chọn M0, M1, M7, M8, M9.

**12. Improve "Doppler-like" framing [R2, R4]**
- Hoặc rename "motion-aware spectral profile" (giữ format hiện tại), hoặc áp RFFT trên amplitude/phase raw với visualization để justify từ "Doppler".

**13. Memory unit annotation [R1, R4]**
- Table 1 caption: "Mem. (MB, peak allocated CUDA memory at batch size N)".

### TIER 3 — NICE TO HAVE (polish)

**14. Replace slide-deck figures with vector figures [R0]**
- Vẽ system overview (Fig 1) và rectified-flow architecture (Fig 2) dạng TikZ hoặc Inkscape SVG. Slide deck PDF hiện tại không đạt chuẩn IEEE.

**15. Cross-environment generalization discussion [R2, R4]**
- 1 đoạn trong §VI Limitations bàn về cross-room / cross-subject — vấn đề kinh điển của WiFi sensing.

**16. Conditioning ablation [R3]**
- 1 row trong Table 2: M9_RF2 với $c = 0$ (unconditional). Để chứng minh WiFi-conditioning thực sự đóng góp.

**17. Privacy claim refinement [R4]**
- Hoặc remove "privacy-preserving" claim, hoặc bàn về CSI-based re-identification literature và limit claim.

**18. Multi-person association quality metrics [R2]**
- Precision/recall, ID switches, OKS — bổ sung Table 2 hoặc appendix.

---

## Author Response Checklist (for revision submission)

Khi resubmit, viết Response-to-Reviewers theo cấu trúc R→A→C (Reviewer comment → Author response → Change location):

```
[R1.1] Reviewer 1, comment 1: "Single seed reporting..."
  Response: We re-trained M0, M1, M7, M8, M9 with 3 seeds each.
            Updated Table 1 with mean±std. Paired bootstrap test gives p=...
  Change:   §V-A, Table 1 rows updated; Appendix B added with raw per-seed numbers.
```

Mỗi CRITICAL/MAJOR issue trong Issue List của R4 phải có entry tương ứng.

---

## Final Editorial Summary

**Recommendation:** **MAJOR REVISION** with potential for workshop acceptance after revision.

**Decision rationale:**
- Paper trình bày trung thực và có observation có giá trị (2-step Euler refinement).
- Nhưng *3 CRITICAL* issues (single-seed, test-set tuning, title overclaim) ngăn không cho Accept.
- Sau khi sửa Tier 1 + ít nhất 4 mục Tier 2, paper sẽ ở mức publishable tại workshop tier.

**Estimated revision effort:** ~4–6 tuần (chủ yếu là multi-seed re-training + validation split + writing 8–10 đoạn mới).

**Strongest takeaway for authors:** Câu chuyện thực sự của paper là *"compact Mamba2 + 2-step learned refinement đạt 210 FPS với MPJPE cạnh tranh PETR baseline"*. Đây là claim khả thi, có giá trị deployment, và đủ scope cho student venue. Không cần phóng đại bằng title "Flow is All You Need".

---

*Review compiled following the academic-paper-reviewer v1.10.0 workflow (5-reviewer panel + editorial synthesis). All criticisms grounded in specific passages of `main.tex`; line numbers and section identifiers cited throughout. Iron Rules followed: read-only constraint respected (manuscript not modified), no fabricated comments, every consensus point traceable to ≥1 reviewer report.*
