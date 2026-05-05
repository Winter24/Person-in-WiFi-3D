# Plan Presentation - ResFes 2026

## Mục tiêu

Deck trình bày hiện đã được chốt về nội dung. Sau hai chỉnh sửa cuối cùng:

- đổi toàn bộ `MO` thành `M0`
- xóa slide trùng của `Why flow helps beyond direct regression`

nhóm **không tiếp tục sửa nội dung slide** nữa. Việc còn lại là đồng bộ toàn bộ tài liệu planning, slide blueprint, và speaker script với đúng deck PDF hiện tại.

Thông điệp trung tâm của bài nói:

> Flow là yếu tố kéo accuracy lên, còn WiMamba là yếu tố kéo mô hình tới trạng thái triển khai thực tế.

Điều hội đồng cần nhớ sau phần trình bày:

1. Bài toán xuất phát từ nhu cầu sensing trong không gian nhạy cảm, nơi camera không luôn phù hợp.
2. `M0-M4` là một ladder có kiểm soát, không phải tập hợp các biến thể rời rạc.
3. `M3` là accuracy winner, còn `M4` là deployment winner trong local benchmark hiện tại.

---

## Narrative chính thức của deck

Narrative hiện tại bám đúng slide `Content` và không quay lại cấu trúc 13-slide cũ:

```text
01 Problem & Motivation
02 Proposed System
03 Baseline Method
04 What We Propose: A Staged Redesign
05 Experiments & Results
06 Appendix
```

Deck kể câu chuyện theo trình tự:

```text
privacy motivation
-> WiFi sensing setup
-> baseline M0
-> staged redesign M0 -> M4
-> quantitative + qualitative evidence
-> concise summary
```

Nguyên tắc khóa nội dung:

- Giữ nguyên toàn bộ thông điệp chính của deck hiện tại.
- Không thêm nhánh mới ngoài `M0-M4`.
- Không đưa `BoneLengthLoss` vào narrative chính.
- Không đổi lại cấu trúc slide chỉ để “đẹp logic” hơn trên giấy.
- Tài liệu nội bộ phải phản ánh **đúng deck đang dùng để nói**, không phản ánh một phương án đề xuất cũ.

---

## Cấu trúc deck chính thức

Sau khi bỏ slide trùng, deck hiện tại gồm:

- `26` slide main presentation, tính cả `Title`, `Content`, các section divider, và `Thanks`
- `5` slide appendix
- tổng cộng `31` slide

Mapping tổng quát:

| Phần | Slide | Vai trò |
|---|---:|---|
| Opening | 1-2 | Title + agenda |
| Problem & Motivation | 3-6 | Section divider + 3 slide framing |
| Proposed System | 7-8 | Divider + hệ thống tổng quan |
| Baseline Method | 9-11 | Divider + `M0` architecture/loss |
| Staged Redesign | 12-18 | Divider + ladder + từng stage + full M4 pipeline |
| Experiments & Results | 19-25 | Divider + dataset + metric + quantitative + qualitative + flow rationale |
| Closing | 26 | Thanks |
| Appendix | 27-31 | Backup cho Q&A |

---

## Phase 1 - Khóa narrative và claim

**Trạng thái:** Hoàn tất  
**Mục tiêu:** Đảm bảo mọi tài liệu nội bộ kể cùng một câu chuyện với deck cuối.

Điểm khóa:

- mở bằng privacy-preserving indoor sensing
- giải thích baseline `M0` rõ ràng trước khi nói contribution
- dùng ladder `M0 -> M1 -> M2 -> M3 -> M4`
- kết luận theo trục:
  - `M3` tốt nhất về MPJPE
  - `M4` tốt nhất về trade-off triển khai

Claim boundary cần giữ nguyên:

- không nói `M4` thắng mọi metric
- không gọi benchmark hiện tại là state-of-the-art
- qualitative chỉ là selected challenging cases
- appendix mới là nơi trả lời sâu về training pipeline và anatomy kỹ thuật

Output của phase này:

- narrative lock
- claim lock
- section order lock

---

## Phase 2 - Khóa evidence và số liệu

**Trạng thái:** Hoàn tất  
**Mục tiêu:** Đồng bộ cách diễn giải số liệu với đúng slide kết quả.

Các mốc số liệu đang được deck sử dụng:

| Model | MPJPE | FPS | Params | Peak Memory |
|---|---:|---:|---:|---:|
| M0 | 169.34 | 45.69 | 13.13M | 155.60 MB |
| M1 | 164.60 | 49.36 | 13.20M | 155.86 MB |
| M2 | 169.41 | 51.48 | 11.97M | 151.16 MB |
| M3 | 151.99 | 138.79 | 7.06M | 38.49 MB |
| M4 | 159.00 | 159.85 | 5.83M | 27.02 MB |

Diễn giải thống nhất cần giữ:

- `M3` là accuracy winner
- `M4` là deployment winner
- `M4` cải thiện rõ so với `M0` về tốc độ, tham số, và bộ nhớ
- Flow là thành phần kéo accuracy lên
- WiMamba là thành phần kéo deployability lên

Output của phase này:

- bảng số liệu chốt
- wording chốt cho slide quantitative
- wording chốt cho slide summary

---

## Phase 3 - Khóa visual và asset

**Trạng thái:** Hoàn tất ở mức deck hiện tại  
**Mục tiêu:** Đảm bảo blueprint tài liệu khớp với đúng visual đang xuất hiện trong PDF.

Các visual chính hiện đang được deck dùng:

- slide problem/motivation theo hướng privacy framing
- slide `Proposed System` với sơ đồ end-to-end
- slide `M0 baseline` và `What M0 optimizes`
- cụm slide stage redesign:
  - ablation ladder
  - Spectral Tokenization
  - WiMamba
  - draft-to-refine
  - proposed `M4` pipeline
  - `M4 Training Pipeline`
- cụm results:
  - dataset overview
  - MPJPE metric
  - quantitative comparison
  - qualitative comparison
  - `Why flow helps beyond direct regression`
- appendix:
  - layered pipeline / baseline recap
  - full architecture + training + loss
  - backbone and pose head
  - detailed qualitative figure

Điểm cần đồng bộ trong tài liệu:

- dùng đúng tên slide hiện tại
- phản ánh việc slide duplicate đã bị xóa
- phản ánh việc tất cả baseline đã đổi sang `M0`

---

## Phase 4 - Đồng bộ blueprint slide

**Trạng thái:** Cần đồng bộ với deck cuối  
**File chính:** [phase4_full_slide_detail.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase4_full_slide_detail.md)

Mục tiêu của phase này không còn là “thiết kế deck mới”, mà là:

- mô tả đúng **từng slide hiện có**
- giữ đúng thứ tự và tên section
- chỉ ra vai trò của mỗi slide trong câu chuyện chung
- tách rõ main deck và appendix

Output cần có:

- slide-by-slide blueprint cho `31` slide hiện tại
- note rõ slide `Why flow helps...` chỉ còn **một** bản
- note rõ baseline dùng `M0`

---

## Phase 5 - Đồng bộ speaker script

**Trạng thái:** Cần cập nhật theo deck cuối  
**File chính:** [phase5_speaker_script.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase5_speaker_script.md)

Speaker script cần bám đúng deck hiện tại:

- script cho `Title`, `Content`, và các section divider phải ngắn
- script chính tập trung vào các slide content thật sự
- phần kỹ thuật phải chia nhịp rõ:
  - baseline
  - staged redesign
  - results
  - summary

Điểm speaker bắt buộc nói đúng:

- `M0` là baseline nghiêm túc
- `M3` là model tốt nhất về accuracy
- `M4` là model tốt nhất về trade-off triển khai
- slide `Why flow helps beyond direct regression` là slide chốt ý tưởng, không phải slide appendix

Output cần có:

- timing plan cho main deck
- speaker split cho 2 người và 3 người
- handoff lines
- script ngắn cho từng slide chính
- Q&A anchors gắn với appendix

---

## Phase 6 - Rehearsal và sử dụng appendix

**Trạng thái:** Tiếp tục dùng cho rehearsal  
**Mục tiêu:** Nói trơn tru trên deck hiện tại mà không phải đổi thêm slide.

Checklist rehearsal:

- các section divider có bị dừng quá lâu không
- slide `Proposed System` và `M4 Training Pipeline` có bị nói quá chi tiết không
- slide quantitative có nói rõ `M3` vs `M4` không
- slide qualitative có tránh overclaim không
- slide `In summarize` có chốt đúng ba ý không

Checklist appendix:

- biết khi nào mở slide 28-31
- không tự động đi vào appendix nếu chưa có câu hỏi
- dùng appendix để trả lời:
  - architecture sâu hơn
  - training/loss sâu hơn
  - backbone/pose head
  - qualitative chi tiết

---

## Mapping tài liệu cần đồng bộ

Sau lần cập nhật này, ba file sau phải phản ánh cùng một version của deck:

- [plan_presentation.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/plan_presentation.md)
- [phase4_full_slide_detail.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase4_full_slide_detail.md)
- [phase5_speaker_script.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/phase5_speaker_script.md)

Nguyên tắc cuối cùng:

> Từ giờ tài liệu nội bộ phải mô tả đúng deck đang nói, không mô tả một deck “lý tưởng” khác.
