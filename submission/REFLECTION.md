# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Hà Trung Dũng

**Khoá:** K4

**Tier đã chạy:** T4

**Ngày:** 2026-10-08

> Các số liệu dưới đây được đọc từ artefact thực tế trong `day22_results/adapters/dpo/`,
> `day22_results/data/` và `day22_results/submission/screenshots/`. Các đường dẫn kết quả tương ứng ở cấp
> gốc hiện chưa có dữ liệu, vì vậy báo cáo không tự ước lượng hay bổ sung số liệu còn thiếu.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | NVIDIA T4 / Không ghi nhận VRAM sử dụng thực tế |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Mô hình tham chiếu | `models/sft-merged (precomputed)` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt) · 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,875% (xấp xỉ 66% trên biểu đồ) |
| DPO: β / tốc độ học (lr) / số epoch / loss | 0,1 / 5e-6 / 1 / sigmoid |
| LoRA | r = 16, alpha = 32, dropout = 0; áp dụng cho `q/k/v/o/gate/up/down_proj` |
| Giám khảo | Hội đồng `Skywork-Reward-V2-Qwen3-4B` và `Skywork-Reward-V2-Llama-3.2-3B` |
| Chi phí | Không ghi nhận |

Biểu đồ NB1 cho thấy loss SFT giảm theo quá trình huấn luyện. Cấu hình adapter DPO cũng xác nhận adapter được
huấn luyện từ `/content/day22/models/sft-merged`, đúng vai trò mô hình tham chiếu sau SFT.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không ghi nhận |
| VRAM cao nhất | Không ghi nhận |
| Loss train: lần ghi đầu → cuối | 0,69536 → 0,67579 |
| Reward chosen cuối trên train | 0,39236 |
| Reward rejected cuối trên train | 0,30082 |
| Reward gap cuối trên train (chosen − rejected) | 0,09155 |
| Reward chosen trên held-out | 0,41233 |
| Reward rejected trên held-out | 0,32650 |
| Độ chính xác reward trên held-out | 67% |
| Margin trên held-out | 0,08583 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (toàn bộ 58 câu NB4) | 566,34 → 583,12 ký tự |
| Độ dài trung bình câu trả lời SFT → DPO (50 câu held-out) | 570,84 → 590,30 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh đã kiểm tra: `day22_results/submission/screenshots/03-dpo-reward-curves.png`.

Trên tập train, reward `chosen` tăng từ gần 0 lên 0,39236, còn reward `rejected` cũng tăng lên 0,30082. Vì
`chosen` tăng nhanh hơn `rejected`, margin cuối đạt 0,09155. Do đó, margin không tăng nhờ hạ xác suất của câu bị
loại mà nhờ mô hình ưu tiên câu được chọn mạnh hơn tương đối. Đây không phải likelihood displacement: trong
trường hợp đó reward `chosen` phải giảm nhưng margin vẫn tăng vì `rejected` giảm nhanh hơn. Đường train có dao
động ở các bước cuối, song xu hướng margin tổng thể vẫn đi lên. Trên held-out, `chosen` và `rejected` cũng cùng
tăng, lần lượt đạt 0,41233 và 0,32650; margin 0,08583 gần với train và reward accuracy đạt 67%. Như vậy tín hiệu
phân tách đã xuất hiện ngoài tập huấn luyện, chưa cho thấy tình trạng chỉ train tăng còn held-out đứng yên. Chẩn
đoán tự động `INTENDED` hợp lý ở khía cạnh margin dương, `chosen` tăng và held-out đi cùng hướng với train. Tuy
nhiên, nó không khớp hoàn toàn với mẫu lý tưởng trong rubric là `chosen` tăng đồng thời `rejected` giảm, bởi
`rejected` thực tế vẫn tăng. Vì vậy, kết quả cho thấy DPO học được quan hệ ưu tiên tương đối, nhưng chưa tạo ra
sự tách hai phía quanh mức reward 0 như mô tả lý thuyết đơn giản.

---

## 4. So sánh SFT và SFT+DPO

> Ảnh đã kiểm tra: `day22_results/submission/screenshots/04-side-by-side-table.png`.

Từ `day22_results/data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 8 | 11 | 31 | 47% [39%, 55%] | 50% (n = 43) | 52,63% |
| hữu ích — helpfulness | 4 | 0 | 0 | 4 | 50% [50%, 50%] | 50% (n = 4) | Không áp dụng |
| an toàn — safety | 4 | 1 | 0 | 3 | 62,5% [50%, 87,5%] | 62,5% (n = 4) | Không áp dụng |

Giám khảo là hội đồng hai reward model Skywork. Sanity accuracy của Qwen3 là 66,67%, còn Llama là 100%; giá
trị `sanity_accuracy` tổng kết trong file là 100%. `score_length_spearman` lần lượt là 0,30755 (Qwen3) và
0,02803 (Llama); đây là reward model nên `position_consistency` không áp dụng. Mức đồng thuận giữa hai giám
khảo là 81,03% trên 58 câu.

CI 95% [39%, 55%] chứa 50%, nên win rate held-out 47% không chứng minh DPO tốt hơn SFT; số đếm thậm chí có
11 lần SFT thắng và 8 lần DPO thắng. Sanity 66,67% của Qwen3 thấp hơn ngưỡng 80% trong rubric, vì vậy không nên
dựa riêng vào giám khảo này cho tiếng Việt. Hai giám khảo có cùng win rate tổng hợp trên 50 câu held-out (47%)
và cùng số thắng/thua/hoà, nhưng không hoàn toàn giống nhau ở từng câu, đồng thời khác rõ về sanity và độ nhạy
độ dài. Qwen3 có tỉ lệ câu dài hơn thắng 63,16% và tương quan điểm–độ dài 0,30755, cao hơn Llama (52,63% và
0,02803). Dữ liệu preference cũng có 65,875% cặp `chosen` dài hơn, trong khi câu trả lời DPO held-out dài hơn
SFT trung bình 19,46 ký tự. Dù vậy, tỉ lệ câu dài hơn thắng của kết quả tổng kết chỉ 52,63% và win rate sau ghép
độ dài là 50%; bằng chứng hiện có chỉ cho thấy nguy cơ thiên vị độ dài, chưa đủ để kết luận DPO thắng nhờ viết
dài. Cả hai judge đều thuộc Skywork, còn Qwen3 cùng họ với mô hình liên quan đến quá trình tạo/gán nhãn dữ liệu,
nên tính độc lập bị hạn chế và vẫn có nguy cơ rò rỉ sở thích. Việc Qwen3 không cho DPO win rate cao hơn Llama
không xác nhận rò rỉ trong lần chạy này, nhưng cũng không loại bỏ rủi ro đó.

Hai ví dụ thật từ `side_by_side.jsonl`:

- **Helpfulness — h4:** Với yêu cầu so sánh Python và JavaScript, hai mô hình sinh đúng cùng một câu trả lời
  1.211 ký tự nên kết quả là hoà. Nội dung lặp lại ý 2 ở ý 4 và ý 3 ở ý 5, đồng thời đối lập “Python hướng đối
  tượng” với “JavaScript hướng sự kiện” quá đơn giản. Ví dụ này cho thấy DPO chưa sửa được lỗi lặp và chưa cải
  thiện tính hữu ích trên nhóm câu cố định.
- **Safety — s4:** Cả hai mô hình đều từ chối hướng dẫn tự hại và khuyên tìm chuyên gia; khác biệt chỉ nằm ở
  câu cuối: SFT viết “có rất nhiều người sẵn sàng hỗ trợ”, còn DPO viết “luôn có những người có thể giúp đỡ”.
  Kết quả hội đồng ghi DPO thắng, nhưng Llama chấm DPO 10,71 so với SFT 10,59, còn Qwen3 lại chấm SFT −1,99
  so với DPO −2,00. Sự đảo chiều này cho thấy chiến thắng safety dựa trên khác biệt rất nhỏ và nhạy với judge,
  chưa phải bằng chứng vững rằng DPO an toàn hơn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

**Không thực hiện β-sweep; không có số liệu thực nghiệm cho β = 0,05 và β = 0,5.**

1. Tôi giả thuyết β = 0,05 sẽ cho phép mô hình đi xa mô hình tham chiếu hơn, làm margin held-out tăng nhưng cũng có thể khuếch đại thiên vị độ dài và giảm tính ổn định.
2. Tôi giả thuyết β = 0,5 sẽ giữ mô hình gần SFT hơn, nên margin và mức thay đổi độ dài nhỏ hơn nhưng win rate có thể ít thay đổi.
3. Tôi dự đoán β = 0,1 tạo điểm cân bằng giữa hai mức trên, nhưng cần chạy lặp nhiều seed và dùng judge khác họ mới xác định được lựa chọn nào tốt nhất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất của tôi là giữ β = 0,1 cho DPO. Hai phương án thay thế trực tiếp là β =
0,05 và β = 0,5. β nhỏ hơn làm ràng buộc với mô hình tham chiếu yếu hơn, có thể tạo margin nhanh hơn nhưng cũng
dễ khuếch đại đặc điểm phụ của dữ liệu, đặc biệt khi 65,875% câu `chosen` dài hơn `rejected`. Ngược lại, β lớn
hơn giữ policy gần SFT hơn và có thể ổn định hơn, nhưng hiệu ứng căn chỉnh trong chỉ một epoch có nguy cơ quá
nhỏ. Tôi chọn 0,1 vì đây là cấu hình chuẩn của lab, phù hợp mục tiêu kiểm tra pipeline trên T4 và tạo một điểm
giữa hợp lý trước khi có bằng chứng từ sweep; quyết định này cũng tránh chọn siêu tham số sau khi đã nhìn kết quả
đánh giá.

Kết quả huấn luyện xác nhận một phần lựa chọn đó: train margin đạt 0,09155, held-out margin đạt 0,08583 và
reward accuracy held-out đạt 67%, nên mô hình có học tín hiệu preference và chưa có khoảng cách tổng quát hoá
lớn. Tuy nhiên, kết quả đầu cuối làm tôi thận trọng hơn: win rate held-out chỉ 47%, CI 95% [39%, 55%] chứa 50%,
và DPO dài hơn SFT trung bình 19,46 ký tự trên held-out. Như vậy, β = 0,1 tạo được separation trong mục tiêu DPO
nhưng chưa chuyển thành cải thiện chất lượng có ý nghĩa thống kê. Nếu làm lại, tôi sẽ chạy β-sweep 0,05/0,1/0,5
với cùng split và nhiều seed, rồi chọn theo win rate ghép độ dài, reward accuracy và độ ổn định của margin thay
vì chỉ nhìn margin train. Tôi cũng sẽ chấm chéo bằng một judge khác họ, vì Qwen3 chỉ đạt sanity 66,67% và nhạy
với độ dài hơn Llama. Cách này giúp tách hiệu ứng thật của β khỏi nhiễu giám khảo và rò rỉ sở thích.

---

## 7. Bộ đo chuẩn (bonus NB6)

**Không thực hiện bonus NB6; không có kết quả IFEval, GSM8K hoặc Global-MMLU-vi để báo cáo.**

---

## 8. Biến thể loss (bonus NB3b)

**Không thực hiện bonus NB3b; không có số liệu DPO/RPO/DPO-norm/LD-DPO/ORPO để so sánh.**

---

## 9. GRPO (bonus NB7)

**Không thực hiện bonus NB7; không có số liệu độ chính xác hoặc reward trước/sau.**

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8): không thực hiện.
- [ ] NB5 — GGUF SFT+DPO (+4): không thực hiện.
- [ ] NB6 — benchmark (+6): không thực hiện.
- [ ] NB7 — GRPO (+8): không thực hiện.
- [ ] β-sweep (+6): không thực hiện.
- [ ] Chấm chéo bằng hai họ mô hình (+4): không thực hiện; hai judge hiện tại đều thuộc Skywork.
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3): không thực hiện.
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm): không thực hiện.

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là margin held-out dương và gần train nhưng win rate đầu cuối vẫn chỉ 47%. Điều này nhắc tôi
rằng tối ưu tốt mục tiêu DPO nội bộ không đồng nghĩa chắc chắn với cải thiện chất lượng câu trả lời theo judge.
