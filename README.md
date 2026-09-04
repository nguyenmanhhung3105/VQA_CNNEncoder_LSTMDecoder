# VQA – Visual Question Answering (CNN Encoder + LSTM Decoder)

Hệ thống Visual Question Answering trên bộ dữ liệu VQA v2, sinh câu trả lời dạng chuỗi từ (free-form, tối đa 10 token) từ một cặp ảnh + câu hỏi. Mô hình kết hợp một CNN visual encoder (huấn luyện từ đầu hoặc ResNet-50 pretrained) với một bidirectional LSTM question encoder, sau đó fusion và giải mã câu trả lời bằng một LSTM decoder tự hồi quy (autoregressive), có hoặc không có cơ chế spatial attention theo câu hỏi.

Notebook huấn luyện và đánh giá đồng thời **4 biến thể mô hình** để so sánh tác động của backbone (scratch vs. pretrained) và attention (có/không).

## Kiến trúc mô hình

```
Image ──► CNN Encoder ────────────────────────────────┐
                                                        ▼
Question ─► Bidirectional LSTM Encoder → (h, c) ─► Fusion (concat + Linear + Tanh)
                                                        ▼
                                                  LSTM Decoder (2 layers, autoregressive)
                                                        ▼
                                                  Answer tokens (greedy decode)
```

**4 biến thể (`VQAModel(use_pretrained, use_attention)`):**

| Model | CNN backbone | Attention | Số tham số trainable |
|---|---|---|---|
| Model 1 | CNN từ đầu (4 conv block + AdaptiveAvgPool) | Không | 13,026,084 |
| Model 2 | ResNet-50 pretrained ImageNet (đóng băng phần lớn layer, giữ 20 tham số cuối trainable) | Không | 22,483,492 |
| Model 3 | CNN từ đầu | Có (dạng element-wise gating giữa image feature và question hidden state — **không phải attention trên feature map không gian**) | 13,026,084 |
| Model 4 | ResNet-50 pretrained | Có (soft spatial attention thật sự, tính trên feature map 7×7×2048 = 49 vùng không gian, điều kiện theo câu hỏi) | 24,188,196 |

**Các thành phần chính:**
- `CNN_Scratch`: 4 lớp Conv2d + BatchNorm + ReLU (stride 2), AdaptiveAvgPool2d, Linear projection → `HIDDEN_DIM` (512)
- `CNN_Pretrained`: ResNet-50 (`torchvision.models.resnet50`, weights `IMAGENET1K_V1`), bỏ avgpool/fc gốc, có thể trả về feature map không gian (49×2048) khi dùng attention
- `QuestionEncoder`: Embedding + Bidirectional LSTM 2 layer, dropout 0.3, hidden_dim=512
- `SpatialAttention`: additive/Bahdanau-style attention (`tanh(W_img·img + W_ques·ques)` → softmax) trên 49 vùng ảnh, chỉ dùng ở Model 4
- `LSTMDecoder`: Embedding + LSTM 2 layer + Linear, huấn luyện bằng teacher forcing, sinh câu trả lời bằng greedy decode khi inference

## Dataset

- **Nguồn**: VQA v2 (câu hỏi/annotation chính thức từ `s3.amazonaws.com/cvmlp/vqa/mscoco/vqa`) trên ảnh COCO 2014 (`train2014`, `val2014`)
- **Vocabulary**: câu hỏi giới hạn top 10,000 từ phổ biến nhất; câu trả lời giới hạn top 3,000 answer phổ biến nhất (tokenized thành 2,420 token riêng)
- **Chia tập**:
  - Train: 221,878 mẫu (= 50% của `train2014`, được lấy mẫu ngẫu nhiên có `seed=42` để giảm thời gian huấn luyện trên Colab)
  - Val: 42,870 mẫu (= 25% của 80% `val2014`)
  - Test: 42,871 mẫu (= 20% còn lại của `val2014`)
- **Tiền xử lý ảnh**: resize 224×224, chuẩn hóa theo mean/std ImageNet
- **Câu hỏi**: lowercase, loại ký tự không phải a-z0-9, cắt tối đa 20 token
- **Câu trả lời**: lấy answer xuất hiện nhiều nhất trong 10 câu trả lời của human annotator, bọc `<sos>`/`<eos>`, cắt tối đa 10 token

[CẦN BỔ SUNG] Số phiên bản chính xác của các file zip VQA v2 tải về (URL trong notebook không có version tag), và license của bộ dữ liệu VQA v2/COCO chưa được ghi trong notebook.

## Cài đặt

Notebook chạy trên **Google Colab**, không có file `requirements.txt` hay `setup.py` trong repo. Dependency được cài trực tiếp trong notebook:

```bash
pip install pycocoevalcap -q
pip install timm -q
```

Các thư viện khác (giả định đã có sẵn trên môi trường Colab, không được pin version trong notebook):

```
torch, torchvision
nltk
numpy
pandas
matplotlib
tqdm
Pillow
```

[CẦN BỔ SUNG] Version cụ thể của từng thư viện (PyTorch, torchvision, timm...) không được ghi lại trong notebook — nên pin version nếu muốn tái lập kết quả chính xác.

Notebook cũng dùng `nltk.download("wordnet")` và `nltk.download("omw-1.4")` cho việc tính điểm METEOR.

## Cách sử dụng

Notebook được thiết kế để chạy trên nhiều phiên Colab khác nhau (do giới hạn thời gian session ~4.5 tiếng/phiên), điều phối qua biến `CURRENT_MAIL`:

```python
CURRENT_MAIL = "mail1"   # build vocab + lưu vocab_checkpoint.pt
CURRENT_MAIL = "mail2"   # train Model 1 (CNN scratch, no attention)
CURRENT_MAIL = "mail3"   # train Model 2 (ResNet50 pretrained, no attention)
CURRENT_MAIL = "mail4"   # train Model 3 (CNN scratch, attention/gating)
CURRENT_MAIL = "mail5"   # train Model 4 (ResNet50 pretrained, attention)
```

Dữ liệu và checkpoint được lưu/đọc từ Google Drive (`/content/drive/MyDrive/vqa`), yêu cầu mount Drive (`google.colab.drive.mount`).

**Training** (sau khi đã build vocab, chạy tuần tự trong notebook — không có CLI script riêng):

```python
model1 = train_model(model1, "Model1 CNN Scratch No Attention", "model1", lr=1e-3)
model2 = train_model(model2, "Model2 CNN Pretrain No Attention", "model2", lr=5e-4)
model3 = train_model(model3, "Model3 CNN Scratch Spatial Attn", "model3", lr=1e-3)
model4 = train_model(model4, "Model4 CNN Pretrain Spatial Attn", "model4", lr=5e-4)
```

- Optimizer: AdamW, weight_decay=1e-4; scheduler: CosineAnnealingLR
- 2 epoch/model (`EPOCHS = 2`), batch size 128 (train) / 256 (eval)
- Mixed-precision training (`torch.amp`), gradient clipping (max_norm=5.0)
- Tự động lưu checkpoint giữa chừng (`*_mid.pt`) và dừng nếu phiên Colab sắp hết giờ (giới hạn 4.6 giờ), để resume ở phiên sau

**Inference / sinh câu trả lời** (greedy decode):

```python
model.eval()
answers = model.generate(images, questions, lengths)  # list[str]
```

## Kết quả

Đánh giá trên checkpoint tốt nhất (val_loss thấp nhất) của mỗi model, sau 2 epoch huấn luyện:

**Validation set**

| Model | VQA Accuracy | BLEU-1 | METEOR | CIDEr |
|---|---|---|---|---|
| Model 1 – Scratch, No Attention | 31.27% | 0.3821 | 0.1868 | 0.6008 |
| Model 2 – Pretrained, No Attention | 33.56% | 0.4067 | 0.2008 | 0.6466 |
| Model 3 – Scratch, Attention (gating) | 34.96% | 0.4242 | 0.2113 | 0.6756 |
| Model 4 – Pretrained, Spatial Attention | **41.44%** | **0.4943** | **0.2502** | **0.8127** |

**Test set**

| Model | VQA Accuracy | BLEU-1 | METEOR | CIDEr |
|---|---|---|---|---|
| Model 1 – Scratch, No Attention | 30.69% | 0.3757 | 0.1836 | 0.5894 |
| Model 2 – Pretrained, No Attention | 32.74% | 0.3986 | 0.1965 | 0.6302 |
| Model 3 – Scratch, Attention (gating) | 34.34% | 0.4171 | 0.2074 | 0.6666 |
| Model 4 – Pretrained, Spatial Attention | **40.80%** | **0.4896** | **0.2470** | **0.8003** |

Model 4 (pretrained ResNet-50 + spatial attention thật sự) cải thiện VQA Accuracy từ 31.27% (Model 1 baseline) lên 41.44% trên validation — tăng 10.17 điểm — đồng thời cải thiện đồng đều trên cả BLEU-1, METEOR và CIDEr.

VQA Accuracy được tính theo công thức chính thức của VQA v2: `min(#human_answers_matching / 3, 1)`.

## Cấu trúc thư mục

Toàn bộ pipeline nằm trong một notebook Colab duy nhất, dữ liệu và checkpoint lưu trên Google Drive (`/content/drive/MyDrive/vqa`):

```
vqa/
├── VQA_CNNEncoder_LSTMDecoder_final.ipynb
├── VQA_CNNEncoder_LSTMDecoder_final_optimized_(1).ipynb
├── train2014/                          # 82,782 ảnh COCO train2014
├── val2014/                            # 40,504 ảnh COCO val2014
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── v2_mscoco_train2014_annotations.json
├── v2_mscoco_val2014_annotations.json
├── checkpoints/                        # vocab_checkpoint.pt, model{1..4}_epoch{01,02}.pt
├── checkpoint/                         # checkpoint phụ/backup (23 file)
├── chart_grouped_bar.png
├── chart_heatmap.png
├── chart_radar.png
├── chart_tong_hop.png
├── vqa_attention_maps.png
└── vqa_visual_examples.png
```

