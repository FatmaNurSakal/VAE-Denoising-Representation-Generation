# VAE Denoising + Representation Learning + Generation (Fashion-MNIST) — Kaggle Notebook
Variational Auto-Encoder (VAE) ile Denoising ve Representation Generation.

Bu proje, **Fashion-MNIST** üzerinde **Variational Auto-Encoder (VAE)** kullanarak:

1) **Denoising** (gürültülü görüntü → temiz görüntü)  
2) **Representation Learning** (latent uzayda sınıf ayrışması analizi)  
3) **Generation** (prior’dan örnekleyerek yeni görüntü üretme)  
4) **Latent Interpolation** (latent uzayda süreklilik gösterimi)

işlerini **tek bir Kaggle hücresinde** çalıştırılabilir şekilde gerçekleştirir.

Notebook; eğitim, metrik hesaplama ve tüm görsel çıktıları otomatik üretir.

---

## 1) Problem Tanımı

Gerçek dünyada görüntüler sensör gürültüsü, düşük ışık, sıkıştırma veya iletim hatalarıyla bozulabilir. Bu projede amaç:

- **Gürültülü** görüntülerden **temiz** görüntüleri yeniden üretmek (denoising)
- VAE’nin latent uzayında **anlamlı temsil** öğrenip öğrenmediğini göstermek (PCA/UMAP/t-SNE)
- VAE’nin “örneklenebilir” latent uzayı sayesinde **yeni örnekler üretebilmek** (generation)
- İki örnek arasında latent uzayda yürüyerek **sürekliliği** göstermek (interpolation)

---

## 2) Veri Seti ve Ön İşleme

### Dataset: Fashion-MNIST
- 28×28 çözünürlükte, tek kanallı (grayscale)
- 10 sınıf (kıyafet kategorileri)
- Eğitim: 60.000, Test: 10.000

### Veri yükleme
Notebook, Kaggle input dizininden IDX formatını okur:
- `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`

### Ön işleme adımları
- `float32`’a çevirme
- `[0,1]` normalize
- Kanal ekleme: `(28, 28, 1)`

---

## 3) Denoising (Gürültü Ekleme)

Notebook, giriş görüntülerine Gaussian gürültü ekler:

- `NOISE_FACTOR = 0.45`
- `x_noisy = clip(x + noise_factor * N(0,1), 0, 1)`

Modelin eğitimi şu hedefle yapılır:

- **Girdi:** `x_train_noisy`
- **Hedef:** `x_train` (clean)

---

## 4) Model Mimarisi ve Yaklaşım

### VAE bileşenleri
- **Encoder:** `x → z_mean, z_log_var, z`
- **Sampling layer:** reparameterization trick
- **Decoder:** `z → x_recon`

### Kayıp fonksiyonu
Toplam kayıp şu terimlerin birleşimidir:

1) **Reconstruction loss (BCE):** `x_clean` ile `x_recon` benzerliği  
2) **KL divergence:** latent dağılımı prior’a yaklaştırma (örneklenebilir latent)  
3) **Supervised Contrastive Loss (SupCon):** `z_mean` üzerinde sınıf ayrışmasını teşvik etme

Form:
`Total = Recon + beta * KL + gamma_sc * SupCon`

### Beta Warm-up
KL ağırlığı eğitim boyunca kademeli artırılır:
- `BETA_MAX = 1.0`
- `WARMUP_EPOCHS = 8`

Bu strateji, modelin önce rekonstrüksiyonu öğrenmesini, sonra latent düzenlemeyi sağlamasını hedefler.

---

## 5) Deney Ayarları (Hiperparametreler)

Notebook içindeki temel değerler:
- `LATENT_DIM = 16`
- `BATCH_SIZE = 256`
- `EPOCHS = 20`
- `NOISE_FACTOR = 0.45`
- `GAMMA_SC = 1.0` (SupCon ağırlığı)

---

## 6) Kaggle’da Çalıştırma Talimatı

### 1) Dataset’i ekle
Kaggle Notebook → **Add data**:
- Dataset yolu notebookta şu şekilde bekleniyor:
  - `DATA_DIR = "/kaggle/input/fashionmnist"`

Dataset içinde şu dosyalar olmalı:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

### 2) Notebook hücresini çalıştır
Notebook tek hücre çalıştırılabilir yapıdadır:
- UMAP yüklü değilse otomatik `pip install umap-learn` dener.
- Eğitim + metrikler + tüm görseller otomatik üretilir.

### 3) Çıktı klasörü
Tüm çıktılar şu dizine kaydedilir:
- `OUT_DIR = "/kaggle/working/vae_outputs_v3"`

Notebook sonunda `vae_outputs_v3` içeriği listelenir.

---

## 7) Üretilen Çıktılar (Dosyalar)

Notebook, aşağıdaki dosyaları üretir:

### Eğitim eğrileri
- `01_train_total_loss.png`
<img width="1236" height="785" alt="01_train_total_loss (1)" src="https://github.com/user-attachments/assets/7d662713-236c-43a9-ba6e-fb6fa7cb8cd2" />

- `02_train_recon_loss.png`
<img width="1236" height="785" alt="02_train_recon_loss (1)" src="https://github.com/user-attachments/assets/e701dddb-2206-436e-b265-979c19c63370" />

- `03_train_kl_loss.png`
<img width="1219" height="785" alt="03_train_kl_loss (1)" src="https://github.com/user-attachments/assets/0ecaaf16-02ce-4320-90b3-82db3bf69919" />

- `03B_train_supcon_loss.png`
<img width="1245" height="785" alt="03B_train_supcon_loss" src="https://github.com/user-attachments/assets/f0f62a47-5114-4469-b58b-9197113c7f50" />

- `04_beta_warmup.png`
<img width="1227" height="708" alt="04_beta_warmup (1)" src="https://github.com/user-attachments/assets/60be0ad6-1a8b-4b87-9ad0-1984c2b75abc" />

### Denoising görseli
- `05_denoising_grid.png`
<img width="3469" height="1030" alt="05_denoising_grid (1)" src="https://github.com/user-attachments/assets/b2914173-a9da-463b-a81b-4d8d801efb06" />

### Latent analiz görselleri
- `06A_latent_pca_numbers.png`
<img width="1416" height="1247" alt="06A_latent_pca_numbers (1)" src="https://github.com/user-attachments/assets/ad4609ab-f745-4ffe-93c3-9f139159f4e3" />

- `06B_latent_pca_text_tr.png`
<img width="1640" height="1247" alt="06B_latent_pca_text_tr (1)" src="https://github.com/user-attachments/assets/9a0abe33-f4ec-464b-ba91-28ca3ef888ed" />

- `06C_latent_umap_text_tr.png`
<img width="1635" height="1247" alt="06C_latent_umap_text_tr" src="https://github.com/user-attachments/assets/832e014e-db05-4dcb-a82b-8d61e1ec9e21" />

- `06D_latent_tsne_text_tr.png`

### Generation / Interpolation
- `07_generated_prior.png`
<img width="2354" height="744" alt="07_generated_prior (1)" src="https://github.com/user-attachments/assets/c304d20c-06ca-47fc-b48f-cbf7d24db8ea" />

- `08_latent_interpolation.png`
<img width="2859" height="397" alt="08_latent_interpolation (1)" src="https://github.com/user-attachments/assets/a640ffd7-a38e-49dc-abaa-dfdfc3540a89" />

### Bölüm başlık panelleri (sunum için)
- `A_title_denoising.png`
<img width="1700" height="850" alt="A_title_denoising (1)" src="https://github.com/user-attachments/assets/e87e7d6c-edfd-4890-aada-6a50eabad9d6" />

- `B_title_representation_learning.png`
<img width="1194" height="1185" alt="B_title_representation_learning (1)" src="https://github.com/user-attachments/assets/efc9445b-0376-455d-a66c-5c37996275e1" />

- `C_title_generation.png`
<img width="2506" height="556" alt="C_title_generation (1)" src="https://github.com/user-attachments/assets/6c78fac9-6fbe-41d1-bfeb-b20532620e03" />

- `D_title_interpolation.png`
<img width="2579" height="448" alt="D_title_interpolation (1)" src="https://github.com/user-attachments/assets/5f20d868-cd14-4de8-b20c-f920f7fd00ab" />


### Log / rapor dosyaları
- `history.csv` (epoch bazlı eğitim geçmişi)
- `metrics.csv` (MSE/PSNR/SSIM tablo)
- `metrics.txt` (okunabilir metrik raporu)
- === Denoising Metrics (Test) ===
- NOISE_FACTOR=0.45
- LATENT_DIM=16
- BATCH_SIZE=256
- EPOCHS=20
- ===
- Baseline (Noisy->Clean)  MSE=0.101500 | PSNR=9.945 dB | SSIM=0.2871
- VAE (Recon->Clean)       MSE=0.018064 | PSNR=17.962 dB | SSIM=0.5950
- === Improvement ===
- MSE improvement: 82.20%
- PSNR gain      : 8.017 dB
- SSIM gain      : 0.3079

- `pca_explained_variance.txt` (PCA varyans oranları)
- PCA Explained Variance Ratio (2 components)
- [0.14242163 0.11660043]
- Sum: 0.25902206

---

## 8) Test Metrikleri (Notebook Örneği)

Notebook, test setinde baseline ile VAE’yi karşılaştırır:

- **Baseline (Noisy→Clean):** MSE, PSNR, SSIM
- **VAE (Recon→Clean):** MSE, PSNR, SSIM

Ardından iyileşmeyi hesaplar:
- MSE improvement (%)
- PSNR gain (dB)
- SSIM gain

Bu sonuçlar:
- `metrics.txt` ve `metrics.csv` içine kaydedilir.

---

## 9) Notlar

- PCA 2D bir projeksiyon olduğu için sınıflar arasında **tam ayrışma** garanti değildir.
- Ayrışmayı daha iyi görmek için UMAP/t-SNE kullanılmaktadır.
- SupCon loss, latent uzayda sınıf ayrışmasını güçlendirmek için eklenmiştir.

---
