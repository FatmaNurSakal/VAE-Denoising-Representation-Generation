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
- `02_train_recon_loss.png`
- `03_train_kl_loss.png`
- `03B_train_supcon_loss.png`
- `04_beta_warmup.png`

### Denoising görseli
- `05_denoising_grid.png`

### Latent analiz görselleri
- `06A_latent_pca_numbers.png`
- `06B_latent_pca_text_tr.png`
- `06C_latent_umap_text_tr.png`
- `06D_latent_tsne_text_tr.png`

### Generation / Interpolation
- `07_generated_prior.png`
- `08_latent_interpolation.png`

### Bölüm başlık panelleri (sunum için)
- `A_title_denoising.png`
- `B_title_representation_learning.png`
- `C_title_generation.png`
- `D_title_interpolation.png`

### Log / rapor dosyaları
- `history.csv` (epoch bazlı eğitim geçmişi)
- `metrics.csv` (MSE/PSNR/SSIM tablo)
- `metrics.txt` (okunabilir metrik raporu)
- `pca_explained_variance.txt` (PCA varyans oranları)

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
