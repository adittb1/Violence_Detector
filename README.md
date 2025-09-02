# 🛡️ Violence Detector

Aplikasi **Violence Detector** menggunakan deep learning untuk mendeteksi aksi kekerasan pada video secara real-time.  
Aplikasi ini dilengkapi dengan integrasi **Firebase Realtime Database** dan notifikasi **Telegram Bot**.

---

## ⚙️ Instalasi

**Clone repository:**
```bash
git clone https://github.com/username/violence_detector.git
cd violence_detector
```

**Install dependencies:**
```bash
pip install -r req.txt
```

---

## 📂 Dataset

1. **RWF-2000**  
   Download dataset dari [RWF-2000 Dataset](https://github.com/mchengny/RWF2000-Video-Database).  
   Ekstrak ke folder `RWF-2000/`.

2. **Primer Dataset (Custom)**  
   Siapkan folder `Primer/` berisi dataset tambahan (jika ada).

**Struktur folder dataset:**
```
violence_detector/
│── RWF-2000/
│── Primer/
│── train.ipynb
│── app.py
│── serviceAccountKey.json
```

---

## 🏋️ Training

Jalankan notebook training:
```bash
jupyter notebook train.ipynb
```

Tunggu hingga proses training selesai dan model tersimpan.

---

## 🔑 Konfigurasi Firebase

Buat file `serviceAccountKey.json` di root project dengan isi:

```json
{
  "type": "service_account",
  "project_id": "violence-detection---comvis",
  "private_key_id": "",
  "private_key": "",
  "client_email": "",
  "client_id": "",
  "auth_uri": "",
  "token_uri": "",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40violence-detection---comvis.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```

⚠️ **Isi bagian `private_key_id`, `private_key`, `client_email`, `client_id`, dan URL sesuai kredensial dari Firebase Console.**

---

## 📝 Konfigurasi `app.py`

Edit bagian konfigurasi pada `app.py`:

```python
# =========================
# 🔧 KONFIGURASI
# =========================
# Telegram
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

# Firebase
FIREBASE_CRED_JSON = "serviceAccountKey.json"
FIREBASE_DB_URL = "https://your-project-id.firebaseio.com/"
```

---

## ▶️ Menjalankan Aplikasi

Setelah konfigurasi selesai, jalankan:

```bash
python app.py
```

Jika ada aksi kekerasan terdeteksi:
- Data akan dikirim ke **Firebase Realtime Database**
- Notifikasi akan dikirim ke **Telegram Bot**

---

## 📌 Catatan
- Pastikan sudah membuat **Telegram Bot** via [@BotFather](https://t.me/BotFather).
- Gunakan **Chat ID** sesuai grup/user tujuan notifikasi.
- Model hasil training disimpan otomatis setelah `train.ipynb` selesai dijalankan.

---

## 📜 Lisensi
Proyek ini dikembangkan untuk tujuan penelitian dan pembelajaran.  
Gunakan dengan bijak dan **hanya untuk keperluan akademik**.
