# Hướng Dẫn Chi Tiết Triển Khai Ứng Dụng Phân Loại Email Spam (SVM + FastAPI + Docker + ngrok)

Dự án này hướng dẫn xây dựng và triển khai ứng dụng Học máy (Machine Learning) phân loại Email Spam dựa trên thuật toán **Support Vector Machine (SVM)** (Nội dung Buổi 07), phục vụ qua REST API (**FastAPI**), đóng gói bằng **Docker Desktop** và chia sẻ ra Internet bằng **ngrok**.

---

## 1. Cấu Trúc Dự Án (`spam_svm_app`)

Khi hoàn thành, cấu trúc thư mục dự án sẽ như sau:

```text
spam_svm_app/
├── dataset/                  # Thư mục chứa tập dữ liệu sau khi giải nén
│   └── spam_ham_dataset.csv  # File CSV chứa dữ liệu email và nhãn (spam/ham)
├── train.py                  # Script đọc dataset, huấn luyện SVM và xuất file mô hình (.pkl)
├── app.py                    # REST API web server (FastAPI) nhận text và trả về dự đoán
├── requirements.txt          # Danh sách các thư viện Python cần thiết
├── Dockerfile                # Cấu hình môi trường đóng gói Docker Container
└── README.md                 # Tài liệu hướng dẫn cấu hình và vận hành dự án
```

---

## 2. Mã Nguồn Các Tệp Trong Dự Án

### 2.1. File `requirements.txt`

Chứa toàn bộ các thư viện phụ thuộc:

```text
fastapi
uvicorn
scikit-learn
joblib
pydantic
pandas
```

---

### 2.2. File `train.py` (Huấn luyện mô hình SVM)

Đọc dữ liệu từ `dataset/spam_ham_dataset.csv`, chuyển đổi văn bản qua **TF-IDF Vectorizer**, huấn luyện thuật toán **Support Vector Machine (SVC)** và lưu mô hình ra file `.pkl`:

```python
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def train():
    # 1. Kiểm tra và đọc dữ liệu từ thư mục dataset/
    dataset_path = os.path.join("dataset", "spam_ham_dataset.csv")

    if not os.path.exists(dataset_path):
        # Mẫu dữ liệu dự phòng nếu chưa có file CSV trong thư mục dataset/
        print("Cảnh báo: Không tìm thấy dataset/spam_ham_dataset.csv. Sử dụng dữ liệu mẫu dự phòng...")
        data = [
            ("Free money now! Win big prize!", "spam"),
            ("Urgent: Claim your cash bonus immediately.", "spam"),
            ("Meeting schedule for tomorrow at 10 AM.", "ham"),
            ("Please review the project report attached.", "ham"),
            ("Congratulations, you won a free gift card!", "spam"),
            ("Let's grab lunch together today.", "ham"),
        ]
        X_text, y_labels = zip(*data)
    else:
        print(f"Đang đọc dữ liệu từ: {dataset_path}")
        df = pd.read_csv(dataset_path)
        # Giả định cột văn bản là 'text' và nhãn là 'label' (hoặc 'label_num')
        X_text = df["text"] if "text" in df.columns else df.iloc[:, 0]
        y_labels = df["label"] if "label" in df.columns else df.iloc[:, 1]

    # 2. Biến đổi văn bản thành Vector đặc trưng (TF-IDF)
    print("Đang trích xuất đặc trưng TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(X_text)

    # 3. Huấn luyện thuật toán Support Vector Machine (Buổi 07 - SVM)
    print("Đang huấn luyện mô hình Support Vector Machine (SVM)...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X, y_labels)

    # 4. Lưu mô hình và bộ biến đổi ra file .pkl
    joblib.dump(model, "svm_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("-> Đã hoàn thành huấn luyện và lưu file 'svm_model.pkl', 'tfidf_vectorizer.pkl'!")

if __name__ == "__main__":
    train()
```

---

### 2.3. File `app.py` (FastAPI REST Service)

Đọc mô hình đã lưu và phục vụ API dự đoán email:

```python
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Spam Classification API - Support Vector Machine (SVM)")

# Nạp model và vectorizer
MODEL_PATH = "svm_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None

class EmailInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "SVM Email Spam Classification API đang hoạt động!",
        "status": "Ready" if model else "Model not found. Please run train.py first."
    }

@app.post("/predict")
def predict_spam(data: EmailInput):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Mô hình chưa được huấn luyện (thiếu file .pkl).")

    X_input = vectorizer.transform([data.text])
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    return {
        "text": data.text,
        "prediction": str(prediction),
        "confidence": float(max(probabilities))
    }
```

---

### 2.4. File `Dockerfile`

Đóng gói toàn bộ ứng dụng thành Container Image độc lập:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn và mô hình
COPY . .

# Mở cổng 8000
EXPOSE 8000

# Lệnh khởi chạy Web Server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. Quy Trình Hướng Dẫn Thực Hiện Từng Bước (Cho Copilot Agent / Developer)

### Bước 1: Chuẩn bị thư mục và giải nén Dữ liệu

1. Tạo thư mục `dataset/` trong thư mục gốc dự án.
2. Giải nén file `.zip` chứa tập dữ liệu Kaggle đã tải về, chép file CSV vào thư mục `dataset/` và đặt tên là `spam_ham_dataset.csv`.

### Bước 2: Cài đặt thư viện & Huấn luyện Mô hình

Chạy lệnh sau để huấn luyện mô hình và sinh ra 2 file `svm_model.pkl` và `tfidf_vectorizer.pkl`:

```bash
pip install -r requirements.txt
python train.py
```

### Bước 3: Đóng gói Docker Image

Mở **Docker Desktop**, sau đó mở terminal tại thư mục dự án và build image:

```bash
docker build -t spam-svm-app .
```

### Bước 4: Khởi chạy Docker Container

Chạy container và mở cổng 8000:

```bash
docker run -d -p 8000:8000 --name spam_svm_container spam-svm-app
```

_Kiểm tra API cục bộ:_ Truy cập trình duyệt tại `http://localhost:8000/docs`.

### Bước 5: Public API ra Internet qua ngrok

Mở một cửa sổ Terminal mới và chạy ngrok:

```bash
ngrok http 8000
```
