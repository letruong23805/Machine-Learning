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
