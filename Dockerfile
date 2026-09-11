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
