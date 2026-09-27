# 📈 Web App Kiểm Định Chiến Lược Trading EMA & OBV (Stock Backtesting App)

Ứng dụng web tương tác xây dựng trên nền tảng **Streamlit** giúp kiểm định, trực quan hóa và tối ưu hóa chiến lược giao dịch kết hợp chỉ báo kỹ thuật **EMA (Exponential Moving Average)** và **OBV (On-Balance Volume)** cho cổ phiếu (ví dụ: ACB, FPT, VNM, TCB, HPG,...) dựa trên dữ liệu giá lịch sử.

---

## 🌟 Tính Năng Nổi Bật

1. **Đa Dạng Nguồn Dữ Liệu**:
   - Sử dụng dữ liệu giá mặc định của cổ phiếu ACB (`ACB.csv`).
   - Hỗ trợ người dùng tải file CSV tùy chỉnh của bất kỳ mã cổ phiếu nào (cần có các cột `Date`, `Close`, `Volume`).

2. **Hạch Toán Vị Thế Chuẩn (Position Accounting)**:
   - Mô phỏng thực tế với trạng thái giữ tiền mặt / giữ cổ phiếu (**Long-only**).
   - Tự động bỏ qua các tín hiệu Mua trùng lặp khi đang nắm giữ vị thế.
   - Hỗ trợ cài đặt mức cắt lỗ **Stop Loss** cố định (ví dụ: 7%).

3. **Triệt Tiêu Look-Ahead Bias**:
   - Tín hiệu giao dịch được dịch chuyển 1 phiên (`shift(1)`) để đảm bảo quyết định mua/bán ở phiên $T$ sử dụng dữ liệu kết thúc của phiên $T-1$.

4. **Trực Quan Hóa Tương Tác (Plotly Charts)**:
   - Biểu đồ nến / giá đóng cửa kết hợp đường EMA và các điểm đánh dấu lệnh Mua/Bán.
   - Biểu đồ Độ dốc OBV (OBV Slope).
   - Biểu đồ Đường vốn (Equity Curve) so sánh trực quan với chiến lược Mua & Nắm giữ (**Buy & Hold Benchmark**).
   - Biểu đồ Mức sụt giảm tài sản (**Drawdown %**).
   - Nhật ký chi tiết từng lệnh giao dịch (Trade Log) hỗ trợ xuất file CSV.

5. **Tối Ưu Hóa Tham Số (Train vs Test / Hold-out Validation)**:
   - Thuật toán quét lưới (Grid Search / Hyperopt) để tìm chu kỳ EMA và OBV tối ưu trên tập Train dựa trên chỉ số **Sharpe Ratio**.
   - Tích hợp bộ lọc **`MIN_TRADES >= 5`** giúp loại bỏ các tham số Sharpe ảo do không phát sinh đủ giao dịch.
   - Bảng so sánh chi tiết hiệu suất bộ tham số Mặc định vs Tối ưu trên cả tập Train và tập Test (Hold-out).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
Tạo app trading/
├── app.py              # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt    # Danh sách thư viện Python phụ thuộc
├── README.md           # Hướng dẫn sử dụng & Deploy
└── ACB.csv             # Dữ liệu mẫu cổ phiếu ACB
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Trên Máy Cục Bộ (Local)

### 1. Yêu cầu môi trường
- Python 3.9 trở lên.
- Trình duyệt web (Chrome, Edge, Firefox, Brave,...).

### 2. Cài đặt các thư viện phụ thuộc
Mở terminal/command prompt tại thư mục dự án và chạy lệnh:

```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng Streamlit
Chạy lệnh sau để khởi chạy web app:

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở giao diện trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Tải Mã Nguồn Lên GitHub & Triển Khai (Deploy) Trên Streamlit Cloud

### BƯỚC 1: Tải mã nguồn lên GitHub

1. Tạo một tài khoản tại [GitHub.com](https://github.com/) (nếu chưa có).
2. Tạo một Repository mới trên GitHub (ví dụ đặt tên: `stock-trading-ema-obv`).
3. Tại thư mục dự án trên máy tính, mở Terminal / Git Bash và thực hiện các lệnh sau:

```bash
# Khởi tạo git repository
git init

# Thêm tất cả các file vào git
git add app.py requirements.txt README.md ACB.csv

# Tạo commit đầu tiên
git commit -m "Initial commit - Streamlit Trading App EMA OBV"

# Đổi tên branch chính thành main
git branch -M main

# Liên kết với repository trên GitHub (Thay URL bằng link repo của bạn)
git remote add origin https://github.com/USERNAME/stock-trading-ema-obv.git

# Đẩy mã nguồn lên GitHub
git push -u origin main
```

---

### BƯỚC 2: Triển khai (Deploy) ứng dụng lên Streamlit Community Cloud

1. Truy cập trang web [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"Create app"** (hoặc **"New app"**).
3. Chọn tùy chọn **"I already have an app"**.
4. Điền các thông tin triển khai:
   - **Repository:** Chọn repository `USERNAME/stock-trading-ema-obv` vừa tạo trên GitHub.
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Nhấn **"Deploy!"**.

Streamlit Cloud sẽ tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy ứng dụng của bạn trong khoảng 1–2 phút. Bạn sẽ nhận được một đường link public để chia sẻ web app cho mọi người truy cập!

---

## 📊 Phương Pháp Luận Chiến Lược Trading

- **Tín hiệu Mua (Entry Signal):**
  $$Close_t > EMA(N)_t \quad \text{AND} \quad OBV\_Slope(M)_t > 0$$
- **Tín hiệu Bán (Exit Signal):**
  $$Close_t < EMA(N)_t$$
- **Loại bỏ Look-ahead bias:** Tín hiệu ở ngày $t-1$ được dùng để giao dịch ở ngày $t$.
- **Cắt lỗ (Stop-loss):** Nếu giá sụt giảm $\ge 7\%$ so với giá mua ($Entry\_Price$), vị thế sẽ tự động đóng.

---

## ✉️ Hỗ Trợ & Đóng Góp
Nếu bạn có bất kỳ câu hỏi hoặc góp ý nào để cải thiện ứng dụng, hãy mở một *Issue* hoặc gửi *Pull Request* trên GitHub nhé!
