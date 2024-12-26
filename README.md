# Stock-Prediction-for-VN30-Index-Stocks-in-Vietnam
This project develops statistical models on VNINDEX stock time series data to forecast price trends. By applying data preprocessing, parameter optimization, and model training, it utilizes methods such as ARIMA, VAR, Holt-Winters, and Facebook Prophet, showcasing their effectiveness in time series prediction.

## Dưới đây là mô tả các bước trong dự án **dự đoán chứng khoán**, bắt đầu từ bước 
### 1. **Data Mining (Khai thác dữ liệu)**

#### 1.1. **Identifying Data Sources_Xác định nguồn dữ liệu**
   - Tìm kiếm các nguồn dữ liệu chứng khoán đáng tin cậy  Vietstock, v.v.
   - Lựa chọn các cổ phiếu trong nhóm VN30 hoặc các chỉ số liên quan để làm đối tượng phân tích.

#### 1.2. **Data Collection_Thu thập dữ liệu**
   - **Thu thập dữ liệu lịch sử**: Lấy dữ liệu lịch sử về giá cổ phiếu, bao gồm giá mở cửa, giá đóng cửa, giá cao, giá thấp, và khối lượng giao dịch.

#### 1.3. **Data Cleaning_Làm sạch dữ liệu**
   - **Handling Missing Data_Xử lý dữ liệu thiếu**: Kiểm tra và xử lý các giá trị thiếu trong bộ dữ liệu (loại bỏ hoặc điền vào dữ liệu thiếu).
   - **Removing Outliers_Loại bỏ outliers**: Phát hiện và loại bỏ các giá trị bất thường (outliers) có thể làm sai lệch kết quả phân tích.
   - **Data Normalization_Chuẩn hóa dữ liệu**: Đảm bảo rằng tất cả các đặc trưng (features) trong dữ liệu có định dạng và đơn vị giống nhau để dễ dàng xử lý.

#### 1.4. **Data Preprocessing_Tiền xử lý dữ liệu**
   - **Chuyển đổi dữ liệu thành dạng thích hợp**: Chuyển dữ liệu thô thành các định dạng có thể sử dụng trong mô hình.
   - **Data Splitting_Phân chia dữ liệu**: Chia dữ liệu thành các tập huấn luyện và tập kiểm tra (train/test) để sử dụng trong việc xây dựng và đánh giá mô hình.


#   S t o c k - P r e d i c t i o n - f o r - V N 3 0 - I n d e x - S t o c k s - i n - V i e t n a m 
 
 