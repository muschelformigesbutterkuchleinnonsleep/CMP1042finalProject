import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

def main():
    st.title("Phân tích mối quan hệ giữa biến và Mô hình hóa dữ liệu")
    
    # Upload file dữ liệu
    st.sidebar.header("Upload dữ liệu")
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV hoặc Excel", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Đọc file dữ liệu
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Hiển thị dữ liệu
            st.header("Dữ liệu đã tải lên")
            st.write(df.head())
            
            # Thông tin cơ bản về dữ liệu
            st.subheader("Thông tin dữ liệu")
            buffer = st.expander("Hiển thị thông tin chi tiết")
            with buffer:
                st.write(f"Số dòng: {df.shape[0]}")
                st.write(f"Số cột: {df.shape[1]}")
                st.write("Các loại dữ liệu:")
                st.write(df.dtypes)
                st.write("Thống kê mô tả:")
                st.write(df.describe())
                st.write("Kiểm tra giá trị null:")
                st.write(df.isnull().sum())
            
            # Chọn biến để phân tích
            st.header("Chọn biến để phân tích")
            
            # Lọc ra các cột số
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Dữ liệu cần có ít nhất 2 biến số để phân tích mối quan hệ.")
                return
                
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Chọn biến X (độc lập)", numeric_cols)
            with col2:
                y_var = st.selectbox("Chọn biến Y (phụ thuộc)", [col for col in numeric_cols if col != x_var])
            
            # Vẽ biểu đồ phân tán
            st.header("Biểu đồ phân tán và phân tích mối quan hệ")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
            sns.regplot(data=df, x=x_var, y=y_var, scatter=False, ax=ax, color='red')
            ax.set_title(f'Biểu đồ phân tán giữa {x_var} và {y_var}')
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            st.pyplot(fig)
            
            # Kiểm tra giá trị null trong các biến được chọn
            if df[x_var].isnull().sum() > 0 or df[y_var].isnull().sum() > 0:
                st.warning("Có giá trị null trong dữ liệu. Đang loại bỏ các dòng có giá trị null để phân tích.")
                df_clean = df[[x_var, y_var]].dropna()
            else:
                df_clean = df[[x_var, y_var]]
                
            if df_clean.shape[0] < 2:
                st.error("Không đủ dữ liệu để phân tích sau khi loại bỏ giá trị null.")
                return
            
            # Phân tích thống kê
            st.header("Phân tích thống kê mối quan hệ")
            
            # Tính hệ số tương quan
            pearson_coef, pearson_p = pearsonr(df_clean[x_var], df_clean[y_var])
            spearman_coef, spearman_p = spearmanr(df_clean[x_var], df_clean[y_var])
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Hệ số tương quan")
                st.write(f"Hệ số tương quan Pearson: {pearson_coef:.4f} (p-value: {pearson_p:.4f})")
                st.write(f"Hệ số tương quan Spearman: {spearman_coef:.4f} (p-value: {spearman_p:.4f})")
                
                # Diễn giải hệ số tương quan
                if abs(pearson_coef) < 0.3:
                    correlation_strength = "yếu"
                elif abs(pearson_coef) < 0.7:
                    correlation_strength = "trung bình"
                else:
                    correlation_strength = "mạnh"
                    
                correlation_direction = "dương" if pearson_coef > 0 else "âm"
                st.write(f"Mối quan hệ {correlation_strength} và {correlation_direction} giữa {x_var} và {y_var}.")
                
                # Đánh giá ý nghĩa thống kê
                alpha = 0.05
                if pearson_p < alpha:
                    st.write(f"Mối quan hệ có ý nghĩa thống kê (p < {alpha}).")
                else:
                    st.write(f"Mối quan hệ không có ý nghĩa thống kê (p > {alpha}).")
            
            # Mô hình hóa ML
            st.header("Ứng dụng mô hình Machine Learning")
            
            # Chuẩn bị dữ liệu
            X = df_clean[x_var].values.reshape(-1, 1)
            y = df_clean[y_var].values
            
            # Chia dữ liệu thành tập huấn luyện và kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Chọn mô hình
            model_type = st.selectbox("Chọn mô hình ML", ["Linear Regression", "Random Forest Regression"])
            
            if model_type == "Linear Regression":
                # Huấn luyện mô hình hồi quy tuyến tính
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Dự đoán
                y_pred = model.predict(X_test)
                
                # Metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Kết quả mô hình hồi quy tuyến tính")
                    st.write(f"Hệ số góc (m): {model.coef_[0]:.4f}")
                    st.write(f"Hệ số tự do (b): {model.intercept_:.4f}")
                    st.write(f"Phương trình: {y_var} = {model.coef_[0]:.4f} × {x_var} + {model.intercept_:.4f}")
                    st.write(f"R² (Coefficient of determination): {r2:.4f}")
                    st.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
                
                with col2:
                    # Vẽ biểu đồ kết quả mô hình
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(X_test, y_test, color='blue', alpha=0.7, label='Dữ liệu thực tế')
                    ax.plot(X_test, y_pred, color='red', linewidth=2, label='Dự đoán')
                    ax.set_title('So sánh giữa giá trị thực tế và dự đoán')
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                    ax.legend()
                    st.pyplot(fig)
                
                # Phân tích chi tiết hơn với statsmodels
                X_sm = sm.add_constant(X)
                model_sm = sm.OLS(y, X_sm).fit()
                st.subheader("Phân tích hồi quy chi tiết")
                st.text(model_sm.summary().as_text())
                
            else:  # Random Forest
                # Huấn luyện mô hình Random Forest
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Dự đoán
                y_pred = model.predict(X_test)
                
                # Metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Kết quả mô hình Random Forest")
                    st.write(f"R² (Coefficient of determination): {r2:.4f}")
                    st.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
                    st.write(f"Feature importance: {model.feature_importances_[0]:.4f}")
                
                with col2:
                    # Vẽ biểu đồ kết quả mô hình
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(X_test, y_test, color='blue', alpha=0.7, label='Dữ liệu thực tế')
                    ax.scatter(X_test, y_pred, color='red', alpha=0.7, label='Dự đoán')
                    ax.set_title('So sánh giữa giá trị thực tế và dự đoán')
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                    ax.legend()
                    st.pyplot(fig)
            
            # Công cụ dự đoán
            st.header("Công cụ dự đoán")
            user_input = st.number_input(f"Nhập giá trị của {x_var} để dự đoán {y_var}", value=float(df[x_var].mean()))
            
            if st.button("Dự đoán"):
                prediction = model.predict([[user_input]])[0]
                st.success(f"Dự đoán {y_var} cho {x_var} = {user_input}: {prediction:.4f}")
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")
    
    else:
        # Hiển thị hướng dẫn
        st.info("Tải lên file CSV hoặc Excel có ít nhất 2 cột số để bắt đầu phân tích.")
        st.write("""
        ### Hướng dẫn sử dụng:
        1. Tải lên file dữ liệu CSV hoặc Excel từ menu bên trái
        2. Chọn hai biến số cần phân tích mối quan hệ
        3. Xem biểu đồ phân tán và phân tích thống kê
        4. Ứng dụng mô hình ML để hiểu sâu hơn về mối quan hệ giữa các biến
        5. Sử dụng công cụ dự đoán để ước tính giá trị
        
        ### Lưu ý:
        - Dữ liệu phải có ít nhất 2 cột số
        - Nếu dữ liệu có giá trị null, chúng sẽ bị loại bỏ khi phân tích
        """)
        
        # Tạo dữ liệu mẫu để minh họa
        st.subheader("Dữ liệu mẫu:")
        sample_data = pd.DataFrame({
            'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Y': [2, 3.9, 6.1, 8, 9.8, 12.2, 14.1, 16, 17.9, 20.1]
        })
        st.write(sample_data)
        
        # Vẽ biểu đồ mẫu
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=sample_data, x='X', y='Y', ax=ax)
        sns.regplot(data=sample_data, x='X', y='Y', scatter=False, ax=ax, color='red')
        ax.set_title('Ví dụ: Biểu đồ phân tán và đường hồi quy')
        st.pyplot(fig)

if __name__ == "__main__":
    main()