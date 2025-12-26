import os
import csv
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import google.generativeai as genai
from dotenv import load_dotenv

# --- CẤU HÌNH HỆ THỐNG ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CÔNG CỤ 1: TÍNH LOSS FUNCTION ---
def calculate_loss(y_true: list, y_pred: list, loss_type: str = "mse"):
    """Tính MSE, MAE hoặc Log Loss cho mô hình ML."""
    y_t, y_p = np.array(y_true), np.array(y_pred)
    if loss_type == "mse":
        return float(np.mean((y_t - y_p)**2))
    elif loss_type == "mae":
        return float(np.mean(np.abs(y_t - y_p)))
    elif loss_type == "log_loss":
        y_p = np.clip(y_p, 1e-15, 1 - 1e-15)
        return float(-np.mean(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p)))
    return "Loại Loss không hỗ trợ."

# --- CÔNG CỤ 2: VẼ ĐỒ THỊ TOÁN HỌC (Hàm số) ---
def plot_math_function(expression: str, start: float = -10, end: float = 10):
    """Vẽ đồ thị hàm số từ biểu thức (vd: x**2)."""
    x_sym = sp.symbols('x')
    f = sp.lambdify(x_sym, sp.sympify(expression), 'numpy')
    x_vals = np.linspace(start, end, 400)
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, f(x_vals), label=f"f(x) = {expression}", color='blue')
    plt.grid(True, linestyle='--')
    plt.legend()
    path = "math_plot.png"
    plt.savefig(path); plt.close()
    return f"✅ Đã vẽ đồ thị hàm số tại {path}"

# --- CÔNG CỤ 3: ĐỌC CSV ---
def read_csv_data(filepath: str, column_name: str = None):
    """Đọc dữ liệu từ file CSV (toàn bộ hoặc theo cột)."""
    if not os.path.exists(filepath): return "Lỗi: Không tìm thấy file."
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column_name:
                if column_name in row:
                    try: data.append(float(row[column_name]))
                    except: pass
                else: return f"Cột {column_name} không tồn tại."
            else: data.append(row)
    return data

# --- CÔNG CỤ 4: VẼ BIỂU ĐỒ SO SÁNH DỮ LIỆU CSV ---
def plot_csv_comparison(y_true: list, y_pred: list, title: str = "So sánh Thực tế vs Dự đoán"):
    """Vẽ biểu đồ đường so sánh kết quả mô hình từ dữ liệu CSV."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Thực tế (Actual)', marker='o', color='green')
    plt.plot(y_pred, label='Dự đoán (Predicted)', marker='x', linestyle='--', color='red')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    path = "csv_comparison.png"
    plt.savefig(path); plt.close()
    return f"✅ Đã vẽ biểu đồ so sánh tại {path}"

# --- CẤU HÌNH AGENT ---
SYSTEM_INSTRUCTION = """
Bạn là Chuyên gia AI Data Scientist. Quy trình của bạn:
1. Đọc dữ liệu CSV bằng 'read_csv_data'.
2. Tính toán sai số bằng 'calculate_loss'.
3. Trực quan hóa bằng 'plot_math_function' (cho hàm số) hoặc 'plot_csv_comparison' (cho dữ liệu CSV).
4. Giải thích kết quả bằng ngôn ngữ chuyên môn nhưng dễ hiểu.
"""

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    tools=[calculate_loss, plot_math_function, read_csv_data, plot_csv_comparison],
    system_instruction=SYSTEM_INSTRUCTION
)

# Bắt đầu vòng lặp chat
chat = model.start_chat(enable_automatic_function_calling=True)
print("--- AGENT DATA SCIENCE ĐÃ SẴN SÀNG ---")

# (Vòng lặp while True của bạn...)