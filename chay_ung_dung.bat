@echo off
chcp 65001 > nul
echo =====================================================================
echo    KHỞI CHẠY HỆ THỐNG VIETNAMESE FOOD NUTRITION (YOLO + OLLAMA)
echo =====================================================================
echo.
echo [*] Đang kiểm tra kết nối Ollama...
curl -s http://localhost:11434/api/tags > nul
if %errorlevel% neq 0 (
    echo [!] CẢNH BÁO: Dịch vụ Ollama chưa được bật!
    echo [*] Hãy đảm bảo Ollama đang chạy trên máy của bạn.
    echo.
) else (
    echo [OK] Ollama đang chạy bình thường trên cổng 11434.
    echo.
)

echo [*] Khởi động ứng dụng Streamlit...
echo [>] Truy cập trình duyệt tại: http://localhost:8501
echo.
streamlit run main.py
pause
