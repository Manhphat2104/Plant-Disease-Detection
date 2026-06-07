import google.generativeai as genai
import os

def get_treatment_solution(disease_name):
   
    API_KEY = os.environ.get("GEMINI_API_KEY")
    
    if not API_KEY:
        return " Vui lòng thiết lập biến môi trường GEMINI_API_KEY với API Key của bạn."
    
    genai.configure(api_key=API_KEY.split())

    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')


    # Ép LLM đóng vai chuyên gia và trả lời theo format định sẵn
    prompt = f"""
    Bạn là một kỹ sư nông nghiệp chuyên nghiệp và tận tâm.
    Hệ thống thị giác máy tính của tôi vừa phát hiện cây trồng của nông dân đang mắc bệnh: "{disease_name}".
    
    Hãy đưa ra một giải pháp điều trị ngắn gọn, rõ ràng, dễ hiểu cho nông dân theo đúng cấu trúc sau (không cần giới thiệu dài dòng). BẮT BUỘC giữ nguyên tên bệnh và loại cây như tôi đã cung cấp:
    
    🌿 BỆNH: {disease_name}
    ⚠️ 1. Nguyên nhân gây bệnh: (Ngắn gọn 1-2 câu)
    ✂️ 2. Xử lý cấp tốc: (Cần làm gì ngay bây giờ để bệnh không lây lan)
    💊 3. Thuốc đặc trị / Biện pháp sinh học: (Gợi ý tên hoạt chất hoặc cách chữa dân gian)
    🛡️ 4. Phòng ngừa: (Làm sao để vụ mùa sau không bị lại)
    """

    try:
        print(f"\n🤖 Đang kết nối với Chuyên gia Nông nghiệp AI để tìm thuốc trị [{disease_name}]...")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Có lỗi xảy ra khi gọi LLM: {str(e)}"
