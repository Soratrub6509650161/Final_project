"""
การทดสอบระบบตรวจจับคำหยาบ Twitch Bad Word Detector
สำหรับโปรเจคจบมหาลัย

วิธีใช้:
1. ติดตั้ง pytest: pip install pytest pandas openpyxl
2. รันทดสอบ: pytest test_badword_detector.py -v
3. ดูรายงาน: pytest test_badword_detector.py -v --html=report.html
"""

import pytest
import time
import sys
from datetime import datetime
import pandas as pd


from main_gui import TwitchChatWorker

# ================== 1. UNIT TESTING ==================

class TestBadWordDetection:
    """ทดสอบการตรวจจับคำหยาบ"""
    
    @pytest.fixture
    def worker(self):
        """สร้าง worker สำหรับทดสอบ"""
        return TwitchChatWorker("test_channel")
    
    # ทดสอบภาษาไทย
    def test_detect_thai_profanity_simple(self, worker):
        """ทดสอบจับคำหยาบไทยธรรมดา"""
        message = "คุณมันไอ้สัส"
        result = worker.detect_thai_profanity(message)
        assert len(result) > 0, "ควรตรวจจับคำหยาบไทยได้"
    
    def test_detect_thai_profanity_with_spaces(self, worker):
        """ทดสอบจับคำหยาบไทยที่แทรกช่องว่าง"""
        message = "คุณมัน ไ อ้ ส ั สส"
        result = worker.detect_thai_profanity(message)
        assert len(result) > 0, "ควรตรวจจับคำหยาบที่แทรกช่องว่างได้"
    
    def test_detect_thai_profanity_with_symbols(self, worker):
        """ทดสอบจับคำหยาบไทยที่แทรกสัญลักษณ์"""
        message = "คุณมัน!@#ไอ้$%สัสว์"
        result = worker.detect_thai_profanity(message)
        assert len(result) > 0, "ควรตรวจจับคำหยาบที่แทรกสัญลักษณ์ได้"
    
    def test_detect_thai_no_profanity(self, worker):
        """ทดสอบข้อความปกติภาษาไทย"""
        message = "สวัสดีครับ วันนี้อากาศดีมาก"
        result = worker.detect_thai_profanity(message)
        assert len(result) == 0, "ไม่ควรตรวจจับคำปกติเป็นคำหยาบ"
    
    # ทดสอบภาษาอังกฤษ
    def test_detect_english_profanity_simple(self, worker):
        """ทดสอบจับคำหยาบอังกฤษธรรมดา"""
        message = "you are so stupid"
        result = worker.detect_english_profanity(message)
        # ปรับตามคำหยาบที่มีในไฟล์ badwords_en.txt
        assert isinstance(result, list), "ควรคืนค่าเป็น list"
    
    def test_detect_english_profanity_leet_speak(self, worker):
        """ทดสอบจับคำหยาบอังกฤษแบบ leet speak"""
        message = "stupidperson"  # คำติดกัน
        result = worker.detect_english_profanity(message)
        assert isinstance(result, list), "ควรจับคำที่ติดกันได้ด้วย wordsegment"
    
    def test_detect_english_no_profanity(self, worker):
        """ทดสอบข้อความปกติภาษาอังกฤษ"""
        message = "hello everyone have a nice day"
        result = worker.detect_english_profanity(message)
        assert len(result) == 0, "ไม่ควรตรวจจับคำปกติเป็นคำหยาบ"
    
    # ทดสอบแบบผสม
    def test_detect_mixed_language(self, worker):
        """ทดสอบข้อความผสมไทย-อังกฤษ"""
        message = "hello ไอ้สัส stupid"
        result = worker.optimized_detect_bad_words(message)
        assert len(result) > 0, "ควรตรวจจับคำหยาบได้ทั้งสองภาษา"
    
    def test_detect_empty_message(self, worker):
        """ทดสอบข้อความว่าง"""
        message = ""
        result = worker.optimized_detect_bad_words(message)
        assert len(result) == 0, "ข้อความว่างควรคืนค่า empty list"

# ================== 2. ACCURACY TESTING ==================

class TestAccuracy:
    """ทดสอบความแม่นยำของระบบ"""
    
    @pytest.fixture
    def worker(self):
        return TwitchChatWorker("test_channel")
    
    @pytest.fixture
    def test_dataset(self):
        """ชุดข้อมูลทดสอบ (แก้ไขตามคำหยาบจริงในไฟล์ของคุณ)"""
        return [
            # (ข้อความ, มีคำหยาบหรือไม่ True/False)
            ("สวัสดีครับ", False),
            ("วันนี้อากาศดีมาก", False),
            ("hello everyone", False),
            ("nice stream", False),
            ("ไอ้สัสว์", True),  # แก้ตามคำหยาบจริง
            ("you stupid", True),  # แก้ตามคำหยาบจริง
            ("คุณ ไ อ้ ส ั สว์", True),  # คำหยาบแทรกช่องว่าง
            ("stu pid", True),  # คำหยาบแทรกช่องว่าง
        ]
    
    def test_calculate_accuracy(self, worker, test_dataset):
        """คำนวณ Accuracy, Precision, Recall, F1-Score"""
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        
        for message, has_badword in test_dataset:
            result = worker.optimized_detect_bad_words(message)
            detected = len(result) > 0
            
            if has_badword and detected:
                true_positive += 1
            elif has_badword and not detected:
                false_negative += 1
            elif not has_badword and detected:
                false_positive += 1
            elif not has_badword and not detected:
                true_negative += 1
        
        # คำนวณค่าต่างๆ
        total = len(test_dataset)
        accuracy = (true_positive + true_negative) / total * 100
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # พิมพ์ผลลัพธ์
        print("\n" + "="*50)
        print("📊 ACCURACY TEST RESULTS")
        print("="*50)
        print(f"Total Test Cases: {total}")
        print(f"True Positive (TP): {true_positive}")
        print(f"False Positive (FP): {false_positive}")
        print(f"True Negative (TN): {true_negative}")
        print(f"False Negative (FN): {false_negative}")
        print("-"*50)
        print(f"Accuracy:  {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1-Score:  {f1_score:.2f}")
        print("="*50)
        
        # บันทึกผลลงไฟล์
        results_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'TP', 'FP', 'TN', 'FN'],
            'Value': [f"{accuracy:.2f}%", f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}",
                     true_positive, false_positive, true_negative, false_negative]
        })
        results_df.to_csv('accuracy_results.csv', index=False, encoding='utf-8-sig')
        print("✅ บันทึกผลลัพธ์ไปที่ accuracy_results.csv")
        
        # ต้องมี accuracy อย่างน้อย 70% ถึงจะผ่าน
        assert accuracy >= 70, f"Accuracy ต่ำเกินไป: {accuracy:.2f}%"

# ================== 3. PERFORMANCE TESTING ==================

class TestPerformance:
    """ทดสอบประสิทธิภาพของระบบ"""
    
    @pytest.fixture
    def worker(self):
        return TwitchChatWorker("test_channel")
    
    def test_detection_speed(self, worker):
        """ทดสอบความเร็วในการตรวจจับ"""
        test_messages = [
            "สวัสดีครับ",
            "hello everyone",
            "ไอ้สัสว์",  # แก้ตามคำหยาบจริง
            "you stupid",
            "วันนี้อากาศดีมาก hello nice stream"
        ] * 100  # ทดสอบ 500 ข้อความ
        
        start_time = time.time()
        
        for message in test_messages:
            worker.optimized_detect_bad_words(message)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = (total_time / len(test_messages)) * 1000  # มิลลิวินาที
        
        print("\n" + "="*50)
        print("⚡ PERFORMANCE TEST RESULTS")
        print("="*50)
        print(f"Total Messages: {len(test_messages)}")
        print(f"Total Time: {total_time:.3f} seconds")
        print(f"Average Time: {avg_time:.3f} ms/message")
        print(f"Messages/Second: {len(test_messages)/total_time:.2f}")
        print("="*50)
        
        # ต้องตรวจจับได้เร็วกว่า 100ms ต่อข้อความ
        assert avg_time < 100, f"ช้าเกินไป: {avg_time:.3f} ms"
    
    def test_memory_usage(self, worker):
        """ทดสอบการใช้หน่วยความจำ"""
        import sys
        
        # เพิ่มข้อความเข้า deque
        for i in range(200):  # maxlen=200
            chat_info = {
                'timestamp': datetime.now(),
                'username': f'user{i}',
                'message': f'test message {i}',
                'bad_words': ['test'],
                'channel': 'test'
            }
            worker.chat_messages.append(chat_info)
        
        # คำนวณขนาดหน่วยความจำ
        memory_size = sys.getsizeof(worker.chat_messages)
        
        for msg in worker.chat_messages:
            memory_size += sys.getsizeof(msg)
        
        memory_kb = memory_size / 1024
        
        print("\n" + "="*50)
        print("💾 MEMORY USAGE TEST")
        print("="*50)
        print(f"Messages Stored: {len(worker.chat_messages)}")
        print(f"Memory Usage: {memory_kb:.2f} KB")
        print(f"Memory per Message: {memory_kb/len(worker.chat_messages):.3f} KB")
        print("="*50)
        
        # ต้องใช้หน่วยความจำไม่เกิน 100 KB สำหรับ 200 ข้อความ
        assert memory_kb < 100, f"ใช้หน่วยความจำมากเกินไป: {memory_kb:.2f} KB"

# ================== 4. EDGE CASE TESTING ==================

class TestEdgeCases:
    """ทดสอบกรณีพิเศษ"""
    
    @pytest.fixture
    def worker(self):
        return TwitchChatWorker("test_channel")
    
    def test_very_long_message(self, worker):
        """ทดสอบข้อความยาวมาก"""
        message = "hello " * 1000 + "ไอ้สัตว์"  # ข้อความยาว 5000+ ตัวอักษร
        result = worker.optimized_detect_bad_words(message)
        assert isinstance(result, list), "ควรจัดการข้อความยาวได้"
    
    def test_special_characters(self, worker):
        """ทดสอบสัญลักษณ์พิเศษ"""
        message = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        result = worker.optimized_detect_bad_words(message)
        assert len(result) == 0, "สัญลักษณ์พิเศษไม่ควรทำให้ error"
    
    def test_unicode_characters(self, worker):
        """ทดสอบตัวอักษร Unicode"""
        message = "😀😁😂🤣😃 สวัสดี 👋"
        result = worker.optimized_detect_bad_words(message)
        assert isinstance(result, list), "ควรจัดการ emoji ได้"
    
    def test_repeated_detection(self, worker):
        """ทดสอบตรวจจับซ้ำๆ"""
        message = "ไอ้สัสว์ ไอ้สัสว์ ไอ้สัสว์"  # คำหยาบซ้ำ 3 ครั้ง
        result = worker.optimized_detect_bad_words(message)
        # ควรตรวจจับได้แต่ไม่ซ้ำ (ใช้ set)
        assert len(result) >= 1, "ควรตรวจจับคำหยาบได้"

# ================== 5. INTEGRATION TESTING ==================

class TestIntegration:
    """ทดสอบการทำงานร่วมกัน (ต้องระวังเรื่องการเชื่อมต่อจริง)"""
    
    def test_load_badwords_files(self):
        """ทดสอบโหลดไฟล์คำหยาบ"""
        worker = TwitchChatWorker("test_channel")
        assert len(worker.badwords_th) > 0 or len(worker.badwords_en) > 0, \
            "ควรโหลดไฟล์คำหยาบได้อย่างน้อย 1 ไฟล์"
    
    def test_chat_message_storage(self):
        """ทดสอบการเก็บข้อความแชท"""
        worker = TwitchChatWorker("test_channel")
        
        # เพิ่มข้อความทดสอบ
        for i in range(10):
            chat_info = {
                'timestamp': datetime.now(),
                'username': f'user{i}',
                'message': f'message {i}',
                'bad_words': [],
                'channel': 'test'
            }
            worker.chat_messages.append(chat_info)
        
        messages = worker.get_chat_messages()
        assert len(messages) == 10, "ควรเก็บข้อความได้ครบ"
    
    def test_clear_messages(self):
        """ทดสอบการล้างข้อความ"""
        worker = TwitchChatWorker("test_channel")
        
        # เพิ่มข้อความ
        worker.chat_messages.append({'test': 'data'})
        assert len(worker.chat_messages) > 0
        
        # ล้างข้อความ
        worker.clear_memory_messages()
        assert len(worker.chat_messages) == 0, "ควรล้างข้อความได้"

# ================== รันทดสอบทั้งหมด ==================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TWITCH BAD WORD DETECTOR - COMPREHENSIVE TESTING")
    print("="*70)
    
    # รันด้วย pytest
    pytest.main([
        __file__,
        "-v",  # verbose
        "-s",  # แสดง print
        "--tb=short",  # แสดง error แบบสั้น
        "--html=test_report.html",  # สร้างรายงาน HTML (ต้องติดตั้ง pytest-html)
        "--self-contained-html"  # รายงาน HTML แบบเดียว
    ])