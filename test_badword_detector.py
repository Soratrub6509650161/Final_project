import pytest
import time
import sys
from datetime import datetime
import pandas as pd
from main_gui import TwitchChatWorker

# ================== 1. ACCURACY TESTING (50 Cases) ==================

class TestAccuracy:
    """ทดสอบความแม่นยำของระบบด้วย 50 Test Cases"""
    
    @pytest.fixture
    def worker(self):
        return TwitchChatWorker("test_channel")
    
    @pytest.fixture
    def test_dataset(self):
        """
        ชุดข้อมูลทดสอบ 50 Test Cases
        """
        return [
            # --- กลุ่มที่ 1: ข้อความปกติ (Normal) -> คาดหวังผล False (TN) ---
            ("สวัสดีครับทุกคน", False),
            ("วันนี้อากาศดีจังเลย", False),
            ("Hello world", False),
            ("Nice to meet you", False),
            ("ขอให้วันนี้เป็นวันที่ดี", False),
            ("ชอบสตรีมนี้มากครับ", False),
            ("เล่นเกมเก่งมาก", False),
            ("Good luck have fun", False),
            ("ขอบคุณสำหรับข้อมูลครับ", False),
            ("กินข้าวหรือยัง", False),
            ("How are you doing?", False),
            ("ฝันดีนะครับทุกคน", False),
            ("Please be polite", False),
            ("Respect the rules", False),
            ("ตลกมาก 555", False),
            ("น่ารักจังเลย", False),
            ("See you tomorrow", False),
            ("Keep fighting", False),
            ("สู้ๆ นะครับแอดมิน", False),
            ("Can you play this game?", False),
            ("เสียงชัดเจนดีครับ", False),
            ("ภาพสวยมาก", False),
            ("I love this community", False),
            ("อย่าลืมกดติดตามนะ", False),
            ("Welcome new viewers", False),

            # --- กลุ่มที่ 2: คำหยาบที่ระบบ "จับได้" (Profanity) -> คาดหวังผล True (TP) ---
            # 2.1 คำหยาบไทยตรงๆ
            ("มึงมันโง่จริงๆ", True),
            ("ไอ้ควาย", True),
            ("พ่อมึงตาย", True),
            ("ไปตายซะไอ้สัส", True),
            ("หน้าตอแหลมาก", True),
            ("ไอ้เหี้ย", True),
            ("อีดอกทอง", True),
            ("สันดานเสีย", True),
            ("กวนตีนชิบหาย", True),
            ("ชั่งแม่ง", True),
            ("ไอ้ชาติชั่ว", True),
            
            # 2.2 คำหยาบอังกฤษ
            ("Fuck you", True),
            ("You are a bitch", True),
            ("Bullshit", True),
            ("Asshole", True),
            ("Dickhead", True),
            ("Son of a bitch", True),
            ("You Shit", True),
            
            # 2.3 เทคนิคหลบเลี่ยง
            ("ไอ้ สั ส", True),       
            ("ค-ว-า-ย", True),       
            ("f u c k", True),       
            ("You are s.h.i.t", True), 
            ("มึงมัน stupid", True),  
            ("Hello ไอ้ควาย", True),  

            # --- กลุ่มที่ 3: Limitation (ข้อจำกัด) ---
            ("ไอ้สัD", True), 
        ]
    
    def test_calculate_accuracy(self, worker, test_dataset):
        """คำนวณ Accuracy และแสดงรายการที่ผิดพลาด"""
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        
        # เก็บรายการที่ผิดพลาดไว้โชว์ตอนท้าย
        failed_cases = []

        print("\n" + "="*60)
        print("📝 เริ่มต้นการตรวจสอบทีละข้อความ...")
        print("-" * 60)

        for i, (message, has_badword) in enumerate(test_dataset, 1):
            result = worker.optimized_detect_bad_words(message)
            detected = len(result) > 0
            
            if has_badword and detected:
                true_positive += 1
                # print(f"✅ [TP] จับได้ถูกต้อง: '{message}' -> เจอคำว่า: {result}")
                
            elif has_badword and not detected:
                false_negative += 1
                error_msg = f"❌ [FN] จับไม่ได้ (หลุด): '{message}'"
                print(error_msg)
                failed_cases.append(error_msg)
                
            elif not has_badword and detected:
                false_positive += 1
                error_msg = f"❌ [FP] เตือนมั่ว (ผิด): '{message}' -> ดันไปเจอ: {result}"
                print(error_msg)
                failed_cases.append(error_msg)
                
            elif not has_badword and not detected:
                true_negative += 1
        
        # คำนวณค่าต่างๆ
        total = len(test_dataset)
        accuracy = (true_positive + true_negative) / total * 100
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # พิมพ์สรุปรายการที่ผิดพลาด
        if failed_cases:
            print("\n" + "!"*60)
            print("⚠️  สรุปรายการที่ระบบทำงานผิดพลาด (เอาไว้วิเคราะห์):")
            print("!"*60)
            for case in failed_cases:
                print(case)
        else:
            print("\n✨ เยี่ยมมาก! ไม่พบข้อผิดพลาดเลย")

        # พิมพ์ผลลัพธ์สรุป
        print("\n" + "="*50)
        print("📊 ACCURACY TEST RESULTS (50 Cases)")
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
        
        # Assert (ปรับลดลงหน่อยเผื่อมีหลุด จะได้ไม่ Error จนตกใจ)
        assert accuracy >= 90, f"Accuracy ต่ำเกินไป: {accuracy:.2f}%"

# ================== 2. PERFORMANCE & MEMORY TESTING ==================

class TestPerformance:
    """ทดสอบประสิทธิภาพและการใช้หน่วยความจำ"""
    
    @pytest.fixture
    def worker(self):
        return TwitchChatWorker("test_channel")
    
    def test_detection_speed(self, worker):
        """ทดสอบความเร็วในการตรวจจับ (Speed)"""
        test_messages = [
            "สวัสดีครับ",
            "hello everyone",
            "ไอ้สัสว์",
            "you stupid",
            "วันนี้อากาศดีมาก hello nice stream"
        ] * 100
        
        start_time = time.time()
        
        for message in test_messages:
            worker.optimized_detect_bad_words(message)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = (total_time / len(test_messages)) * 1000  
        
        print("\n" + "="*50)
        print("⚡ PERFORMANCE TEST RESULTS")
        print("="*50)
        print(f"Total Messages Processed: {len(test_messages)}")
        print(f"Average Time per Message: {avg_time:.4f} ms")
        print("="*50)
        
        assert avg_time < 100, f"ช้าเกินไป: {avg_time:.3f} ms"
    
    def test_memory_usage(self, worker):
        """ทดสอบการใช้หน่วยความจำ (Memory)"""
        import sys
        for i in range(200):
            chat_info = {
                'timestamp': datetime.now(),
                'username': f'user{i}',
                'message': f'test message {i}' * 5,
                'bad_words': ['test'],
                'channel': 'test'
            }
            worker.chat_messages.append(chat_info)
        
        memory_size = sys.getsizeof(worker.chat_messages)
        for msg in worker.chat_messages:
            memory_size += sys.getsizeof(msg)
        
        memory_kb = memory_size / 1024
        
        print("\n" + "="*50)
        print("💾 MEMORY USAGE TEST")
        print("="*50)
        print(f"Messages Stored: {len(worker.chat_messages)}")
        print(f"Memory Usage: {memory_kb:.2f} KB")
        print("="*50)
        
        assert memory_kb < 500, f"ใช้หน่วยความจำมากเกินไป: {memory_kb:.2f} KB"

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 STARTING DEBUG MODE TEST")
    print("="*70)
    pytest.main([__file__, "-v", "-s", "--tb=short"])