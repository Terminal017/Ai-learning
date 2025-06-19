import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mian import Net

class DigitImageTester:
    def __init__(self, model_path='./mnist_cnn.pt'):
        """加载训练好的模型"""
        self.model = Net()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print("手写数字识别模型加载成功！")
    
    def predict_image(self, image_path):
        """识别图片中的手写数字"""
        try:
            # 加载图片
            img = Image.open(image_path)
            print(f"图片路径: {image_path}")
            print(f"原始尺寸: {img.size}")
            
            # 转换为灰度图
            if img.mode != 'L':
                img = img.convert('L')
            
            # 调整到28x28像素
            img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            # 转换为数组并标准化
            img_array = np.array(img_resized).astype(np.float32)
            
            # 检查是否需要反转颜色（MNIST是黑底白字）
            if np.mean(img_array) > 127:
                img_array = 255 - img_array
                print("已反转颜色（适应MNIST格式）")
            
            # 标准化处理
            img_array = img_array / 255.0
            img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
            img_tensor = (img_tensor - 0.1307) / 0.3081
            
            # 模型预测
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_digit = output.argmax(dim=1).item()
                confidence = probabilities[0][predicted_digit].item()
            
            # 显示结果
            self.show_result(img, img_array, predicted_digit, confidence, probabilities[0])
            
            return predicted_digit, confidence
            
        except Exception as e:
            print(f"识别失败: {e}")
            return None, None
    
    def show_result(self, original_img, processed_img, digit, confidence, probs):
        """显示识别结果"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 原始图片
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 处理后的图片
        axes[1].imshow(processed_img, cmap='gray')
        axes[1].set_title('Processed (28x28)')
        axes[1].axis('off')
        
        # 预测概率
        bars = axes[2].bar(range(10), probs.numpy())
        axes[2].set_title(f'Prediction: {digit} ({confidence:.1%})')
        axes[2].set_xlabel('Digit')
        axes[2].set_ylabel('Probability')
        axes[2].set_xticks(range(10))
        bars[digit].set_color('red')
        
        print(f"\n识别结果:")
        print(f"预测数字: {digit}")
        print(f"置信度: {confidence:.2%}")
        
        plt.tight_layout()
        plt.show()
   

if __name__ == '__main__':
    # 测试指定图片
    tester = DigitImageTester()
    image_path = "./image/b1.png"
    
    print(f"开始识别图片: {image_path}")
    digit, confidence = tester.predict_image(image_path)
    
    if digit is not None:
        print(f"\n最终结果: 识别出数字 【{digit}】, 置信度 {confidence:.2%}")
    else:
        print("\n识别失败，请检查图片格式和路径")