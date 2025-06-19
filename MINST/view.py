import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from mian import Net

import matplotlib
import platform
if platform.system() == 'Darwin':  # macOS
    # 优先使用 Arial Unicode MS，这是Mac系统自带的支持中文的字体
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'SimHei']
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']


class DigitRecognizer:
    def __init__(self, model_path='./mnist_cnn.pt'):
        self.model = Net()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.drawing = False
        self.canvas = np.zeros((280, 280))  # 10倍大小，便于绘制
        
    def predict_digit(self, image_array):
        """预测数字"""
        # 调整大小到28x28并标准化
        img = Image.fromarray(image_array).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        
        # 标准化（与训练时相同）
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor - 0.1307) / 0.3081
        
        # 预测
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_digit = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_digit].item()
            
        return predicted_digit, confidence, probabilities[0]
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.draw_point(event.xdata, event.ydata)
    
    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        self.draw_point(event.xdata, event.ydata)
    
    def on_release(self, event):
        self.drawing = False
        # 实时预测
        if np.any(self.canvas > 0):
            digit, confidence, probs = self.predict_digit(self.canvas)
            self.update_prediction(digit, confidence, probs)
    
    def draw_point(self, x, y):
        if x is None or y is None:
            return
        # 画粗一点的线
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                if 0 <= int(x) + dx < 280 and 0 <= int(y) + dy < 280:
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance <= 8:
                        intensity = max(0, 255 * (1 - distance/8))
                        self.canvas[int(y) + dy, int(x) + dx] = min(255, 
                            self.canvas[int(y) + dy, int(x) + dx] + intensity)
        
        self.im.set_array(self.canvas)
        self.fig.canvas.draw_idle()
    
    def clear_canvas(self, event=None):
        self.canvas.fill(0)
        self.im.set_array(self.canvas)
        self.prediction_text.set_text('预测: -\n置信度: -')
        self.prob_bars.clear()
        self.prob_bars.bar(range(10), [0]*10)
        self.prob_bars.set_ylim(0, 1)
        self.fig.canvas.draw_idle()
    
    def update_prediction(self, digit, confidence, probabilities):
        self.prediction_text.set_text(f'预测: {digit}\n置信度: {confidence:.1%}')
        
        # 更新概率条形图
        self.prob_bars.clear()
        bars = self.prob_bars.bar(range(10), probabilities.numpy(), 
                                 color=['red' if i == digit else 'blue' for i in range(10)])
        self.prob_bars.set_ylim(0, 1)
        self.prob_bars.set_xlabel('数字')
        self.prob_bars.set_ylabel('概率')
        self.prob_bars.set_title('各数字预测概率')
        
        self.fig.canvas.draw_idle()
    
    def run(self):
        # 创建界面
        self.fig, (self.ax, self.prob_ax) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 绘图区域
        self.im = self.ax.imshow(self.canvas, cmap='gray', vmin=0, vmax=255)
        self.ax.set_title('在此处绘制数字 (鼠标拖拽)')
        self.ax.set_xlim(0, 280)
        self.ax.set_ylim(280, 0)
        
        # 预测结果显示
        self.prediction_text = self.ax.text(10, 30, '预测: -\n置信度: -', 
                                          fontsize=12, color='white',
                                          bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor="black", alpha=0.7))
        
        # 概率图
        self.prob_bars = self.prob_ax
        self.prob_bars.bar(range(10), [0]*10)
        self.prob_bars.set_ylim(0, 1)
        self.prob_bars.set_xlabel('数字')
        self.prob_bars.set_ylabel('概率')
        self.prob_bars.set_title('各数字预测概率')
        
        # 添加清除按钮
        from matplotlib.widgets import Button
        ax_button = plt.axes([0.45, 0.01, 0.1, 0.05])
        button = Button(ax_button, '清除')
        button.on_clicked(self.clear_canvas)
        
        # 绑定鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    recognizer = DigitRecognizer()
    recognizer.run()