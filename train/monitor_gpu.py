"""
GPU显存监控工具 - 针对RTX 4070 8GB优化
实时监控显存使用，防止OOM
"""
import torch
import psutil
import os
from datetime import datetime


class GPUMonitor:
    """GPU和系统资源监控器"""
    
    def __init__(self, threshold_gb=7.5):
        """
        初始化监控器
        
        Args:
            threshold_gb: 显存警告阈值(GB)，默认7.5GB
        """
        self.threshold_gb = threshold_gb
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.device_name = torch.cuda.get_device_name(0)
            self.total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU检测成功: {self.device_name}")
            print(f"✅ 显存总量: {self.total_memory_gb:.2f} GB")
        else:
            print("⚠️  警告: 未检测到CUDA设备，将使用CPU训练")
    
    def get_gpu_memory(self):
        """获取GPU显存使用情况"""
        if not self.cuda_available:
            return None
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = self.total_memory_gb - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": self.total_memory_gb,
            "percent": (reserved / self.total_memory_gb) * 100
        }
    
    def get_system_memory(self):
        """获取系统内存使用情况"""
        ram = psutil.virtual_memory()
        return {
            "used": ram.used / 1024**3,
            "total": ram.total / 1024**3,
            "percent": ram.percent
        }
    
    def check_and_report(self, step=None, force_gc=False):
        """
        检查资源使用并报告
        
        Args:
            step: 当前训练步数
            force_gc: 是否强制执行垃圾回收
            
        Returns:
            bool: 是否超过警告阈值
        """
        gpu_mem = self.get_gpu_memory()
        sys_mem = self.get_system_memory()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        step_info = f"Step {step}" if step is not None else "Current"
        
        print(f"\n{'='*60}")
        print(f"[{timestamp}] {step_info} - 资源监控")
        print(f"{'='*60}")
        
        if gpu_mem:
            print(f"🖥️  GPU显存:")
            print(f"   - 已分配: {gpu_mem['allocated']:.2f} GB")
            print(f"   - 已保留: {gpu_mem['reserved']:.2f} GB")
            print(f"   - 空闲:   {gpu_mem['free']:.2f} GB")
            print(f"   - 使用率: {gpu_mem['percent']:.1f}%")
            
            # 警告检查
            if gpu_mem['reserved'] > self.threshold_gb:
                print(f"⚠️  警告: 显存使用 {gpu_mem['reserved']:.2f} GB 超过阈值 {self.threshold_gb} GB!")
                print(f"   建议: 减小batch_size或启用更多优化选项")
                return True
        
        print(f"💾 系统内存:")
        print(f"   - 已使用: {sys_mem['used']:.2f} GB / {sys_mem['total']:.2f} GB")
        print(f"   - 使用率: {sys_mem['percent']:.1f}%")
        
        if force_gc or (gpu_mem and gpu_mem['reserved'] > self.threshold_gb * 0.9):
            self.clear_cache()
        
        return False
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            print("🧹 已清理GPU缓存")
    
    def get_recommended_batch_size(self):
        """根据显存情况推荐batch size"""
        if not self.cuda_available:
            return 1
        
        gpu_mem = self.get_gpu_memory()
        free_gb = gpu_mem['free']
        
        if free_gb > 6:
            return 4
        elif free_gb > 4:
            return 2
        else:
            return 1


def test_gpu_setup():
    """测试GPU配置"""
    print("\n" + "="*60)
    print("GPU配置测试")
    print("="*60)
    
    monitor = GPUMonitor()
    
    if not monitor.cuda_available:
        print("❌ 错误: 未检测到CUDA")
        return False
    
    print(f"\n📊 CUDA信息:")
    print(f"   - CUDA版本: {torch.version.cuda}")
    print(f"   - PyTorch版本: {torch.__version__}")
    print(f"   - cuDNN版本: {torch.backends.cudnn.version()}")
    
    # 测试小模型加载
    try:
        print(f"\n🧪 测试加载GPT-2 Small...")
        from transformers import GPT2LMHeadModel
        
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = model.to("cuda")
        
        monitor.check_and_report()
        
        del model
        torch.cuda.empty_cache()
        
        print(f"\n✅ GPU配置测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ GPU测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_gpu_setup()
    
    if success:
        print("\n" + "="*60)
        print("建议的训练配置:")
        print("="*60)
        monitor = GPUMonitor()
        recommended_bs = monitor.get_recommended_batch_size()
        print(f"✅ 推荐 batch_size: {recommended_bs}")
        print(f"✅ 推荐 gradient_accumulation_steps: {32 // recommended_bs}")
        print(f"✅ 必须启用: fp16=True, gradient_checkpointing=True")
        print(f"✅ 必须使用: optim='adamw_bnb_8bit'")
