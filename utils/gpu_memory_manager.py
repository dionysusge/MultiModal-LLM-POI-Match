"""
GPU内存监控和自动切换管理器

功能:
1. 实时监控GPU内存使用情况
2. 当GPU内存不足时自动切换到CPU
3. 定期检查GPU内存状态，条件允许时切回GPU
4. 提供内存清理和优化功能

作者: Dionysus
"""

import torch
import psutil
import time
import threading
import logging
from typing import Optional, Callable, Dict, Any
import warnings
from contextlib import contextmanager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """
    GPU内存管理器
    
    功能:
        1. 监控GPU内存使用情况
        2. 自动设备切换（GPU <-> CPU）
        3. 内存清理和优化
        4. 异常处理和恢复
    """
    
    def __init__(self, 
                 memory_threshold: float = 0.85,
                 check_interval: int = 30,
                 auto_switch: bool = True,
                 enable_monitoring: bool = True):
        """
        初始化GPU内存管理器
        
        参数:
            memory_threshold: GPU内存使用阈值（0-1），超过此值时切换到CPU
            check_interval: 检查间隔（秒）
            auto_switch: 是否启用自动设备切换
            enable_monitoring: 是否启用后台监控
        """
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.auto_switch = auto_switch
        self.enable_monitoring = enable_monitoring
        
        # 设备状态
        self.current_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.preferred_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_switch_count = 0
        self.last_switch_time = time.time()
        
        # 监控状态
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.memory_stats = {
            'max_memory_allocated': 0,
            'max_memory_reserved': 0,
            'switch_to_cpu_count': 0,
            'switch_to_gpu_count': 0,
            'oom_errors': 0
        }
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化管理器"""
        if torch.cuda.is_available():
            # 清理GPU内存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 获取GPU信息
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU内存管理器初始化完成")
            logger.info(f"GPU总内存: {self.gpu_total_memory / 1024**3:.2f}GB")
            logger.info(f"内存阈值: {self.memory_threshold * 100:.1f}%")
            logger.info(f"检查间隔: {self.check_interval}秒")
            
            # 启动监控线程
            if self.enable_monitoring:
                self.start_monitoring()
        else:
            logger.warning("CUDA不可用，GPU内存管理器将以CPU模式运行")
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """
        获取GPU内存信息
        
        返回:
            dict: 包含内存使用信息的字典
        """
        if not torch.cuda.is_available():
            return {
                'total_gb': 0,
                'allocated_gb': 0,
                'reserved_gb': 0,
                'free_gb': 0,
                'usage_percent': 0
            }
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)
            free_memory = total_memory - reserved_memory
            
            return {
                'total_gb': total_memory / 1024**3,
                'allocated_gb': allocated_memory / 1024**3,
                'reserved_gb': reserved_memory / 1024**3,
                'free_gb': free_memory / 1024**3,
                'usage_percent': reserved_memory / total_memory
            }
        except Exception as e:
            logger.error(f"获取GPU内存信息失败: {e}")
            return {
                'total_gb': 0,
                'allocated_gb': 0,
                'reserved_gb': 0,
                'free_gb': 0,
                'usage_percent': 1.0  # 假设内存已满
            }
    
    def check_memory_status(self) -> bool:
        """
        检查内存状态
        
        返回:
            bool: True表示内存充足，False表示内存不足
        """
        if not torch.cuda.is_available():
            return False
        
        memory_info = self.get_gpu_memory_info()
        return memory_info['usage_percent'] < self.memory_threshold
    
    def cleanup_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU内存清理完成")
    
    def switch_to_cpu(self, reason: str = "内存不足"):
        """
        切换到CPU设备
        
        参数:
            reason: 切换原因
        """
        if self.current_device != 'cpu':
            self.current_device = 'cpu'
            self.device_switch_count += 1
            self.last_switch_time = time.time()
            self.memory_stats['switch_to_cpu_count'] += 1
            
            # 清理GPU内存
            self.cleanup_gpu_memory()
            
            logger.warning(f"设备切换到CPU - 原因: {reason}")
            logger.info(f"累计设备切换次数: {self.device_switch_count}")
    
    def switch_to_gpu(self, reason: str = "内存充足"):
        """
        切换到GPU设备
        
        参数:
            reason: 切换原因
        """
        if torch.cuda.is_available() and self.current_device != 'cuda':
            # 检查内存状态
            if self.check_memory_status():
                self.current_device = 'cuda'
                self.device_switch_count += 1
                self.last_switch_time = time.time()
                self.memory_stats['switch_to_gpu_count'] += 1
                
                logger.info(f"设备切换到GPU - 原因: {reason}")
                logger.info(f"累计设备切换次数: {self.device_switch_count}")
            else:
                logger.warning("GPU内存仍然不足，无法切换到GPU")
    
    def get_optimal_device(self) -> str:
        """
        获取当前最优设备
        
        返回:
            str: 设备名称 ('cuda' 或 'cpu')
        """
        if not self.auto_switch:
            return self.current_device
        
        if not torch.cuda.is_available():
            return 'cpu'
        
        # 检查内存状态
        if self.check_memory_status():
            if self.current_device == 'cpu' and self.preferred_device == 'cuda':
                # 如果当前在CPU但GPU内存充足，切换到GPU
                self.switch_to_gpu("内存状态良好")
        else:
            if self.current_device == 'cuda':
                # 如果当前在GPU但内存不足，切换到CPU
                self.switch_to_cpu("内存使用率过高")
        
        return self.current_device
    
    @contextmanager
    def safe_gpu_operation(self):
        """
        安全的GPU操作上下文管理器
        
        用法:
            with memory_manager.safe_gpu_operation():
                # GPU操作代码
                pass
        """
        original_device = self.current_device
        try:
            # 获取最优设备
            optimal_device = self.get_optimal_device()
            yield optimal_device
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU内存不足错误: {e}")
                self.memory_stats['oom_errors'] += 1
                self.switch_to_cpu("OOM错误")
                # 重新尝试使用CPU
                yield 'cpu'
            else:
                raise e
        except Exception as e:
            logger.error(f"GPU操作异常: {e}")
            raise e
        finally:
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def monitor_memory(self):
        """后台内存监控线程"""
        logger.info("GPU内存监控线程启动")
        
        while not self.stop_monitoring:
            try:
                if torch.cuda.is_available():
                    memory_info = self.get_gpu_memory_info()
                    
                    # 更新统计信息
                    self.memory_stats['max_memory_allocated'] = max(
                        self.memory_stats['max_memory_allocated'],
                        memory_info['allocated_gb']
                    )
                    self.memory_stats['max_memory_reserved'] = max(
                        self.memory_stats['max_memory_reserved'],
                        memory_info['reserved_gb']
                    )
                    
                    # 检查是否需要设备切换
                    if self.auto_switch:
                        self.get_optimal_device()
                    
                    # 记录内存状态（每5分钟记录一次）
                    if int(time.time()) % 300 == 0:
                        logger.info(f"GPU内存状态: "
                                  f"已用 {memory_info['allocated_gb']:.2f}GB/"
                                  f"{memory_info['total_gb']:.2f}GB "
                                  f"({memory_info['usage_percent']*100:.1f}%)")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"内存监控异常: {e}")
                time.sleep(self.check_interval)
        
        logger.info("GPU内存监控线程停止")
    
    def start_monitoring(self):
        """启动后台监控"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring = False
            self.monitoring_thread = threading.Thread(target=self.monitor_memory, daemon=True)
            self.monitoring_thread.start()
            logger.info("GPU内存监控已启动")
    
    def stop_monitoring_thread(self):
        """停止后台监控"""
        self.stop_monitoring = True
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("GPU内存监控已停止")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        返回:
            dict: 统计信息
        """
        memory_info = self.get_gpu_memory_info()
        
        return {
            'current_device': self.current_device,
            'device_switch_count': self.device_switch_count,
            'memory_threshold': self.memory_threshold,
            'current_memory_usage': memory_info['usage_percent'],
            'current_memory_gb': memory_info['allocated_gb'],
            'total_memory_gb': memory_info['total_gb'],
            'memory_stats': self.memory_stats.copy(),
            'last_switch_time': self.last_switch_time
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("GPU内存管理器统计信息")
        print("="*50)
        print(f"当前设备: {stats['current_device']}")
        print(f"设备切换次数: {stats['device_switch_count']}")
        print(f"内存阈值: {stats['memory_threshold']*100:.1f}%")
        print(f"当前内存使用: {stats['current_memory_usage']*100:.1f}% "
              f"({stats['current_memory_gb']:.2f}GB/{stats['total_memory_gb']:.2f}GB)")
        print(f"切换到CPU次数: {stats['memory_stats']['switch_to_cpu_count']}")
        print(f"切换到GPU次数: {stats['memory_stats']['switch_to_gpu_count']}")
        print(f"OOM错误次数: {stats['memory_stats']['oom_errors']}")
        print(f"最大内存分配: {stats['memory_stats']['max_memory_allocated']:.2f}GB")
        print(f"最大内存保留: {stats['memory_stats']['max_memory_reserved']:.2f}GB")
        print("="*50)
    
    def __del__(self):
        """析构函数，停止监控线程"""
        self.stop_monitoring_thread()


# 全局内存管理器实例
_global_memory_manager: Optional[GPUMemoryManager] = None


def get_memory_manager(memory_threshold: float = 0.85,
                      check_interval: int = 30,
                      auto_switch: bool = True,
                      enable_monitoring: bool = True) -> GPUMemoryManager:
    """
    获取全局内存管理器实例
    
    参数:
        memory_threshold: GPU内存使用阈值
        check_interval: 检查间隔（秒）
        auto_switch: 是否启用自动设备切换
        enable_monitoring: 是否启用后台监控
        
    返回:
        GPUMemoryManager: 内存管理器实例
    """
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = GPUMemoryManager(
            memory_threshold=memory_threshold,
            check_interval=check_interval,
            auto_switch=auto_switch,
            enable_monitoring=enable_monitoring
        )
    
    return _global_memory_manager


def safe_cuda_operation(func: Callable) -> Callable:
    """
    安全CUDA操作装饰器
    
    参数:
        func: 要装饰的函数
        
    返回:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        memory_manager = get_memory_manager()
        
        with memory_manager.safe_gpu_operation() as device:
            # 更新kwargs中的device参数
            if 'device' in kwargs:
                kwargs['device'] = device
            
            return func(*args, **kwargs)
    
    return wrapper


def get_optimal_device() -> str:
    """
    获取当前最优设备
    
    返回:
        str: 设备名称
    """
    memory_manager = get_memory_manager()
    return memory_manager.get_optimal_device()


def cleanup_gpu_memory():
    """清理GPU内存"""
    memory_manager = get_memory_manager()
    memory_manager.cleanup_gpu_memory()


def print_memory_stats():
    """打印内存统计信息"""
    memory_manager = get_memory_manager()
    memory_manager.print_statistics()


if __name__ == "__main__":
    # 测试代码
    print("GPU内存管理器测试")
    
    # 创建内存管理器
    manager = GPUMemoryManager(
        memory_threshold=0.8,
        check_interval=5,
        auto_switch=True,
        enable_monitoring=True
    )
    
    # 测试内存信息获取
    memory_info = manager.get_gpu_memory_info()
    print(f"GPU内存信息: {memory_info}")
    
    # 测试安全操作
    with manager.safe_gpu_operation() as device:
        print(f"当前使用设备: {device}")
        
        # 模拟一些GPU操作
        if device == 'cuda':
            x = torch.randn(1000, 1000).to(device)
            y = torch.mm(x, x.t())
            print(f"GPU操作完成，结果形状: {y.shape}")
    
    # 打印统计信息
    manager.print_statistics()
    
    # 等待一段时间观察监控
    print("等待10秒观察监控...")
    time.sleep(10)
    
    # 停止监控
    manager.stop_monitoring_thread()
    print("测试完成")