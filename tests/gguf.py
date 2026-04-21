#!/usr/bin/env python3
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class GGUFTorchLoader:
    """PyTorch 风格的 GGUF 权重加载器，专为 C++/Python 交叉验证设计"""
    
    def __init__(self, filepath: str, base_offset: int, tensor_list: List[Tuple[str, str, List[int]]]):
        self.filepath = filepath
        self.tensor_meta: Dict[str, dict] = {}
        self._build_index(tensor_list, base_offset)

    def _build_index(self, tensor_list: List[Tuple[str, str, List[int]]], base_offset: int):
        curr = base_offset
        for name, dtype_str, shape in tensor_list:
            elem_size = 2 if dtype_str == "BF16" else 4
            size = int(np.prod(shape)) * elem_size
            self.tensor_meta[name] = {"dtype": dtype_str, "shape": shape, "offset": curr, "size": size}
            curr += size
        print(f"✅ 索引构建完成 | 共 {len(tensor_list)} 个张量 | 数据区结束偏移: 0x{curr:x}")

    def load_tensor(self, name: str, transpose: bool = False, 
                    dtype: torch.dtype = torch.float32, device: Optional[str] = None) -> torch.Tensor:
        """加载单个张量，返回 torch.Tensor"""
        if name not in self.tensor_meta:
            raise KeyError(f"❌ 未找到张量: {name}")
            
        meta = self.tensor_meta[name]
        with open(self.filepath, 'rb') as f:
            f.seek(meta["offset"])
            raw = f.read(meta["size"])

        # 强制小端序读取，避免跨平台字节序问题
        if meta["dtype"] == "BF16":
            # BF16 -> FP32: uint16 << 16 -> float32 (标准无损转换)
            arr_u16 = np.frombuffer(raw, dtype='<u2').copy()
            arr_f32 = (arr_u16.astype(np.uint32) << 16).view(np.float32)
            tensor = torch.from_numpy(arr_f32)
        else:
            tensor = torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy())

        # 重塑 & 转置
        tensor = tensor.reshape(meta["shape"])
        if transpose:
            tensor = tensor.T.contiguous()  # PyTorch 的 T 是视图，需 contiguous 保证内存连续

        # 转换目标 dtype 和设备
        if dtype != torch.float32:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
            
        return tensor

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.load_tensor(name)

    def state_dict(self, transpose_map: Optional[Dict[str, bool]] = None, 
                   device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """返回标准 PyTorch state_dict 格式"""
        if transpose_map is None: transpose_map = {}
        return {
            name: self.load_tensor(name, transpose=transpose_map.get(name, False), device=device)
            for name in self.tensor_meta
        }

    @staticmethod
    def verify(cpp_data: Union[np.ndarray, torch.Tensor, List], 
               torch_tensor: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
        """验证 C++ 输出与 PyTorch 加载的权重是否一致（自动处理 dtype/shape）"""
        if not isinstance(cpp_data, torch.Tensor):
            cpp_tensor = torch.tensor(cpp_data, dtype=torch.float32)
        else:
            cpp_tensor = cpp_data.float()
            
        # 统一转为 float32 比较，避免 bfloat16 精度截断导致误报
        is_close = torch.allclose(cpp_tensor, torch_tensor.float(), atol=atol, rtol=rtol)
        
        if is_close:
            print(f"✅ 验证通过: 形状 {list(torch_tensor.shape)}, 最大差异 < {atol}")
        else:
            diff = (cpp_tensor - torch_tensor.float()).abs()
            print(f"⚠️ 验证失败: 最大差异 {diff.max().item():.6e}, 平均差异 {diff.mean().item():.6e}")
            # 打印前 3 个差异点供调试
            idx = torch.where(diff > atol)[0][:3]
            if idx.numel() > 0:
                for i in idx:
                    print(f"   位置 {i.item()}: C++={cpp_tensor.view(-1)[i].item():.6f} | Torch={torch_tensor.view(-1)[i].item():.6f}")
        return is_close