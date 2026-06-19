#!/usr/bin/env python3
import struct
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class GGUFTorchLoader:
    """PyTorch 风格的 GGUF 权重加载器，直接从 GGUF 元数据读取偏移量"""

    # GGUF scalar value type → numpy dtype/byte size
    _SCALAR_FMT = {
        0:  ('<u1', 1),  # UINT8
        1:  ('<i1', 1),  # INT8
        2:  ('<u2', 2),  # UINT16
        3:  ('<i2', 2),  # INT16
        4:  ('<u4', 4),  # UINT32
        5:  ('<i4', 4),  # INT32
        6:  ('<f4', 4),  # FLOAT32
        7:  ('<?', 1),   # BOOL
        10: ('<u8', 8),  # UINT64
        11: ('<i8', 8),  # INT64
        12: ('<f8', 8),  # FLOAT64
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tensor_meta: Dict[str, dict] = {}
        self.metadata: Dict[str, object] = {}
        self._parse_gguf()

    def _read_value(self, f, vt: int):
        """Read a single GGUF metadata value of the given type."""
        if vt == 8:  # string
            sl = struct.unpack('<Q', f.read(8))[0]
            return f.read(sl).decode('utf-8', errors='replace')
        if vt == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        if vt == 13:  # BF16
            raw = f.read(2)
            return (np.frombuffer(raw, dtype='<u2')[0].astype(np.uint32) << 16).view(np.float32).item()
        fmt, size = self._SCALAR_FMT[vt]
        return np.frombuffer(f.read(size), dtype=fmt)[0].item()

    def _parse_gguf(self):
        """直接解析 GGUF 文件，获取每个 tensor 的真实 offset/dtype/dims"""
        with open(self.filepath, 'rb') as f:
            magic = f.read(4)
            assert magic == b'GGUF', f"Not a GGUF file: {magic}"
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            meta_count = struct.unpack('<Q', f.read(8))[0]

            # 解析 metadata key-value pairs
            for _ in range(meta_count):
                kl = struct.unpack('<Q', f.read(8))[0]
                key = f.read(kl).decode('utf-8', errors='replace')
                vt = struct.unpack('<I', f.read(4))[0]
                self.metadata[key] = self._read_value(f, vt)

            # 解析 tensor info section，先收集原始信息
            dtype_size_map = {0: 4, 1: 2, 30: 2}
            dtype_name_map = {0: "F32", 1: "F16", 30: "BF16"}
            raw_tensors = []
            for _ in range(tensor_count):
                nl = struct.unpack('<Q', f.read(8))[0]
                name = f.read(nl).decode('utf-8')
                nd = struct.unpack('<I', f.read(4))[0]
                dims_raw = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
                gguf_dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]

                # GGUF stores dims from innermost to outermost, reverse to get PyTorch order
                dims = list(reversed(dims_raw))

                elem_size = dtype_size_map.get(gguf_dtype, 4)
                dtype_name = dtype_name_map.get(gguf_dtype, f"UNKNOWN_{gguf_dtype}")

                n_elem = 1
                for d in dims: n_elem *= d
                size = n_elem * elem_size

                raw_tensors.append((name, dims, offset, dtype_name, size, gguf_dtype))

            # 记录 tensor info section 结束位置，对齐到 32 字节边界为 data_offset
            raw_offset = f.tell()
            alignment = 32
            data_offset = raw_offset + (alignment - raw_offset % alignment) % alignment

            for name, dims, offset, dtype_name, size, gguf_dtype in raw_tensors:
                self.tensor_meta[name] = {
                    "dtype": dtype_name,
                    "shape": dims,
                    "offset": data_offset + offset,  # tensor offset 是相对于 data_offset 的
                    "size": size,
                    "gguf_dtype": gguf_dtype,
                }

        print(f"索引构建完成 | 共 {len(self.tensor_meta)} 个张量")

    def load_tensor(self, name: str, transpose: bool = False,dtype: torch.dtype = torch.float32, device: Optional[str] = None) -> torch.Tensor:
        """加载单个张量，返回 torch.Tensor"""
        if name not in self.tensor_meta:
            raise KeyError(f"未找到张量: {name}")

        meta = self.tensor_meta[name]
        with open(self.filepath, 'rb') as f:
            f.seek(meta["offset"])
            raw = f.read(meta["size"])

        if meta["dtype"] == "BF16":
            arr_u16 = np.frombuffer(raw, dtype='<u2').copy()
            arr_f32 = (arr_u16.astype(np.uint32) << 16).view(np.float32)
            tensor = torch.from_numpy(arr_f32).float()
        elif meta["dtype"] == "F32":
            tensor = torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy()).float()
        elif meta["dtype"] == "F16":
            arr_u16 = np.frombuffer(raw, dtype='<u2').copy()
            arr_f32 = np.frombuffer(arr_u16.astype('<u2').tobytes(), dtype='<f2').astype(np.float32)
            tensor = torch.from_numpy(arr_f32).float()
        else:
            raise ValueError(f"Unsupported dtype: {meta['dtype']}")

        tensor = tensor.reshape(meta["shape"])
        if transpose:
            tensor = tensor.T.contiguous()

        if dtype != torch.float32:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.load_tensor(name)

    def state_dict(self, transpose_map: Optional[Dict[str, bool]] = None,
                   device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if transpose_map is None:
            transpose_map = {}
        return {
            name: self.load_tensor(name, transpose=transpose_map.get(name, False), device=device)
            for name in self.tensor_meta
        }

    @staticmethod
    def verify(cpp_data: Union[np.ndarray, torch.Tensor, List],
               torch_tensor: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
        if not isinstance(cpp_data, torch.Tensor):
            cpp_tensor = torch.tensor(cpp_data, dtype=torch.float32)
        else:
            cpp_tensor = cpp_data.float()

        is_close = torch.allclose(cpp_tensor, torch_tensor.float(), atol=atol, rtol=rtol)

        if is_close:
            print(f"验证通过: 形状 {list(torch_tensor.shape)}, 最大差异 < {atol}")
        else:
            diff = (cpp_tensor - torch_tensor.float()).abs()
            print(f"验证失败: 最大差异 {diff.max().item():.6e}, 平均差异 {diff.mean().item():.6e}")
            idx = torch.where(diff > atol)[0][:3]
            if idx.numel() > 0:
                for i in idx:
                    print(f"   位置 {i.item()}: C++={cpp_tensor.view(-1)[i].item():.6f} | Torch={torch_tensor.view(-1)[i].item():.6f}")
        return is_close
