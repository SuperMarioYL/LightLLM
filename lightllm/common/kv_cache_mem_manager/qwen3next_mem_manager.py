import torch
import torch.distributed as dist
import triton
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory
from lightllm.common.linear_att_cache_manager import LinearAttCacheConfig, LinearAttCacheManager
from .operator import LinearAttMemOperator
from typing import Tuple, Any

logger = init_logger(__name__)


class Qwen3NextMemManager(MemoryManager):
    operator_class = LinearAttMemOperator

    def __init__(
        self,
        size,
        dtype,
        num_kv_heads,
        head_dim,
        full_att_layer_num,
        linear_config: LinearAttCacheConfig,
        always_copy=False,
        mem_fraction=0.9,
    ):
        self.linear_config = linear_config

        super().__init__(size, dtype, num_kv_heads, head_dim, full_att_layer_num, always_copy, mem_fraction)

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        layer_index = layer_index // self.linear_config.full_attention_interval
        return super().get_att_input_params(layer_index)

    def profile_size(self, mem_fraction):
        if self.size is not None:
            return

        torch.cuda.empty_cache()
        args = get_env_start_args()
        reserve_bytes = self.get_req_manager_reserve_bytes()
        req_state_num = (args.running_max_req_size + 8 + 1) * (args.mtp_step + 1)
        reserve_bytes += (
            req_state_num
            * self.linear_config.linear_layer_num
            * (self.linear_config.get_conv_state_bytes_per_layer() + self.linear_config.get_ssm_state_bytes_per_layer())
        )
        reserve_gb = reserve_bytes / (1024 ** 3)

        world_size = dist.get_world_size()
        available_memory = get_available_gpu_memory(world_size) - get_total_gpu_memory() * (1 - mem_fraction)
        available_memory -= reserve_gb
        cell_size = self.get_cell_size()
        self.size = max(int(available_memory * 1024 ** 3 / cell_size), 1)
        if world_size > 1:
            tensor = torch.tensor(self.size, dtype=torch.int64, device=f"cuda:{get_current_device_id()}")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            self.size = tensor.item()
        logger.info(
            f"{str(available_memory)} GB space is available after load the model weight "
            f"and reserve {reserve_gb} GB for qwen3next req_manager\n"
            f"{str(cell_size / 1024 ** 2)} MB is the size of one token kv cache\n"
            f"{self.size} is the profiled max_total_token_num with the mem_fraction {mem_fraction}\n"
        )
        return

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        # TODO 初始化线性 att 对应的部分 buffer.
        self._init_linear_att_buffers()
        return

    def _init_linear_att_buffers(self):
        big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        # 申请大页可能需要对应的资源, 多申请了两个linear att的状态，理论上这个状态
        # 永远不会被 alloc 申请到，只会在 cpu cache中，用于过渡和存储碎页情况下的
        # cpu cache 的页面拷贝。
        self.linear_att_big_page_buffers = LinearAttCacheManager(
            size=triton.cdiv(self.size, big_page_token_num) + 2,
            linear_config=self.linear_config,
            keep_num=2,
        )

        self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 2
        self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 1
        return

    def _free_buffers(self):
        super()._free_buffers()
        self._free_linear_att_buffers()
        return

    def _free_linear_att_buffers(self):
        self.linear_att_big_page_buffers = None
        return
