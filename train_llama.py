import os
import resource
import torch
import time
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from trl import SFTTrainer
from datasets import load_dataset

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Increase file descriptor limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

# Network configuration
os.environ["NCCL_SOCKET_IFNAME"] = "eth1,eth4,eth5,eth6,eth7,eth8"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_NTHREADS"] = "1"
os.environ["NCCL_NSOCKS_PERTHREAD"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # New recommended version
os.environ["NCCL_SHM_DISABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "0"

# Get distributed info
local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", 1))
global_rank = int(os.environ.get("NODE_RANK", 0))
num_nodes = int(os.environ.get("NUM_NODES", 1))
gpus_per_node = int(os.environ.get("NUM_TRAINERS", 1))

# Initialize distributed before model loading
if local_rank != -1:
    if not torch.distributed.is_initialized():
        logger.info(f"Initializing process group with rank {global_rank}")
        torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

# Add custom callback for step timing
class TimingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.last_step_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        step_time = now - self.last_step_time
        if global_rank == 0:  # Only log on main process
            logger.info(f"Step {state.global_step} completed in {step_time:.2f} seconds")
        self.last_step_time = now
        
    def on_train_begin(self, args, state, control, **kwargs):
        if global_rank == 0:
            logger.info(f"=== Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        self.start_time = time.time()
        self.last_step_time = time.time()

if global_rank == 0:
    logger.info(f"Starting distributed training with:")
    logger.info(f"  Number of nodes: {num_nodes}")
    logger.info(f"  GPUs per node: {gpus_per_node}")
    logger.info(f"  Total GPUs: {world_size}")
    logger.info(f"  Master: {os.environ.get('MASTER_ADDR', 'unknown')}:{os.environ.get('MASTER_PORT', 'unknown')}")
    logger.info(f"  Network settings: NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME')}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", model_max_length=4096)
if tokenizer.pad_token is None:
    raise ValueError

micro_batch = 1
grad_acc = 8
total_batch = micro_batch * grad_acc * world_size

# DeepSpeed ZeRO-3 config
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
    },
    "bf16": {"enabled": True},
    "gradient_clipping": 1.0,
    "train_batch_size": total_batch,
    "train_micro_batch_size_per_gpu": micro_batch,
    "gradient_accumulation_steps": grad_acc,
    "zero_allow_untested_optimizer": True,
    "fp16": {"enabled": False},
    "communication_data_type": "bf16"
}

# Load dataset
if global_rank == 0:
    logger.info(f"Loading dataset on master node")
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")
if global_rank == 0:
    logger.info(f"Dataset loaded with {len(dataset)} examples")

# Training arguments
training_args = TrainingArguments(
    output_dir="/workspace/output",
    learning_rate=5e-6,
    per_device_train_batch_size=micro_batch,
    gradient_accumulation_steps=grad_acc,
    bf16=True,
    deepspeed=ds_config,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=300,
    save_total_limit=2,
    gradient_checkpointing=True,
    local_rank=local_rank,
    ddp_find_unused_parameters=False,
    tf32=True,
    dataloader_num_workers=4
)

# Load model
if global_rank == 0:
    logger.info(f"Loading model in CPU memory")
model_load_start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.3-70B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
if global_rank == 0:
    logger.info(f"Model loaded in {time.time() - model_load_start:.2f} seconds")

# Create the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

timer_callback = TimingCallback()
trainer.add_callback(timer_callback)

if global_rank == 0:
    logger.info("=== Starting training ===")
    start_time = time.time()

trainer.train()

if global_rank == 0:
    total_time = time.time() - start_time
    logger.info(f"=== Training completed in {total_time:.2f} seconds ===")