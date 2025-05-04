# Set up the hostfile.txt
i.e.:
10.65.0.2 slots=8
10.65.0.3 slots=8
10.65.0.4 slots=8

export HF_TOKEN=PUT_TOKEN_HERE # Must have access to Llama 3.3 70B
pip install -r requirements.txt
apt update
apt install pdsh
deepspeed --hostfile=hostfile.txt  train_llama.py
