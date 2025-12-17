# nano-PEARL-server
# run in docker container
- just compose up, which may take 20 ~ 25 minutes
```
docker compose up
```

## run in host machine
0. create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```  
1. install FastChat
```bash
cd FastChat
pip install -e .
```
2. install nano-PEARL
```bash
cd nano-PEARL
pip install -e .
```
3. start controller
```bash
bash controller.sh
```
4. start worker
```bash
bash worker.sh
```
5. start api server
```bash
bash api.sh
```
