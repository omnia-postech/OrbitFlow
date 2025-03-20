## How to Run

### Partial DistN 
cache_engine.py/CacheEngine/__init__.py의 self.is_monolithic_distn = False 로 변경
```bash
python offline_inference_offload_long_out.py
```

### Monolithic DistN
cache_engine.py/CacheEngine/__init__.py의 self.is_monolithic_distn = True 로 변경
```bash
python offline_inference_offload_long_out.py
```