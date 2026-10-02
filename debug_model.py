import traceback
from transformers import AutoModel

try:
    m = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base', cache_dir='data/processed/eda/embeddings')
    print('Model loaded', type(m))
except Exception as e:
    traceback.print_exc()
    print(type(e), e)
