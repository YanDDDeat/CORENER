import logging
def download_model(model_id,cache_path="./model"):
    from modelscope import snapshot_download
    model_dir = snapshot_download(model_id, cache_dir=cache_path)
    logging.info(f"{model_id}下载至完成{cache_path}")

