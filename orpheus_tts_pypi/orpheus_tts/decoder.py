import os
snac_device = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")