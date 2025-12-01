import os
from PIL import Image

INPUT_DIR = "."          # 要处理的目录
OUTPUT_DIR = "webp_out"  # 输出目录
SHORT_SIDE = 720         # 短边目标
QUALITY = 80             # webp 质量 0-100
UPSCALE_SMALL = False    # 是否放大小图：False=不放大

def process_one(in_path, out_path):
    with Image.open(in_path) as im:
        im = im.convert("RGB")  # 保险：jpg都能转RGB

        w, h = im.size
        short = min(w, h)

        # 只缩小，不放大
        if short > SHORT_SIDE or (UPSCALE_SMALL and short < SHORT_SIDE):
            scale = SHORT_SIDE / short
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path, "WEBP", quality=QUALITY, method=6)

def main():
    for root, _, files in os.walk(INPUT_DIR):
        for name in files:
            low = name.lower()
            if low.endswith(".jpg") or low.endswith(".jpeg"):
                in_path = os.path.join(root, name)

                rel = os.path.relpath(in_path, INPUT_DIR)
                rel_noext = os.path.splitext(rel)[0]
                out_path = os.path.join(OUTPUT_DIR, rel_noext + ".webp")

                try:
                    process_one(in_path, out_path)
                    print(f"OK  {in_path} -> {out_path}")
                except Exception as e:
                    print(f"FAIL {in_path}: {e}")

if __name__ == "__main__":
    main()
