from playwright.sync_api import sync_playwright
import subprocess
import glob
import os, time

URL = "https://api-dev.tkb.la/story/ring"
SCALE = 10  # 50s -> 5s

# 你的 autoSlide（方案C压缩成 JS 字符串）
AUTO_SLIDE_JS = r"""
async function autoSlide(durationMs = 800){
  const knob = document.getElementById("knob");
  const bar  = document.getElementById("slideBar");
  const pad = parseFloat(getComputedStyle(document.documentElement)
    .getPropertyValue('--pad')) || 0;

  const maxX = bar.clientWidth - knob.clientWidth - 2*pad;
  const distance = maxX * 0.95;

  const r = knob.getBoundingClientRect();
  const sx = r.left + r.width/2;
  const sy = r.top  + r.height/2;
  const ex = sx + distance;

  knob.dispatchEvent(new PointerEvent("pointerdown", {
    bubbles:true, clientX:sx, clientY:sy,
    pointerId:1, pointerType:"mouse", isPrimary:true, buttons:1
  }));

  const steps = Math.max(10, Math.floor(durationMs / 16));
  for (let i=1;i<=steps;i++){
    const x = sx + (ex-sx)*i/steps;
    window.dispatchEvent(new PointerEvent("pointermove", {
      bubbles:true, clientX:x, clientY:sy,
      pointerId:1, pointerType:"mouse", isPrimary:true, buttons:1
    }));
    await new Promise(r=>setTimeout(r, durationMs/steps));
  }

  window.dispatchEvent(new PointerEvent("pointerup", {
    bubbles:true, clientX:ex, clientY:sy,
    pointerId:1, pointerType:"mouse", isPrimary:true
  }));
}
window.autoSlide = autoSlide;
"""

# timeScale 补丁（让页面 10x 时间流逝）
TIME_SCALE_JS = f"""
(function speedUpTime(scale={SCALE}){{
  const _now = performance.now.bind(performance);
  const t0_real = _now();
  const t0_fake = t0_real;
  performance.now = () => t0_fake + (_now() - t0_real) * scale;

  const _dateNow = Date.now.bind(Date);
  const d0_real = _dateNow();
  const d0_fake = d0_real;
  Date.now = () => d0_fake + (_dateNow() - d0_real) * scale;

  const _setTimeout = window.setTimeout.bind(window);
  const _setInterval = window.setInterval.bind(window);
  window.setTimeout = (fn, ms, ...args) => _setTimeout(fn, ms/scale, ...args);
  window.setInterval = (fn, ms, ...args) => _setInterval(fn, ms/scale, ...args);

  const _Audio = window.Audio;
  window.Audio = function(...args){{
    const a = new _Audio(...args);
    a.playbackRate = scale;
    return a;
  }};
  document.querySelectorAll("audio").forEach(a => a.playbackRate = scale);
}})();
"""

def main():
  os.makedirs("videos", exist_ok=True)

  with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
      record_video_dir="videos",
      record_video_size={"width": 390, "height": 844}
    )

    # ✅ 在页面脚本之前注入加速
    context.add_init_script(TIME_SCALE_JS)

    page = context.new_page()
    page.goto(URL, wait_until="networkidle")

    page.add_script_tag(content=AUTO_SLIDE_JS)
    page.wait_for_timeout(300)
    page.evaluate("window.autoSlide(1200)")
    page.wait_for_timeout(6000)   # 录 6 秒足够装下 50 秒内容

    context.close()
    browser.close()

  # 找到录到的 webm
  webms = glob.glob("videos/*.webm")
  if not webms:
    raise RuntimeError("no video found")
  src = sorted(webms)[-1]
  out = "videos/auto-slide.mp4"

  # 转 mp4
  subprocess.check_call([
    "ffmpeg", "-y", "-i", src,
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    out
  ])
  print("✅ mp4:", out)

if __name__ == "__main__":
  main()
