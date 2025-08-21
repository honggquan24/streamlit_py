# resume.py
from pathlib import Path
import base64, json
import streamlit as st

# Đọc PDF (cache để không đọc lặp lại mỗi lần rerun)
@st.cache_data(show_spinner=False)
def _read_pdf_bytes(path: Path) -> bytes:
    return path.read_bytes()

def show_resume_pdfjs(pdf_path: str = "cv/CV_VoHongQuan.pdf",
                      height: int = 900,
                      base_scale: float = 0.90):
    """
    Hiển thị PDF bằng PDF.js (render sắc nét, không bị trình duyệt chặn).
    - height: chiều cao khung xem trong app
    - base_scale: mức zoom cơ sở; độ phân giải thực = base_scale * devicePixelRatio
    """
    p = Path(pdf_path)
    if not p.exists():
        st.error(f"Không tìm thấy file: {p.resolve()}")
        return

    pdf_bytes = _read_pdf_bytes(p)
    # Bọc an toàn cho JS string
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    b64_js = json.dumps(b64)  # đảm bảo không vỡ chuỗi trong JS

    # HTML + JS: dùng PDF.js, vẽ lên canvas với DPI cao (devicePixelRatio)
    html = f"""
    <div id="pdf_container" style="
        height:{height}px; overflow:auto; border-radius:12px; background:transparent;">
      <div id="toolbar" style="position:sticky; top:0; backdrop-filter:blur(6px);
           display:flex; gap:.5rem; padding:.5rem; z-index:1;">
        <button id="zoom_out">-</button>
        <button id="zoom_in">+</button>
        <span id="page_info" style="margin-left:.5rem; font-family:monospace;"></span>
      </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script>
      // ==== Input & khởi tạo ====
      const B64 = {b64_js};                           // PDF base64
      const baseScale = {base_scale};                 // zoom cơ sở
      let curScale = baseScale;                       // zoom hiện tại
      const DPR = Math.max(1, window.devicePixelRatio || 1); // mật độ điểm ảnh của màn hình

      // Chuẩn bị byte array cho PDF.js
      const raw = atob(B64);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

      // Worker của PDF.js
      pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

      const container = document.getElementById('pdf_container');
      const pageInfo  = document.getElementById('page_info');
      const zoomInBtn = document.getElementById('zoom_in');
      const zoomOutBtn= document.getElementById('zoom_out');

      let pdfDoc = null;

      // Vẽ 1 trang với độ phân giải cao (canvas nội bộ nhân theo DPR)
      async function renderPage(pageNum) {{
        const page = await pdfDoc.getPage(pageNum);

        // viewport ở kích thước CSS mong muốn (base scale * curScale)
        const viewport = page.getViewport({{ scale: curScale }});

        // Tính DPI thực: nhân thêm DPR để nét trên màn hình Retina/4K
        const pxRatio = DPR;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', {{ alpha: false }});
        ctx.imageSmoothingEnabled = true; // mượt hơn khi raster hóa

        // Kích thước VẬT LÝ của canvas (nhân DPR)
        canvas.width  = Math.floor(viewport.width  * pxRatio);
        canvas.height = Math.floor(viewport.height * pxRatio);

        // Kích thước HIỂN THỊ (CSS) — giữ đúng layout
        canvas.style.width  = Math.floor(viewport.width)  + "px";
        canvas.style.height = Math.floor(viewport.height) + "px";
        canvas.style.display = "block";
        canvas.style.margin = "0 auto 12px";

        container.appendChild(canvas);

        // Scale context theo pxRatio để PDF.js vẽ sắc nét
        ctx.setTransform(pxRatio, 0, 0, pxRatio, 0, 0);

        await page.render({{
          canvasContext: ctx,
          viewport: viewport,
          // (tuỳ chọn) intent: "display"
        }}).promise;
      }}

      // Vẽ toàn bộ tài liệu (đơn giản, đủ dùng cho resume vài trang)
      async function renderDocument() {{
        container.querySelectorAll('canvas').forEach(c => c.remove());
        for (let n = 1; n <= pdfDoc.numPages; n++) {{
          await renderPage(n);
        }}
        pageInfo.textContent = `Pages: 1–${{pdfDoc.numPages}} | Zoom: ${{(curScale*100).toFixed(0)}}%`;
      }}

      // Load & render lần đầu
      (async () => {{
        pdfDoc = await pdfjsLib.getDocument({{ data: bytes }}).promise;
        await renderDocument();
      }})();

      // Nút Zoom
      zoomInBtn.onclick  = async () => {{ curScale = Math.min(curScale * 1.15, 4.0);  await renderDocument(); }};
      zoomOutBtn.onclick = async () => {{ curScale = Math.max(curScale / 1.15, 0.5); await renderDocument(); }};

      // Re-render khi DPR đổi (kéo cửa sổ sang màn hình khác)
      matchMedia(`(resolution: ${{DPR}}dppx)`).addEventListener('change', renderDocument);
    </script>
    """

    st.components.v1.html(html, height=height, scrolling=True)
    st.download_button("Download Resume", data=pdf_bytes,
                file_name=p.name, mime="application/pdf", key="dl_cv_pdfjs")

    
