"""
OCR PDF scan dùng Mistral OCR API — output Markdown
Dùng requests trực tiếp, không phụ thuộc SDK version.

Cài đặt:
    pip install requests python-dotenv

Lấy API key miễn phí tại: https://console.mistral.ai/api-keys
Tạo file .env cùng thư mục:
    MISTRAL_API_KEY=your_key_here
"""

import sys
import os
import argparse
import base64
import json
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def lay_api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        print("❌ Thiếu MISTRAL_API_KEY.")
        print("   Tạo file .env với nội dung: MISTRAL_API_KEY=your_key_here")
        print("   Hoặc: export MISTRAL_API_KEY=your_key_here")
        sys.exit(1)
    return key


def ocr_pdf(
    duong_dan: str,
    tu_trang: int = None,
    den_trang: int = None,
    luu_file: str = None,
):
    path = Path(duong_dan)
    if not path.exists():
        print(f"❌ Không tìm thấy file: {duong_dan}")
        sys.exit(1)

    kich_thuoc_mb = path.stat().st_size / (1024 * 1024)
    if kich_thuoc_mb > 50:
        print(f"⚠️  File {kich_thuoc_mb:.1f}MB — Mistral OCR giới hạn 50MB")

    print(f"📄 File   : {path.name} ({kich_thuoc_mb:.1f}MB)")
    print(f"📃 Trang  : {tu_trang or 0} → {den_trang if den_trang is not None else 'hết'}")
    print("⏳ Đang upload và OCR...\n")

    with open(duong_dan, "rb") as f:
        pdf_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

    api_key = lay_api_key()

    payload = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_base64}",
        },
    }

    # Chỉ định trang nếu có
    if tu_trang is not None or den_trang is not None:
        start = tu_trang or 0
        if den_trang is not None:
            payload["pages"] = list(range(start, den_trang + 1))

    response = requests.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )

    if response.status_code != 200:
        print(f"❌ Lỗi API: {response.status_code}")
        print(response.text)
        sys.exit(1)

    data = response.json()

    # Ghép markdown từ tất cả các trang
    ket_qua_pages = []
    for page in data["pages"]:
        so_trang = page["index"] + 1
        md = page["markdown"]
        print(f"✅ Trang {so_trang} — {len(md)} ký tự")
        ket_qua_pages.append(f"\n\n---\n<!-- Trang {so_trang} -->\n\n{md}")

    pages_processed = data.get("usage_info", {}).get("pages_processed", len(data["pages"]))
    markdown_day_du = "\n".join(ket_qua_pages).strip()

    print(f"\n{'=' * 60}")
    print(f"✅ OCR xong — {pages_processed} trang")
    print(f"{'=' * 60}\n")
    print(markdown_day_du)

    if luu_file:
        Path(luu_file).write_text(markdown_day_du, encoding="utf-8")
        print(f"\n💾 Đã lưu: {luu_file}")

    return markdown_day_du


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR PDF scan → Markdown dùng Mistral OCR API"
    )
    parser.add_argument("file", help="Đường dẫn tới file PDF")
    parser.add_argument("tu_trang", nargs="?", type=int, default=None,
                        help="Trang bắt đầu, tính từ 0 (mặc định: từ đầu)")
    parser.add_argument("den_trang", nargs="?", type=int, default=None,
                        help="Trang kết thúc, tính từ 0 inclusive (mặc định: đến hết)")
    parser.add_argument("--luu", default=None, help="Lưu kết quả ra file .md")

    args = parser.parse_args()

    ocr_pdf(
        duong_dan=args.file,
        tu_trang=args.tu_trang,
        den_trang=args.den_trang,
        luu_file=args.luu,
    )