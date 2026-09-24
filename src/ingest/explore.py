import sys
import fitz
import pdfplumber

path = sys.argv[1]
first_page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
last_page_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None

doc = fitz.open(path)
print("Pages:", len(doc))

last_page = last_page_arg or len(doc)

toc = doc.get_toc()
print(f"TOC entries: {len(toc)} (showing first 15): {toc[:15]}")

print("\nTOC entries matching fault/warning/tracing:")
for lvl, title, page in toc:
    if any(k in title.lower() for k in ("fault", "warning", "tracing")):
        print(lvl, title, page)

for i in range(first_page - 1, last_page):
    page = doc[i]
    n_img = len(page.get_images())
    n_chars = len(page.get_text())
    print(f"p{i+1}: chars={n_chars} images={n_img}")

with pdfplumber.open(path) as pdf:
    for i in range(first_page - 1, last_page):
        page = pdf.pages[i]
        tables = page.extract_tables()
        if tables:
            print(f"\n--- page {i+1}: {len(tables)} table(s)")
            for row in tables[0][:4]:
                print(row)