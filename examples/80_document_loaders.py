#!/usr/bin/env python3
"""
Document Loaders -- CSV, JSON, HTML, URL loading.

No API key needed. Demonstrates all new document loader methods.

Run: python examples/80_document_loaders.py
"""

import json
import os
import tempfile

from selectools.rag.loaders import DocumentLoader

print("=== Document Loaders Example ===\n")

# 1. CSV Loader
csv_path = os.path.join(tempfile.mkdtemp(), "products.csv")
with open(csv_path, "w") as f:
    f.write("name,description,price\n")
    f.write("Widget A,A useful productivity widget,9.99\n")
    f.write("Widget B,Premium widget with extra features,19.99\n")
    f.write("Gadget X,Compact gadget for daily use,14.99\n")

docs = DocumentLoader.from_csv(csv_path, text_column="description")
print(f"CSV: {len(docs)} documents")
for d in docs:
    print(f"  - {d.text} (row {d.metadata.get('row', '?')})")

# 2. JSON Loader
json_path = os.path.join(tempfile.mkdtemp(), "articles.json")
with open(json_path, "w") as f:
    json.dump(
        [
            {"text": "Python 3.13 released with free-threaded mode", "author": "PSF"},
            {"text": "New AI framework benchmarks show 10x speedup", "author": "ML Weekly"},
            {"text": "WebAssembly gains traction in server-side apps", "author": "Dev Blog"},
        ],
        f,
    )

docs = DocumentLoader.from_json(json_path, text_field="text", metadata_fields=["author"])
print(f"\nJSON: {len(docs)} documents")
for d in docs:
    print(f"  - {d.text[:50]}... (by {d.metadata.get('author', '?')})")

# 3. HTML Loader
html_path = os.path.join(tempfile.mkdtemp(), "page.html")
with open(html_path, "w") as f:
    f.write(
        """<html>
<head><title>Test Page</title></head>
<body>
<h1>Welcome</h1>
<p>This is a test page with some content.</p>
<p>It has multiple paragraphs for extraction.</p>
</body>
</html>"""
    )

docs = DocumentLoader.from_html(html_path)
print(f"\nHTML: {len(docs)} documents")
print(f"  Text: {docs[0].text[:80]}...")

# 4. URL Loader (would fetch from web -- shown as pattern)
print("\nURL loader pattern:")
print("  docs = DocumentLoader.from_url('https://example.com/article')")
print("  # Auto-detects content type, strips HTML tags")

# Cleanup
for p in [csv_path, json_path, html_path]:
    os.unlink(p)

print("\nDone!")
