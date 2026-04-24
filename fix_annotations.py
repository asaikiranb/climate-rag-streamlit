import os

files = ["app.py", "llm.py", "html_renderer.py", "query.py"]
for f in files:
    with open(f, "r") as file:
        content = file.read()
    if "from __future__ import annotations" not in content:
        with open(f, "w") as file:
            file.write("from __future__ import annotations\n" + content)
