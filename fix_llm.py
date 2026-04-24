with open("llm.py", "r") as f:
    text = f.read()
text = text.replace('\\"\\"\\"', '"""')
with open("llm.py", "w") as f:
    f.write(text)
