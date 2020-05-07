from ..preprocess.tools import extract_pdf_text, tokenize

text = extract_pdf_text("vec2rec/data/resume/Clearstream_Eileen Ong_Ops.pdf")

print(tokenize(text))
print(len(tokenize(text)))
