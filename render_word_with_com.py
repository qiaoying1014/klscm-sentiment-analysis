from pathlib import Path
import sys

import win32com.client


docx = Path(sys.argv[1]).resolve()
pdf = Path(sys.argv[2]).resolve()
word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    document = word.Documents.Open(str(docx), ReadOnly=True)
    try:
        document.ExportAsFixedFormat(str(pdf), 17)
    finally:
        document.Close(False)
finally:
    word.Quit()
print(pdf)
