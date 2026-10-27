"""Minimal .xlsx reader (zipfile + ElementTree), so that no extra package is needed. Returns a list of rows (lists of str/None)."""
import re, zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def col_idx(ref):
    letters = re.match(r"[A-Z]+", ref).group(0); n = 0
    for ch in letters: n = n * 26 + ord(ch) - 64
    return n - 1


def read_sheet(path, sheet_xml):
    z = zipfile.ZipFile(path)
    ss = [("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"]))) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    rows = []
    for _, row in ET.iterparse(z.open(sheet_xml), events=("end",)):
        if row.tag != "{%s}row" % NS["m"]: continue
        cells = {}
        for c in row.findall("m:c", NS):
            v = c.find("m:v", NS)
            if v is None: continue
            cells[col_idx(c.get("r"))] = ss[int(v.text)] if c.get("t") == "s" else v.text
        rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
        row.clear()
    return rows
