"""Check the compiled report, render all pages, and optionally package sources.

Requires numpy, Pillow, and PyMuPDF. --package also requires a visual_review.json
record tied to the exact PDF hash; visual inspection is recorded separately.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=Path("/tmp/methods-overview-review"))
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    evidence = json.loads((HERE/"evidence.json").read_text())
    for name, sha in evidence["sha256"].items():
        assert digest(ROOT/name) == sha, f"Input changed: {name}"
    for name, sha in evidence["outputs_sha256"].items():
        assert digest(HERE/name) == sha, f"Generated file changed: {name}"
    log = (HERE/"methods_overview.log").read_text()
    assert not re.search(r"Warning:|Overfull|Underfull|undefined|Missing character", log), "LaTeX warning"
    tex = (HERE/"methods_overview.tex").read_text()
    forbidden = r"\b(ongoing|currently running|queue|queued|forecast|promotion|paused|heartbeat|supervisor|Gauss--Newton|Levenberg|dated snapshot|findings to date)\b"
    assert not re.search(forbidden, tex, re.I)
    assert "\\appendix" not in tex
    figures = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex)
    assert len(figures) == 6 and all((HERE/f).is_file() for f in figures)
    for path in (HERE/"figures").glob("*_96.png"):
        with Image.open(path) as img:
            assert img.size == (96,96)
    with np.load(HERE/"data/image_formation_native.npz", allow_pickle=False) as native:
        assert all(native[name].shape == (2,96,96) for name in ("linear","encoded","quantized","thresholded"))
        np.testing.assert_array_equal(native["quantized"], np.round(255*native["encoded"])/255)
        np.testing.assert_array_equal(native["thresholded"], native["quantized"]*(native["quantized"]>.1))
        assert float(native["angle_degrees"]) == 45
    rows = (HERE/"tables/l4_rows.tex").read_text().splitlines()
    assert len(rows) == 10
    ids = json.loads((HERE/"inputs.json").read_text())["l4_asteroids"]
    for a, row in enumerate(rows,1):
        cells = [cell.strip() for cell in row.removesuffix(r"\\").split("&")]
        assert cells[0] == str(a)+(r"$^\dagger$" if a>3 else "")
        if a not in ids:
            assert cells[1:] == [r"\textemdash"]*4
            continue
        record = evidence["checks"][f"artifacts/octantoid_l4_all10_200_20260910/asteroid{a:02d}/octantoid_l4"]
        scores = record["full_turn_rmse"]
        expected = [str(record["selected_iteration"]), f"{scores['initial']:.4f}", f"{scores['selected']:.4f}",
                    f"{100*(1-scores['selected']/scores['initial']):.1f}"+r"\%"]
        assert cells[1:] == expected
    pdf = HERE/"methods_overview.pdf"
    doc = pymupdf.open(pdf)
    assert 8 <= len(doc) <= 10, f"Unexpected page count: {len(doc)}"
    args.review_dir.mkdir(parents=True,exist_ok=True)
    thumbnails = []
    texts = []
    for index,page in enumerate(doc):
        text = unicodedata.normalize("NFKC",page.get_text())
        texts.append(text)
        assert "??" not in text and not re.search(forbidden,text,re.I)
        for word in page.get_text("words"):
            x0,y0,x1,y1 = word[:4]
            assert x0>=0 and y0>=0 and x1<=page.rect.width and y1<=page.rect.height, f"Text outside page {index+1}"
        image = args.review_dir/f"page-{index+1:02d}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(str(image))
        thumb = Image.open(image);thumb.thumbnail((297,421))
        canvas=Image.new("RGB",(317,451),"#dddddd");canvas.paste(thumb,(10,20))
        ImageDraw.Draw(canvas).text((10,3),str(index+1),fill="black");thumbnails.append(canvas)
    all_text = "\n".join(texts)
    assert "no result reported" in all_text
    # Confirm every rounded L4 number is present on the table's PDF page.
    table_text = next(text for text in texts if "Initial RMSE" in text)
    for row in rows:
        for number in re.findall(r"\d+\.\d+",row):
            assert number in table_text, f"Missing table value in PDF: {number}"
    for number in ("0.1064","0.1193","0.4930","0.4739","0.4598","0.1517"):
        assert number in all_text
    assert all(f"[{i}]" in all_text for i in range(1,6))
    contact=Image.new("RGB",(317*4,451*((len(doc)+3)//4)),"white")
    for index,thumb in enumerate(thumbnails):contact.paste(thumb,((index%4)*317,(index//4)*451))
    contact.save(args.review_dir/"contact.png")
    review_path=HERE/"visual_review.json"
    review=json.loads(review_path.read_text()) if review_path.exists() else {}
    reviewed=review.get("pdf_sha256")==digest(pdf) and review.get("pages_inspected")==list(range(1,len(doc)+1))
    if args.package:
        assert reviewed, "Review every page and record the exact PDF hash in visual_review.json before packaging"
    names=["methods_overview.tex","README.md","make_figure.py","verify_report.py","inputs.json","evidence.json","validation.json"]
    if reviewed:names.append("visual_review.json")
    names += sorted(evidence["outputs_sha256"])
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),pages=len(doc),pdf_sha256=digest(pdf),
        latex_warnings=0,unresolved_references=False,all_pages_rendered=True,
        visual_review_matches_pdf=reviewed,figure_count=len(figures),
        l4_reported_asteroids=ids,l4_table_rows_checked=10,verified_saved_fits=sum(key.startswith("artifacts/") for key in evidence["checks"]),
        native_image_shape=[2,96,96],threshold_reduction_checked=True,
        source_archive_files=names,
        document_sha256={name:digest(HERE/name) for name in
                         ("methods_overview.tex","README.md","verify_report.py","evidence.json")})
    (HERE/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    if args.package:
        with zipfile.ZipFile(HERE/"methods_overview_source.zip","w",zipfile.ZIP_DEFLATED) as archive:
            for name in names:archive.write(HERE/name,name)
        with zipfile.ZipFile(HERE/"methods_overview_source.zip") as archive:
            assert archive.testzip() is None
            assert set(archive.namelist())==set(names)
            assert all(archive.read(name)==(HERE/name).read_bytes() for name in names)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
