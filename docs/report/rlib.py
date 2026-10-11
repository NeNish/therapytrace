"""
Layout primitives for the BCSE497J project report.

Every rule here comes from the template's own PAGE SPECIFICATIONS page rather
than from how that instruction file happens to be laid out. The template
document itself is set to US Letter with narrow margins, but its text states A4
with a 3.81 cm left margin, so the stated specification wins — the binding edge
is the reason for the wide left margin and getting it wrong is visible on paper.

    A4, left 3.81 cm, right/top/bottom 2.54 cm
    Times New Roman 12 pt throughout, 1.5 line spacing
    Chapter heading 16 bold · Section heading 14 CAPS · Subsection 12 CAPS
    Captions 10 pt minimum, single spaced
    Page numbers bottom centre: roman for front matter, arabic from Chapter 1
"""

from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
BODY = 12
LINE = 1.5

# A4 and the stated margins, in centimetres.
PAGE_W, PAGE_H = Cm(21.0), Cm(29.7)
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = Cm(3.81), Cm(2.54), Cm(2.54), Cm(2.54)

# Usable text width, for sizing figures and tables so nothing overflows.
CONTENT_W = Cm(21.0 - 3.81 - 2.54)


# ---------------------------------------------------------------- page setup

def setup_section(sec, first=False):
    sec.page_width, sec.page_height = PAGE_W, PAGE_H
    sec.left_margin, sec.right_margin = M_LEFT, M_RIGHT
    sec.top_margin, sec.bottom_margin = M_TOP, M_BOTTOM
    return sec


def _field(run, instr):
    """A Word field code — used for PAGE numbers and the TOC placeholder."""
    r = run._r
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    txt = OxmlElement("w:instrText")
    txt.set(qn("xml:space"), "preserve")
    txt.text = instr
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, txt, sep, end):
        r.append(el)


def page_number_footer(sec, size=BODY):
    """Arabic or roman numeral, bottom centre, as the specification requires."""
    footer = sec.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.name = FONT
    run.font.size = Pt(size)
    _field(run, "PAGE")


def number_format(sec, fmt="decimal", start=1):
    """
    fmt: 'lowerRoman' for front matter, 'decimal' from Chapter 1.

    Any existing element is removed first. A new section inherits the previous
    section's properties, so appending without clearing leaves two pgNumType
    elements in one sectPr; LibreOffice happens to honour the last, but that is
    not guaranteed and the file is the one a student opens in Word.

    The element's position matters too. The schema fixes the order of sectPr's
    children, and pgNumType belongs after pgMar and before cols; appending it
    at the end puts it after cols and the file fails validation.
    """
    for old in sec._sectPr.findall(qn("w:pgNumType")):
        sec._sectPr.remove(old)
    pgnum = OxmlElement("w:pgNumType")
    pgnum.set(qn("w:fmt"), fmt)
    pgnum.set(qn("w:start"), str(start))

    # Insert before the first child that must follow pgNumType.
    following = ("w:cols", "w:formProt", "w:vAlign", "w:noEndnote",
                 "w:titlePg", "w:textDirection", "w:bidi", "w:rtlGutter",
                 "w:docGrid", "w:printerSettings", "w:sectPrChange")
    anchor = None
    for child in sec._sectPr:
        if child.tag in (qn(t) for t in following):
            anchor = child
            break
    if anchor is not None:
        anchor.addprevious(pgnum)
    else:
        sec._sectPr.append(pgnum)


def no_footer(sec):
    """Cover and title pages carry no number."""
    sec.footer.is_linked_to_previous = False
    for p in sec.footer.paragraphs:
        p.text = ""
    sec.different_first_page_header_footer = False


# --------------------------------------------------------------- paragraphs

def para(doc, text="", size=BODY, bold=False, italic=False, underline=False,
         align="justify", caps=False, before=0, after=6, line=LINE, space=0):
    p = doc.add_paragraph()
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[align]
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if text:
        run = p.add_run(text.upper() if caps else text)
        run.font.name = FONT
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.underline = underline
        if space:
            rpr = run._r.get_or_add_rPr()
            sp = OxmlElement("w:spacing")
            sp.set(qn("w:val"), str(space))
            rpr.append(sp)
    return p


def rich(doc, parts, size=BODY, align="justify", after=6, line=LINE):
    """A paragraph whose runs differ — (text, bold, italic) tuples."""
    p = doc.add_paragraph()
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT, "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[align]
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    for text, bold, italic in parts:
        r = p.add_run(text)
        r.font.name = FONT
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
    return p


def bullets(doc, items, size=BODY, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        p.paragraph_format.line_spacing = LINE
        p.paragraph_format.space_after = Pt(4)
        if isinstance(it, tuple):
            lead, rest = it
            r = p.add_run(lead)
            r.font.name, r.font.size, r.bold = FONT, Pt(size), True
            r2 = p.add_run(rest)
            r2.font.name, r2.font.size = FONT, Pt(size)
        else:
            r = p.add_run(it)
            r.font.name, r.font.size = FONT, Pt(size)


def numbered(doc, items, size=BODY):
    bullets(doc, items, size=size, style="List Number")


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ----------------------------------------------------------------- headings

def chapter(doc, number, title, new_page=True):
    """Chapter heading — 16 pt bold, as specified. Starts a page."""
    if new_page:
        page_break(doc)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(f"CHAPTER {number}")
    r.font.name, r.font.size, r.bold = FONT, Pt(16), True

    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_after = Pt(16)
    h.paragraph_format.line_spacing = 1.0
    r = h.add_run(title.upper())
    r.font.name, r.font.size, r.bold = FONT, Pt(16), True
    _outline(h, 1)
    return h


def section(doc, number, title):
    """Section heading — 14 pt, CAPS."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(f"{number} {title.upper()}")
    r.font.name, r.font.size, r.bold = FONT, Pt(14), True
    _outline(p, 2)
    return p


def subsection(doc, number, title):
    """Subsection heading — 12 pt, CAPS."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(f"{number} {title.upper()}")
    r.font.name, r.font.size, r.bold = FONT, Pt(12), True
    _outline(p, 3)
    return p


def frontmatter_heading(doc, title, underline=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(title.upper())
    r.font.name, r.font.size, r.bold, r.underline = FONT, Pt(14), True, underline
    return p


def _outline(p, level):
    """Needed or the heading will not appear in a generated table of contents."""
    pPr = p._p.get_or_add_pPr()
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), str(level - 1))
    pPr.append(ol)


# ------------------------------------------------------------------ figures

def figure(doc, path, number, caption, width_cm=None, before=10):
    from docx.shared import Cm as _Cm
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run()
    run.add_picture(path, width=_Cm(width_cm) if width_cm else CONTENT_W)

    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(12)
    c.paragraph_format.line_spacing = 1.0
    r = c.add_run(f"Figure {number}: {caption}")
    r.font.name, r.font.size, r.bold = FONT, Pt(10), False
    return c


def table(doc, number, caption, headers, rows, widths=None, size=10,
          caption_above=True):
    """
    Caption above the table, as is conventional for tables, figures below.
    Column widths are set on the table and on every cell, in DXA, because a
    width set in only one place is ignored by some renderers.
    """
    if caption_above:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_before = Pt(10)
        c.paragraph_format.space_after = Pt(4)
        c.paragraph_format.line_spacing = 1.0
        r = c.add_run(f"Table {number}: {caption}")
        r.font.name, r.font.size = FONT, Pt(10)

    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    total = CONTENT_W.cm
    widths = widths or [total / len(headers)] * len(headers)
    scale = total / sum(widths)
    widths = [w * scale for w in widths]

    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.font.name, r.font.size, r.bold = FONT, Pt(size), True
        _shade(hdr[i], "E8E8E8")

    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if i == 0
                           else WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(str(val))
            r.font.name, r.font.size = FONT, Pt(size)

    for row in t.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Cm(widths[i])
    _repeat_header(t)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return t


def _shade(cell, hexval):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")       # never "solid" — renders black
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexval)
    tcPr.append(shd)


def _repeat_header(t):
    """A long table that breaks across pages should repeat its header row."""
    tr = t.rows[0]._tr
    trPr = tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def base_style(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(BODY)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.line_spacing = LINE
    pf.space_after = Pt(6)
    return doc
