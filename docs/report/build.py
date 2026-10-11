#!/usr/bin/env python3
"""
Build the BCSE497J project report.

Two passes. The first lays the document out with a placeholder table of
contents; the page numbers are then read off the rendered PDF and the second
pass writes them in. A Word TOC field would be the lazier route, but it shows
as empty until the reader presses F9, and a report that is printed and bound
should not depend on that.

    python build.py            # pass 1, then pass 2 if page map exists
"""

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ch45, ch67, ch123, front  # noqa: E402
from rlib import (CONTENT_W, FONT, base_style, frontmatter_heading, no_footer,  # noqa: E402
                  number_format, page_break, page_number_footer, para, setup_section)

PAGEMAP = HERE / "pagemap.json"

# Entries are (level, text, key). `key` matches what pass 2 looks up.
CONTENTS = [
    (0, "ABSTRACT", "ABSTRACT"),
    (0, "ACKNOWLEDGEMENT", "ACKNOWLEDGEMENT"),
    (0, "LIST OF FIGURES", "LIST OF FIGURES"),
    (0, "LIST OF TABLES", "LIST OF TABLES"),
    (0, "LIST OF ACRONYMS", "LIST OF ACRONYMS"),
    (0, "CHAPTER 1  INTRODUCTION", "CHAPTER 1"),
    (1, "1.1  Introduction", "1.1"),
    (1, "1.2  Background and Motivation", "1.2"),
    (1, "1.3  Problem Statement", "1.3"),
    (1, "1.4  Objectives", "1.4"),
    (1, "1.5  Scope of the Project", "1.5"),
    (1, "1.6  Organisation of the Report", "1.6"),
    (0, "CHAPTER 2  LITERATURE SURVEY", "CHAPTER 2"),
    (1, "2.1  Introduction", "2.1"),
    (1, "2.2  Linguistic Markers of Psychological State", "2.2"),
    (1, "2.3  Process Coding in Motivational Interviewing", "2.3"),
    (1, "2.4  Prior Computational Work on AnnoMI", "2.4"),
    (1, "2.5  Summary of Reviewed Literature", "2.5"),
    (1, "2.6  Multimodal and Longitudinal Approaches", "2.6"),
    (1, "2.7  Research Gap", "2.7"),
    (0, "CHAPTER 3  REQUIREMENTS AND SYSTEM ANALYSIS", "CHAPTER 3"),
    (1, "3.1  Introduction", "3.1"),
    (1, "3.2  Functional Requirements", "3.2"),
    (1, "3.3  Non-Functional Requirements", "3.3"),
    (1, "3.4  Feasibility Analysis", "3.4"),
    (1, "3.5  Software and Hardware Requirements", "3.5"),
    (0, "CHAPTER 4  SYSTEM DESIGN", "CHAPTER 4"),
    (1, "4.1  Introduction", "4.1"),
    (1, "4.2  System Architecture", "4.2"),
    (1, "4.3  Module Description", "4.3"),
    (1, "4.4  Behavioural Design", "4.4"),
    (1, "4.5  Data Model", "4.5"),
    (1, "4.6  Scoring Methodology", "4.6"),
    (0, "CHAPTER 5  IMPLEMENTATION", "CHAPTER 5"),
    (1, "5.1  Introduction", "5.1"),
    (1, "5.2  Implementation Environment", "5.2"),
    (1, "5.3  Measurement Layer", "5.3"),
    (1, "5.4  Supervised Layer", "5.4"),
    (1, "5.5  Bias Correction Before Display", "5.5"),
    (1, "5.6  Web Application", "5.6"),
    (1, "5.7  Testing", "5.7"),
    (0, "CHAPTER 6  RESULTS AND DISCUSSION", "CHAPTER 6"),
    (1, "6.1  Introduction", "6.1"),
    (1, "6.2  Dataset and Protocol", "6.2"),
    (1, "6.3  Model Comparison", "6.3"),
    (1, "6.4  Error Propagation and Correction", "6.4"),
    (1, "6.5  Reframing the Task", "6.5"),
    (1, "6.6  System-Level Results", "6.6"),
    (1, "6.7  Discussion", "6.7"),
    (0, "CHAPTER 7  CONCLUSION AND FUTURE WORK", "CHAPTER 7"),
    (1, "7.1  Conclusion", "7.1"),
    (1, "7.2  Limitations", "7.2"),
    (1, "7.3  Future Work", "7.3"),
    (1, "7.4  Closing Remark", "7.4"),
    (0, "REFERENCES", "REFERENCES"),
    (0, "APPENDICES", "APPENDICES"),
]

FIGURES = [
    ("4.1", "System architecture"),
    ("4.2", "Use case diagram"),
    ("4.3", "Activity diagram for the end-to-end scoring pipeline"),
    ("4.4", "Sequence diagram for adding a session"),
    ("4.5", "Entity relationship diagram"),
    ("5.1", "Case trajectory view"),
    ("5.2", "Model comparison page"),
    ("6.1", "Cross-validated macro-F1 for each learner on both tasks"),
    ("6.2", "Fold-level macro-F1 with paired-test results"),
    ("6.3", "Macro-F1 against classifier fit time"),
    ("6.4", "Per-class F1 for every learner"),
    ("6.5", "Confusion matrices for the deployed model"),
    ("6.6", "Displayed ratio against true ratio, and sustain recall by learner"),
    ("6.7", "Feature ablation"),
    ("6.8", "Change-talk rate by therapist move and client state"),
]

TABLES = [
    ("2.1", "Prior computational work using the AnnoMI corpus"),
    ("2.2", "Summary of reviewed literature"),
    ("3.1", "Functional requirements"),
    ("3.2", "Development environment"),
    ("4.1", "Principal system modules"),
    ("4.2", "The five therapeutic process dimensions"),
    ("6.1", "Task composition after filtering"),
    ("6.2", "Client talk type: held-out results"),
    ("6.3", "Therapist behaviour: held-out results"),
    ("6.4", "Client talk type: per-class recall"),
    ("6.5", "Three-class against binary framing"),
    ("6.6", "Feature ablation, cross-validated macro-F1"),
    ("6.7", "Change talk by therapist move and client state"),
    ("A1.1", "Repository layout"),
    ("A3.1", "Extracted dimension scores for the worked example"),
]

ACRONYMS = [
    ("ACC", "Adjusted Classify and Count"),
    ("ANNOMI", "Annotated Motivational Interviewing corpus"),
    ("API", "Application Programming Interface"),
    ("AUC", "Area Under the Receiver Operating Characteristic Curve"),
    ("CBT", "Cognitive Behavioural Therapy"),
    ("CV", "Cross-Validation"),
    ("DAIC-WOZ", "Distress Analysis Interview Corpus – Wizard of Oz"),
    ("F1", "Harmonic mean of precision and recall"),
    ("GAD-7", "Generalised Anxiety Disorder 7-item scale"),
    ("LLM", "Large Language Model"),
    ("MI", "Motivational Interviewing"),
    ("MISC", "Motivational Interviewing Skill Code"),
    ("MITI", "Motivational Interviewing Treatment Integrity"),
    ("MLP", "Multilayer Perceptron"),
    ("OLS", "Ordinary Least Squares"),
    ("PHQ-9", "Patient Health Questionnaire 9-item scale"),
    ("REST", "Representational State Transfer"),
    ("ROC", "Receiver Operating Characteristic"),
    ("SVM", "Support Vector Machine"),
    ("TF-IDF", "Term Frequency – Inverse Document Frequency"),
    ("TPI", "Therapeutic Progress Index"),
    ("UML", "Unified Modeling Language"),
]


def dotted(doc, text, pageno, level=0):
    """One contents line: text at the left, page number hard right."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(4 if level == 0 else 2)
    pf.line_spacing = 1.0
    pf.left_indent = Cm(0 if level == 0 else 0.9)

    tabs = p.paragraph_format.tab_stops
    from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
    tabs.add_tab_stop(CONTENT_W, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)

    r = p.add_run(text)
    r.font.name, r.font.size = FONT, Pt(12)
    r.bold = level == 0
    r2 = p.add_run("\t" + str(pageno))
    r2.font.name, r2.font.size = FONT, Pt(12)
    r2.bold = level == 0
    return p


def contents_pages(doc, pages):
    page_break(doc)
    frontmatter_heading(doc, "CONTENTS", underline=False)
    for level, text, key in CONTENTS:
        dotted(doc, text, pages.get(key, ""), level)

    page_break(doc)
    frontmatter_heading(doc, "LIST OF FIGURES", underline=False)
    for num, cap in FIGURES:
        dotted(doc, f"Figure {num}   {cap}", pages.get(f"FIG{num}", ""), 1)

    page_break(doc)
    frontmatter_heading(doc, "LIST OF TABLES", underline=False)
    for num, cap in TABLES:
        dotted(doc, f"Table {num}   {cap}", pages.get(f"TAB{num}", ""), 1)

    page_break(doc)
    frontmatter_heading(doc, "LIST OF ACRONYMS", underline=False)
    for ab, full in ACRONYMS:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.0
        tabs = p.paragraph_format.tab_stops
        from docx.enum.text import WD_TAB_ALIGNMENT
        tabs.add_tab_stop(Cm(4.2), WD_TAB_ALIGNMENT.LEFT)
        r = p.add_run(ab)
        r.font.name, r.font.size, r.bold = FONT, Pt(12), True
        r2 = p.add_run("\t" + full)
        r2.font.name, r2.font.size = FONT, Pt(12)


def build(out_path, pages):
    doc = Document()
    base_style(doc)

    # --- section 1: cover and title, unnumbered -------------------------
    s1 = setup_section(doc.sections[0])
    no_footer(s1)
    front.cover_and_title(doc)

    # --- section 2: front matter, lower-case roman ----------------------
    s2 = setup_section(doc.add_section(WD_SECTION.NEW_PAGE))
    s2.footer.is_linked_to_previous = False
    number_format(s2, "lowerRoman", 1)
    page_number_footer(s2)
    front.declaration(doc)
    front.certificate(doc)
    front.abstract(doc)
    front.acknowledgement(doc)
    contents_pages(doc, pages)

    # --- section 3: body, arabic restarting at 1 ------------------------
    s3 = setup_section(doc.add_section(WD_SECTION.NEW_PAGE))
    s3.footer.is_linked_to_previous = False
    number_format(s3, "decimal", 1)
    page_number_footer(s3)

    ch123.ch1(doc, first=True)
    ch123.ch2(doc)
    ch123.ch3(doc)
    ch45.ch4(doc)
    ch45.ch5(doc)
    ch67.ch6(doc)
    ch67.ch7(doc)
    ch67.references(doc)
    ch67.appendices(doc)

    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    pages = {}
    if PAGEMAP.exists():
        pages = json.loads(PAGEMAP.read_text())
        print(f"pass 2 — writing {len(pages)} page numbers into the contents")
    else:
        print("pass 1 — placeholder contents")
    out = build(HERE / "TherapyTrace_Project_Report.docx", pages)
    print(f"wrote {out}")
