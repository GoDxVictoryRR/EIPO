"""Automated verification script for Project Progress Report quality checks A through E."""

import re
import sys
from pathlib import Path
import docx
from docx.oxml.ns import qn
import pymupdf
import pandas as pd
import yaml

REPORT_DIR = Path(__file__).resolve().parent
DOCX_PATH = REPORT_DIR / 'Project_Progress_Report.docx'
PDF_PATH = REPORT_DIR / 'Project_Progress_Report.pdf'
BASE_DIR = REPORT_DIR.parent
RESULTS_DIR = BASE_DIR / 'results'

def check_a():
    print("=== QUALITY CHECK A: Script scan of DOCX ===")
    doc = docx.Document(str(DOCX_PATH))
    
    fonts = set()
    sizes = set()
    colors = set()
    theme_colors = set()
    empty_paragraphs = 0
    
    # Check body paragraphs
    for p in doc.paragraphs:
        p_xml = p._p.xml
        text = p.text.strip()
        has_draw = '<w:drawing' in p_xml or '<w:pict' in p_xml or '<w:fldSimple' in p_xml or '<w:instrText' in p_xml
        if not text and not has_draw:
            empty_paragraphs += 1
            
        for r in p.runs:
            if r.font.name:
                fonts.add(r.font.name)
            rPr = r._r.rPr
            if rPr is not None:
                rFonts = rPr.find(qn('w:rFonts'))
                if rFonts is not None:
                    for k in [qn('w:ascii'), qn('w:hAnsi'), qn('w:eastAsia'), qn('w:cs')]:
                        val = rFonts.get(k)
                        if val:
                            fonts.add(val)
                sz = rPr.find(qn('w:sz'))
                if sz is not None:
                    val = sz.get(qn('w:val'))
                    if val:
                        sizes.add(float(val) / 2.0)
                color = rPr.find(qn('w:color'))
                if color is not None:
                    val = color.get(qn('w:val'))
                    if val:
                        colors.add(val)
                    th = color.get(qn('w:themeColor'))
                    if th:
                        theme_colors.add(th)

    # Check tables
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    text = p.text.strip()
                    p_xml = p._p.xml
                    has_draw = '<w:drawing' in p_xml or '<w:pict' in p_xml
                    if not text and not has_draw:
                        empty_paragraphs += 1
                    for r in p.runs:
                        if r.font.name:
                            fonts.add(r.font.name)
                        rPr = r._r.rPr
                        if rPr is not None:
                            rFonts = rPr.find(qn('w:rFonts'))
                            if rFonts is not None:
                                for k in [qn('w:ascii'), qn('w:hAnsi'), qn('w:eastAsia'), qn('w:cs')]:
                                    val = rFonts.get(k)
                                    if val:
                                        fonts.add(val)
                            sz = rPr.find(qn('w:sz'))
                            if sz is not None:
                                val = sz.get(qn('w:val'))
                                if val:
                                    sizes.add(float(val) / 2.0)
                            color = rPr.find(qn('w:color'))
                            if color is not None:
                                val = color.get(qn('w:val'))
                                if val:
                                    colors.add(val)
                                th = color.get(qn('w:themeColor'))
                                if th:
                                    theme_colors.add(th)

    # Check footer runs
    for s in doc.sections:
        for p in s.footer.paragraphs:
            for r in p.runs:
                if r.font.name:
                    fonts.add(r.font.name)
                rPr = r._r.rPr
                if rPr is not None:
                    rFonts = rPr.find(qn('w:rFonts'))
                    if rFonts is not None:
                        for k in [qn('w:ascii'), qn('w:hAnsi'), qn('w:eastAsia'), qn('w:cs')]:
                            val = rFonts.get(k)
                            if val:
                                fonts.add(val)
                    sz = rPr.find(qn('w:sz'))
                    if sz is not None:
                        val = sz.get(qn('w:val'))
                        if val:
                            sizes.add(float(val) / 2.0)

    # Global XML check for themeColor
    for element in doc.element.iter():
        for attr in element.attrib.keys():
            if 'themeColor' in attr:
                theme_colors.add(element.attrib[attr])

    print(f"Distinct font names: {sorted(list(fonts))}")
    print(f"Distinct font sizes (pt): {sorted(list(sizes))}")
    print(f"Distinct font colours: {sorted(list(colors))}")
    print(f"Theme colour attributes: {sorted(list(theme_colors))}")
    print(f"Empty paragraphs count: {empty_paragraphs}")
    
    pass_a = (fonts == {'Times New Roman'} and 
              sizes.issubset({12.0, 14.5}) and 
              colors.issubset({'000000'}) and 
              len(theme_colors) == 0 and 
              empty_paragraphs == 0)
    print(f"CHECK A RESULT: {'PASS' if pass_a else 'FAIL'}\n")
    return pass_a

def check_b():
    print("=== QUALITY CHECK B: 11 Section Headings Check ===")
    expected_headings = [
        "1. Title of the Project",
        "2. Introduction",
        "3. Literature Survey",
        "4. Problem Statement",
        "5. Motivation",
        "6. Proposed Design Flow",
        "7. Proposed Methodology",
        "8. Results",
        "9. GANTT/PERT Chart",
        "10. Hardware and Software Requirements",
        "11. References"
    ]
    doc = docx.Document(str(DOCX_PATH))
    found_headings = []
    for p in doc.paragraphs:
        if p.style.name.startswith('Heading 1'):
            found_headings.append(p.text.strip())
            
    print(f"Found headings ({len(found_headings)}):")
    for h in found_headings:
        print(f"  - {h}")
        
    pass_b = (found_headings == expected_headings)
    print(f"CHECK B RESULT: {'PASS' if pass_b else 'FAIL'}\n")
    return pass_b

def check_c():
    print("=== QUALITY CHECK C: Language Sentence Length Check ===")
    doc = docx.Document(str(DOCX_PATH))
    clean_sentences = []
    
    # Process paragraph by paragraph
    for p in doc.paragraphs:
        if p.style.name.startswith('Heading') or not p.text.strip():
            continue
        text = p.text.strip()
        # Skip cover page
        if text.startswith('[COLLEGE') or 'SUBMITTED BY' in text or 'UNDER THE GUIDANCE OF' in text:
            continue
        # Split paragraph into sentences
        raw_sents = re.split(r'\.\s+(?=[A-Z0-9])|\.\Z', text)
        for s in raw_sents:
            s_clean = s.strip()
            if s_clean.endswith('.'):
                s_clean = s_clean[:-1].strip()
            # Exclude equation citations, table captions, or references list entries
            if len(s_clean) > 3 and not s_clean.startswith('[') and not s_clean.startswith('('):
                clean_sentences.append(s_clean)
            
    word_counts = [len(s.split()) for s in clean_sentences]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    long_sentences = [s for s in clean_sentences if len(s.split()) > 35]
    
    print(f"Total evaluated sentences: {len(clean_sentences)}")
    print(f"Average sentence length: {avg_words:.1f} words (target <= 20)")
    print(f"Maximum sentence length: {max_words} words (target <= 35)")
    if long_sentences:
        print(f"Found {len(long_sentences)} sentences above 35 words:")
        for s in long_sentences[:5]:
            print(f"  [{len(s.split())} words]: {s}")
            
    pass_c = (avg_words <= 20.0 and max_words <= 35)
    print(f"CHECK C RESULT: {'PASS' if pass_c else 'FAIL'}\n")
    return pass_c

def check_d():
    print("=== QUALITY CHECK D: PDF Inspection with PyMuPDF ===")
    pdf = pymupdf.open(str(PDF_PATH))
    page_count = len(pdf)
    print(f"Total PDF pages: {page_count}")
    
    for i in range(page_count):
        page = pdf[i]
        text = page.get_text().strip()
        pix = page.get_pixmap(dpi=150)
        img_out = REPORT_DIR / 'figures' / f'page_{i+1}.png'
        pix.save(str(img_out))
        first_line = text.split('\n')[0] if text else '[EMPTY]'
        last_line = text.split('\n')[-1] if text else '[EMPTY]'
        print(f"Page {i+1}: length={len(text)} chars | First: '{first_line[:40]}' | Last: '{last_line[:40]}'")
        
    pdf.close()
    pass_d = (page_count >= 18 and page_count <= 26)
    print(f"CHECK D RESULT: {'PASS' if pass_d else 'FAIL'}\n")
    return pass_d, page_count

def check_e():
    print("=== QUALITY CHECK E: Spot-check Numbers against results/ ===")
    metrics_df = pd.read_csv(RESULTS_DIR / 'metrics.csv', index_col=0)
    sig_df = pd.read_csv(RESULTS_DIR / 'significance.csv').set_index('metric')['value'].to_dict()
    
    pdf = pymupdf.open(str(PDF_PATH))
    full_pdf_text = ""
    for p in pdf:
        full_pdf_text += p.get_text() + " "
    pdf.close()

    checks = [
        ("Enhanced LW Return (15.36%)", f"{metrics_df.loc['enhanced_lw', 'ann_return']*100:.2f}%"),
        ("Benchmark Return (14.51%)", f"{metrics_df.loc['benchmark', 'ann_return']*100:.2f}%"),
        ("Annual Net Excess Return (0.84%)", f"{metrics_df.loc['enhanced_lw', 'excess_return']*100:.2f}%"),
        ("Realized Tracking Error (2.38%)", f"{metrics_df.loc['enhanced_lw', 'realized_TE']*100:.2f}%"),
        ("Information Ratio (0.3191 or 0.319)", "0.319"),
        ("Information Ratio t-statistic (1.031 or 1.03)", "1.03"),
        ("Permutation placebo p-value (0.13)", "0.13")
    ]
    
    all_found = True
    for label, target_str in checks:
        found = target_str in full_pdf_text
        print(f"Spot check '{label}': target '{target_str}' -> {'FOUND' if found else 'NOT FOUND'}")
        if not found:
            all_found = False
            
    print(f"CHECK E RESULT: {'PASS' if all_found else 'FAIL'}\n")
    return all_found

if __name__ == '__main__':
    res_a = check_a()
    res_b = check_b()
    res_c = check_c()
    res_d, pages = check_d()
    res_e = check_e()
    print("ALL CHECKS SUMMARY:")
    print(f"A: {res_a}, B: {res_b}, C: {res_c}, D: {res_d} (pages={pages}), E: {res_e}")
