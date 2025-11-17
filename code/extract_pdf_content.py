"""
提取PDF文件内容的工具
Extract content from PDF files for analysis
"""

import PyPDF2
import sys
import os


def extract_pdf_text(pdf_path):
    """
    从PDF文件中提取文本内容
    
    Parameters:
        pdf_path: PDF文件路径
        
    Returns:
        text: 提取的文本内容
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return None
    
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print(f"Reading PDF: {os.path.basename(pdf_path)}")
            print(f"Total pages: {num_pages}")
            print("="*60)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += f"\n\n{'='*60}\nPAGE {page_num + 1}\n{'='*60}\n\n"
                text += page_text
                
                # 显示进度
                if (page_num + 1) % 5 == 0 or page_num == num_pages - 1:
                    print(f"  Processed: {page_num + 1}/{num_pages} pages")
        
        print("="*60)
        print(f"✓ Extraction complete!")
        print(f"Total characters: {len(text)}")
        
        return text
    
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def save_text_to_file(text, output_path):
    """保存文本到文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✓ Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


if __name__ == "__main__":
    # 提取 workflow.pdf
    print("\n" + "="*60)
    print("EXTRACTING WORKFLOW.PDF")
    print("="*60 + "\n")
    
    workflow_path = "../mypaper/workflow.pdf"
    workflow_text = extract_pdf_text(workflow_path)
    
    if workflow_text:
        output_workflow = "../mypaper/workflow_extracted.txt"
        save_text_to_file(workflow_text, output_workflow)
    
    print("\n" + "="*60)
    print("EXTRACTING TBMODEL.PDF")
    print("="*60 + "\n")
    
    # 提取 TBmodel.pdf
    tbmodel_path = "../mypaper/TBmodel.pdf"
    tbmodel_text = extract_pdf_text(tbmodel_path)
    
    if tbmodel_text:
        output_tbmodel = "../mypaper/TBmodel_extracted.txt"
        save_text_to_file(tbmodel_text, output_tbmodel)
    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"1. workflow.pdf → workflow_extracted.txt")
    print(f"2. TBmodel.pdf → TBmodel_extracted.txt")
    print("\nYou can now read these files to analyze the content!")
    print("="*60)
