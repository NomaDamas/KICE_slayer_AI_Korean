import fitz  # PyMuPDF

if __name__ == "__main__":
    # PDF 파일 열기
    pdf_file = "./data/수능국어.pdf"
    pdf_document = fitz.open(pdf_file)
    output_file = "./data/수능국어텍스트변환.txt"

    # txt 파일을 쓰기 모드로 열기
    with open(output_file, "w", encoding="utf-8") as file:
        # 각 페이지의 텍스트 추출
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text()  # 텍스트 추출
            print(f"Page {page_num + 1}:\n{text}")

            file.write(f"Page {page_num + 1}:\n{text}\n\n")

    pdf_document.close()
