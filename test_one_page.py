#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：只处理第一页，用于调试
"""

from paddleocr import PaddleOCRVL
import pypdfium2 as pdfium
import sys
import tempfile
import os

print("=" * 60)
print("测试 PaddleOCR-VL - 处理单页")
print("=" * 60)

# PDF 文件路径
pdf_path = "/home/test/wh/graphrag-main/graphrag-main/ocr2/input/2107472_978-7-109-19624-7_中国农作物病虫害.上册.pdf"
output_dir = "/home/test/wh/graphrag-main/graphrag-main/ocr2/output"

print("\n[1/4] 读取 PDF 第一页...")
pdf = pdfium.PdfDocument(pdf_path)
page = pdf[0]
pil_image = page.render(scale=2.0).to_pil()
page.close()
pdf.close()
print(f"✓ PDF 第一页已转为图片，尺寸: {pil_image.size}")

print("\n[2/4] 初始化 PaddleOCR-VL 模型...")
layout_model_dir = "/home/test/wh/graphrag-main/graphrag-main/ocr2/PaddleOCR-VL/PP-DocLayoutV2"
vl_rec_model_dir = "/home/test/wh/graphrag-main/graphrag-main/ocr2/PaddleOCR-VL/PaddleOCR-VL-0.9B"

try:
    pipeline = PaddleOCRVL(
        layout_detection_model_dir=layout_model_dir,
        vl_rec_model_dir=vl_rec_model_dir
    )
    print("✓ 模型初始化完成")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/4] 处理第一页...")
try:
    # PaddleOCR-VL 只接受文件路径，保存为临时文件
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        temp_image_path = tmp_file.name
        pil_image.save(temp_image_path)
    
    print(f"  临时文件: {temp_image_path}")
    output = pipeline.predict(temp_image_path)
    
    # 删除临时文件
    if os.path.exists(temp_image_path):
        os.unlink(temp_image_path)
    
    print(f"✓ 处理完成，返回 {len(output)} 个结果")
except Exception as e:
    print(f"✗ 处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/4] 保存结果...")
try:
    for idx, res in enumerate(output):
        print(f"\n结果 {idx + 1}:")
        res.print()
        res.save_to_json(save_path=output_dir)
        res.save_to_markdown(save_path=output_dir)
        print(f"✓ 已保存到 {output_dir}")
        break  # 只处理第一个
except Exception as e:
    print(f"✗ 保存失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)

