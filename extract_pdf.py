#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 PaddleOCR-VL 提取 PDF 文档内容
支持进度显示、定时保存和断点续传
"""

from paddleocr import PaddleOCRVL
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import pypdfium2 as pdfium
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue as MPQueue
import threading
import multiprocessing as mp

def get_processed_pages(output_dir):
    """获取已处理的页面列表"""
    processed = set()
    json_files = Path(output_dir).glob("page_*.json")
    for json_file in json_files:
        try:
            page_num = int(json_file.stem.split('_')[1])
            processed.add(page_num)
        except (ValueError, IndexError):
            continue
    return processed

def get_pdf_page_count(pdf_path):
    """获取 PDF 总页数"""
    pdf = pdfium.PdfDocument(pdf_path)
    count = len(pdf)
    pdf.close()
    return count

def save_page_result(res, output_dir, page_num):
    """保存单页结果"""
    import glob
    
    # 先保存到临时目录
    res.save_to_json(save_path=output_dir)
    res.save_to_markdown(save_path=output_dir)
    
    # 找到刚刚生成的临时文件并重命名
    # 获取最新的临时文件
    json_files = sorted(glob.glob(os.path.join(output_dir, "tmp*_res.json")), key=os.path.getmtime, reverse=True)
    md_files = sorted(glob.glob(os.path.join(output_dir, "tmp*.md")), key=os.path.getmtime, reverse=True)
    
    json_path = os.path.join(output_dir, f"page_{page_num:04d}.json")
    md_path = os.path.join(output_dir, f"page_{page_num:04d}.md")
    
    # 重命名文件
    if json_files:
        os.rename(json_files[0], json_path)
    if md_files:
        os.rename(md_files[0], md_path)
    
    return json_path, md_path

def save_progress(output_dir, processed_pages, total_pages, start_time):
    """保存处理进度"""
    progress_file = os.path.join(output_dir, "progress.json")
    progress_data = {
        "total_pages": total_pages,
        "processed_pages": len(processed_pages),
        "processed_page_list": sorted(list(processed_pages)),
        "last_update": datetime.now().isoformat(),
        "elapsed_time_seconds": time.time() - start_time
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

def setup_logging(output_dir):
    """设置日志输出到文件和控制台"""
    log_file = os.path.join(output_dir, "extract.log")
    
    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的 handlers
    logger.handlers.clear()
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 添加 handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def render_page_to_image(pdf_path, page_idx):
    """渲染单页 PDF 为图片（用于并行处理）"""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_idx]
    pil_image = page.render(scale=2.0).to_pil()
    page.close()
    pdf.close()
    
    # 保存为临时文件
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        temp_image_path = tmp_file.name
        pil_image.save(temp_image_path)
    
    return page_idx, temp_image_path

class GPUWorker:
    """GPU 推理工作进程"""
    def __init__(self, gpu_id, worker_id, layout_model_dir, vl_rec_model_dir):
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.layout_model_dir = layout_model_dir
        self.vl_rec_model_dir = vl_rec_model_dir
        self.pipeline = None
    
    def initialize(self):
        """初始化模型（在子进程中调用）"""
        # 设置使用的 GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        # 初始化模型
        self.pipeline = PaddleOCRVL(
            layout_detection_model_dir=self.layout_model_dir,
            vl_rec_model_dir=self.vl_rec_model_dir
        )
        print(f"[Worker-{self.worker_id} GPU-{self.gpu_id}] 模型初始化完成")
    
    def process_image(self, temp_image_path):
        """处理单张图片"""
        try:
            output = self.pipeline.predict(temp_image_path)
            return output
        except Exception as e:
            print(f"[Worker-{self.worker_id}] 处理图片时出错: {e}")
            return None

def gpu_worker_process(gpu_id, worker_id, task_queue, result_queue, layout_model_dir, vl_rec_model_dir):
    """GPU 工作进程函数"""
    # 创建并初始化 worker
    worker = GPUWorker(gpu_id, worker_id, layout_model_dir, vl_rec_model_dir)
    worker.initialize()
    
    # 持续处理任务
    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break
        
        page_idx, temp_image_path = task
        page_num = page_idx + 1
        
        # 处理图片
        output = worker.process_image(temp_image_path)
        
        # 返回结果
        result_queue.put((page_num, temp_image_path, output))
    
    print(f"[Worker-{worker_id} GPU-{gpu_id}] 退出")

def main():
    # PDF 文件路径
    pdf_path = "/home/test/wh/graphrag-main/graphrag-main/ocr2/input/14044948.pdf"
    
    # 输出目录
    output_dir = "/home/test/wh/graphrag-main/graphrag-main/ocr2/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 并行配置
    NUM_RENDER_THREADS = 64  # PDF 渲染线程数（CPU 密集型）
    NUM_GPU_WORKERS = 8  # GPU 推理进程数（每个 GPU 4个）
    PREFETCH_SIZE = 32  # 预渲染队列大小
    TASK_QUEUE_SIZE = 64  # 任务队列大小
    
    # GPU 分配
    GPU_IDS = [0, 1]  # 使用 GPU 0 和 GPU 1
    WORKERS_PER_GPU = NUM_GPU_WORKERS // len(GPU_IDS)  # 每个 GPU 的 worker 数
    
    # 设置日志
    logger = setup_logging(output_dir)
    
    logger.info(f"开始处理 PDF 文件: {pdf_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"日志文件: {output_dir}/extract.log")
    logger.info(f"=== 并行配置 ===")
    logger.info(f"渲染线程数: {NUM_RENDER_THREADS}")
    logger.info(f"GPU 推理进程数: {NUM_GPU_WORKERS} (GPU 0: {WORKERS_PER_GPU}个, GPU 1: {WORKERS_PER_GPU}个)")
    logger.info(f"预渲染队列大小: {PREFETCH_SIZE}")
    logger.info(f"任务队列大小: {TASK_QUEUE_SIZE}")
    
    # 获取 PDF 总页数
    logger.info("\n正在获取 PDF 页数...")
    total_pages = get_pdf_page_count(pdf_path)
    logger.info(f"PDF 总页数: {total_pages}")
    
    # 检查已处理的页面
    processed_pages = get_processed_pages(output_dir)
    if processed_pages:
        logger.info(f"\n检测到已处理 {len(processed_pages)} 页，将从断点继续...")
        logger.info(f"已处理页面: {sorted(list(processed_pages))[:10]}..." if len(processed_pages) > 10 else f"已处理页面: {sorted(list(processed_pages))}")
    
    # 初始化 PaddleOCR-VL - 使用本地模型
    logger.info("\n正在初始化 PaddleOCR-VL 模型...")
    
    # 指定本地模型路径
    layout_model_dir = "/home/test/wh/graphrag-main/graphrag-main/ocr2/PaddleOCR-VL/PP-DocLayoutV2"
    vl_rec_model_dir = "/home/test/wh/graphrag-main/graphrag-main/ocr2/PaddleOCR-VL/PaddleOCR-VL-0.9B"
    
    pipeline = PaddleOCRVL(
        layout_detection_model_dir=layout_model_dir,
        vl_rec_model_dir=vl_rec_model_dir
    )
    
    logger.info("\n模型初始化完成！开始处理文档...")
    
    # 记录开始时间
    start_time = time.time()
    save_interval = 5  # 每处理5页保存一次进度
    
    # 处理 PDF 文件 - 多 GPU 并行推理架构
    try:
        # 需要处理的页面列表
        pages_to_process = [i for i in range(total_pages) if (i + 1) not in processed_pages]
        
        if not pages_to_process:
            logger.info("所有页面已处理完成！")
            return
        
        logger.info(f"待处理页面数: {len(pages_to_process)}")
        
        # 创建多进程队列
        manager = Manager()
        task_queue = manager.Queue(maxsize=TASK_QUEUE_SIZE)
        result_queue = manager.Queue()
        
        # 启动 GPU 工作进程
        gpu_processes = []
        worker_id = 0
        for gpu_id in GPU_IDS:
            for _ in range(WORKERS_PER_GPU):
                p = mp.Process(
                    target=gpu_worker_process,
                    args=(gpu_id, worker_id, task_queue, result_queue, layout_model_dir, vl_rec_model_dir)
                )
                p.start()
                gpu_processes.append(p)
                worker_id += 1
        
        logger.info(f"\n已启动 {len(gpu_processes)} 个 GPU 工作进程")
        time.sleep(5)  # 等待进程启动
        
        # 使用线程池进行并行 PDF 渲染
        with ThreadPoolExecutor(max_workers=NUM_RENDER_THREADS) as render_executor:
            # 渲染任务提交器（异步）
            def submit_render_tasks():
                for page_idx in pages_to_process:
                    future = render_executor.submit(render_page_to_image, pdf_path, page_idx)
                    _, temp_image_path = future.result()
                    task_queue.put((page_idx, temp_image_path))
                
                # 发送结束信号
                for _ in range(NUM_GPU_WORKERS):
                    task_queue.put(None)
            
            # 在单独线程中提交渲染任务
            submit_thread = threading.Thread(target=submit_render_tasks)
            submit_thread.start()
            
            # 收集结果
            processed_count = 0
            results_to_collect = len(pages_to_process)
            
            while processed_count < results_to_collect:
                try:
                    # 获取处理结果
                    page_num, temp_image_path, output = result_queue.get(timeout=300)
                    
                    # 显示进度
                    elapsed = time.time() - start_time
                    if processed_count > 0 and elapsed > 0:
                        speed = processed_count / elapsed
                        eta = (results_to_collect - processed_count) / speed
                    else:
                        speed = 0
                        eta = 0
                    
                    if processed_count % 10 == 0:  # 每10页打印一次
                        logger.info(f"[{page_num}/{total_pages}] 处理中... (速度: {speed:.2f} 页/秒, 预计剩余: {eta/60:.1f} 分钟)")
                    
                    # 保存结果
                    if output:
                        for res in output:
                            json_path, md_path = save_page_result(res, output_dir, page_num)
                            processed_pages.add(page_num)
                            processed_count += 1
                            break
                    
                    # 删除临时文件
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
                    
                    # 定时保存进度
                    if processed_count % save_interval == 0:
                        save_progress(output_dir, processed_pages, total_pages, start_time)
                        logger.info(f">>> 进度: {processed_count}/{results_to_collect} ({processed_count/results_to_collect*100:.1f}%)")
                
                except Exception as e:
                    logger.error(f"收集结果时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # 等待所有任务完成
            submit_thread.join()
        
        # 等待所有 GPU 进程结束
        for p in gpu_processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        
        logger.info(f"\n所有 GPU 工作进程已结束")
        
        logger.info(f"\n所有页面处理完成！")
        
        # 最终保存进度
        save_progress(output_dir, processed_pages, total_pages, start_time)
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"处理完成！")
        logger.info(f"总页数: {total_pages}")
        logger.info(f"已处理: {len(processed_pages)}")
        logger.info(f"总耗时: {total_time/60:.1f} 分钟")
        logger.info(f"平均速度: {len(processed_pages)/total_time:.2f} 页/秒")
        logger.info(f"{'='*60}")
        logger.info(f"\n结果保存在: {output_dir}")
        logger.info(f"- JSON 格式: {output_dir}/page_*.json")
        logger.info(f"- Markdown 格式: {output_dir}/page_*.md")
        logger.info(f"- 进度文件: {output_dir}/progress.json")
        
    except KeyboardInterrupt:
        logger.info("\n\n检测到中断信号，正在保存进度...")
        save_progress(output_dir, processed_pages, total_pages, start_time)
        logger.info(f"进度已保存: {len(processed_pages)}/{total_pages} 页")
        logger.info(f"下次运行将从第 {len(processed_pages)+1} 页继续")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n错误: {e}")
        save_progress(output_dir, processed_pages, total_pages, start_time)
        logger.info(f"进度已保存: {len(processed_pages)}/{total_pages} 页")
        raise

if __name__ == "__main__":
    # 设置多进程启动方式为 'spawn' (跨平台兼容)
    mp.set_start_method('spawn', force=True)
    main()

