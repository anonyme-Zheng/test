from src.rag.pipelines import RAGPipeline
from src.rag.config import RAGConfig

def main():
    # 示例：加载配置并初始化流水线
    cfg = RAGConfig()
    pipeline = RAGPipeline(cfg)
    print('RAGPipeline 初始化成功')

if __name__ == '__main__':
    main()
