# GeoDaedalus 使用指南

## 环境配置

### 1. 复制并配置环境变量

```bash
cp env.example .env
```

然后编辑 `.env` 文件，设置以下必要的 API 密钥：

```bash
# 必需的 API 密钥
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_KEY=your_serpapi_key_here  # 可选，用于网络搜索

# 其他配置通常使用默认值即可
```

### 2. uv 包管理器使用

#### 安装 uv（如果尚未安装）
```bash
pip install uv
```

#### 创建独立环境并安装依赖
```bash
# 创建虚拟环境并安装所有依赖
uv sync

# 安装开发依赖（包括测试、lint等工具）
uv sync --all-extras

# 激活环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 运行项目
uv run python demo.py
```

## Makefile 使用

Makefile 提供了便捷的开发命令：

```bash
# 查看所有可用命令
make help

# 基本操作
make install          # 安装依赖
make dev-install      # 安装开发依赖
make clean           # 清理缓存文件

# 代码质量
make test            # 运行测试
make test-cov        # 运行测试并生成覆盖率报告
make lint            # 代码检查
make format          # 代码格式化
make type-check      # 类型检查

# GeoDaedalus 特定命令
make search          # 运行测试搜索
make config-show     # 显示当前配置
make benchmark-run   # 运行基准测试

# 开发环境设置
make setup-dev       # 一键设置开发环境
```

## 核心组件使用

### 1. MetricsCollector (core/metrics.py) 使用

```python
from geodaedalus.core.metrics import get_metrics_collector, MetricsCollector

# 获取全局收集器
metrics = get_metrics_collector()

# 跟踪操作执行时间
with metrics.track_operation("agent_name", "operation_name") as metric:
    # 执行你的操作
    result = do_something()
    # 自动记录执行时间和状态

# 跟踪 LLM 调用
metrics.track_llm_call(
    agent_name="search_agent",
    model="gpt-3.5-turbo", 
    prompt="Your prompt",
    response="Model response",
    cost=0.002
)

# 获取会话摘要
summary = metrics.get_session_summary()
print(f"Total operations: {summary['total_operations']}")
```

### 2. BaseAgent.execute_with_metrics 效果

```python
from geodaedalus.agents.base import BaseAgent

class MyAgent(BaseAgent):
    async def process(self, input_data, **kwargs):
        # 你的处理逻辑
        return processed_result

agent = MyAgent("my_agent")

# 使用 execute_with_metrics 会自动：
# 1. 开始记录执行时间
# 2. 记录操作开始日志
# 3. 执行 process 方法
# 4. 记录成功/失败状态
# 5. 记录完成日志和执行时间
result = await agent.execute_with_metrics("process_data", input_data)
```

### 3. CLI 使用 (cli/main.py)

```bash
# 基本搜索
geodaedalus search "Find volcanic rocks from Hawaii with major elements"

# 高级搜索选项
geodaedalus search "Cretaceous basalts trace elements" \
    --output ./results \
    --max-papers 50 \
    --engines semantic_scholar,google_scholar \
    --verbose

# 逐步执行模式
geodaedalus search "Archean komatiites REE" --step-by-step

# 干运行（只解析查询，不执行搜索）
geodaedalus search "Permian carbonates" --dry-run

# 配置管理
geodaedalus config --show
geodaedalus config --validate

# 导出指标
geodaedalus export-metrics --session SESSION_ID --format json
```

### 4. Demo.py 使用

```bash
# 快速演示（单个查询完整流程）
python demo.py --mode quick

# 完整演示（多个查询）
python demo.py --mode full

# 基准测试演示
python demo.py --mode benchmark

# 直接运行
python demo.py  # 默认快速模式
```

## 开发工作流

### 1. 日常开发
```bash
# 设置环境
make setup-dev

# 开发过程中
make test          # 频繁运行测试
make lint          # 检查代码质量
make format        # 格式化代码

# 提交前
make ci           # 运行所有 CI 检查
```

### 2. 添加新功能
```bash
# 创建新分支
git checkout -b feature/new-feature

# 开发并测试
make test
make type-check

# 提交
git add .
git commit -m "Add new feature"
```

### 3. 调试和监控
```bash
# 启用详细日志
LOG_LEVEL=DEBUG python demo.py

# 查看指标
python -c "
from geodaedalus.core.metrics import get_metrics_collector
metrics = get_metrics_collector()
print(metrics.get_session_summary())
"
```

## 项目结构优化建议

### 移除冗余代码
经过检查，当前代码结构相对精简，主要优化点：

1. **日志模块已简化** - 从 structlog + rich 简化为只使用 loguru
2. **配置模块完整** - 提供了全面的配置管理
3. **依赖项优化** - 保留必要的 rich（CLI需要）和 tiktoken（metrics需要）

### 建议的后续优化
1. 移除未使用的导入（运行 `ruff check --fix` 自动清理）
2. 确保所有 demo 和 CLI 功能都有对应的测试
3. 考虑将某些 CLI 特定的依赖移到 optional extras 中

## 常见问题

### Q: uv 和 pip 的区别？
A: uv 是更快的 Python 包管理器，兼容 pip，但速度更快且功能更强。

### Q: 为什么需要 Makefile？
A: Makefile 提供了标准化的开发命令，确保团队成员使用相同的工作流程。

### Q: 如何扩展 metrics 收集？
A: 继承 `MetricsCollector` 类或使用 `track_operation` 上下文管理器。

### Q: 如何添加新的 CLI 命令？
A: 在 `cli/main.py` 中添加新的 `@app.command()` 装饰的函数。 