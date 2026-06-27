# PyTRIO 官方文档索引

写 PyTRIO/TRIO 代码前，优先把官方文档作为事实来源。

## URL 规则

大多数文档页面都可以把可视化文档 URL 转成 Markdown 读取：

```text
https://docs.pytrio.cn/docs/<route>
https://docs.pytrio.cn/docs/content/<route>/content.md
```

根页面是：

```text
https://docs.pytrio.cn/docs/content/content.md
```

示例：

```text
https://docs.pytrio.cn/docs/guide/train
https://docs.pytrio.cn/docs/content/guide/train/content.md
```

## 核心页面

| 主题 | Markdown 链接 |
|---|---|
| 什么是 TRIO | https://docs.pytrio.cn/docs/content/content.md |
| 快速开始 | https://docs.pytrio.cn/docs/content/quick-start/content.md |
| 训练 | https://docs.pytrio.cn/docs/content/guide/train/content.md |
| 推理/采样 | https://docs.pytrio.cn/docs/content/guide/sample/content.md |
| 保存权重与继续训练 | https://docs.pytrio.cn/docs/content/guide/resume/content.md |
| 下载权重 | https://docs.pytrio.cn/docs/content/guide/download/content.md |
| 损失函数 | https://docs.pytrio.cn/docs/content/guide/loss_fn/content.md |
| 异步 | https://docs.pytrio.cn/docs/content/guide/async/content.md |
| HuggingFace datasets | https://docs.pytrio.cn/docs/content/advanced/datasets/content.md |
| OpenAI 兼容 API | https://docs.pytrio.cn/docs/content/advanced/openai/content.md |
| 模型列表 | https://docs.pytrio.cn/docs/content/models/content.md |
| 合作与交流 | https://docs.pytrio.cn/docs/content/communication/content.md |

## API 页面

| API | Markdown 链接 |
|---|---|
| `trio.ServiceClient` | https://docs.pytrio.cn/docs/content/api/ServiceClient/content.md |
| `trio.TrainingClient` | https://docs.pytrio.cn/docs/content/api/TrainingClient/content.md |
| `trio.SamplingClient` | https://docs.pytrio.cn/docs/content/api/SamplingClient/content.md |
| `trio.RestClient` | https://docs.pytrio.cn/docs/content/api/RestClient/content.md |
| `trio.Datum` | https://docs.pytrio.cn/docs/content/api/Datum/content.md |
| `trio.ModelInput` | https://docs.pytrio.cn/docs/content/api/ModelInput/content.md |
| `trio.AdamParams` | https://docs.pytrio.cn/docs/content/api/AdamParams/content.md |
| `trio.SamplingParams` | https://docs.pytrio.cn/docs/content/api/SamplingParams/content.md |

## 案例

| 案例 | Markdown 链接 |
|---|---|
| Chat-甄嬛 | https://docs.pytrio.cn/docs/content/example/chat_huanhuan/content.md |
| GSM8K | https://docs.pytrio.cn/docs/content/example/gsm8k/content.md |

## 阅读建议

- 写 SFT 时，读取训练、TrainingClient、Datum、ModelInput、AdamParams 和快速开始示例。
- 写推理时，读取推理、SamplingClient、SamplingParams 和 ModelInput。
- 接入数据集时，读取 HuggingFace datasets 和训练文档。
- 做角色微调、同步/异步 SFT 或 SwanLab 训练记录时，读取 Chat-甄嬛，并参考本 skill 的 `examples/chat-huanhuan.py` 与 `examples/chat-huanhuan-async.py`。
- 训练后要用 OpenAI SDK 部署时，先保存权重，再读取 OpenAI 兼容 API。
- 如果某个页面 404 或看起来过期，打开可视化文档页面，再根据当前导航推导 Markdown 路径。
