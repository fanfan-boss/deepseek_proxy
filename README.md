# DeepSeek Proxy for Android Studio AI Assistant

> **English** | [中文](#chinese)

<a id="english"></a>

A local proxy server that solves the **400 Bad Request** error when connecting Android Studio AI Assistant (Gemini API compatible mode) to DeepSeek models.

## Problem

When configuring DeepSeek as an AI model provider in Android Studio via **Gemini API compatible mode**, you may encounter this error:

```
DeepSeek v4 thinking mode: OpenAIChatModel rejects thinking kwarg,
and ThinkingBlock in context causes 400 on subsequent calls
```

### Root Cause

1. **Android Studio sends requests in Gemini API format**, but DeepSeek only provides an OpenAI-compatible interface — the formats are incompatible.
2. **Android Studio injects a `thinking` parameter** (`thinking` kwarg) into requests, which DeepSeek's OpenAI-compatible interface (`OpenAIChatModel`) **does not recognize**, resulting in a 400 error.
3. **Subsequent requests carry a `ThinkingBlock` in the context**, which DeepSeek also rejects, causing cascading 400 failures.

## Solution

This proxy runs locally as a **middle layer between Android Studio and DeepSeek**:

- **Protocol conversion**: Automatically translates Gemini API format requests (`generateContent` / `streamGenerateContent`) to OpenAI-compatible format before forwarding to DeepSeek
- **Parameter filtering**: Strips incompatible `thinking` parameters during conversion
- **Auto-retry on 400**: If the upstream returns 400 (e.g., `reasoning_content` rejected), it disables `thinking` and retries automatically
- **reasoning_content caching**: Intelligently caches reasoning content from tool call messages to prevent cascading 400 errors

## Quick Start

### Prerequisites

- Python 3.9+
- A valid DeepSeek API Key

### Install Dependencies

```bash
pip install flask requests
```

### Run

```bash
python deepseek_proxy.py
```

Default listening address: `http://127.0.0.2:8081`

### Android Studio Configuration

| Field | Value |
|-------|-------|
| **Description** | `deepseek-proxy` |
| **URL** | `http://127.0.0.2:8081` |
| **URL Schema** | `OpenAI-compatible` ✅ |
| **API Key** | Your DeepSeek API Key |
| **Model** | `deepseek-v4-flash` |

**Path**: `File → Settings → Tools → AI → Model Providers` → Add Provider

### Model Selection

| Model | Use Case |
|-------|----------|
| `deepseek-v4-flash` | Fast conversations (recommended for daily use) |
| `deepseek-v4-pro` | Deep reasoning (complex tasks) |

### Gemini Native Mode Support

If you use **Gemini API mode** in Android Studio (instead of OpenAI-compatible mode), this proxy also works:

- Set URL to `http://127.0.0.2:8081`
- No extra configuration needed — automatic bidirectional Gemini ↔ OpenAI conversion

## Verify

Visit `http://127.0.0.2:8081/v1/models` to see the available model list.

Visit `http://127.0.0.2:8081/stats` to check cache hit statistics.

## Logging

Logs are written to `deepseek_proxy.log` in the same directory as the script, containing detailed request/response debug info.

## Project Structure

```
deepseek_proxy/
├── deepseek_proxy.py   # Proxy server main program
├── README.md           # This file
└── deepseek_proxy.log  # Runtime log (auto-generated)
```

---

<a id="chinese"></a>

# DeepSeek Proxy for Android Studio AI Assistant

解决 Android Studio AI Assistant（Gemini API 兼容模式）接入 DeepSeek 时出现的 **400 Bad Request** 问题。

## 问题描述

在 Android Studio 中通过 **Gemini API 兼容模式** 配置 DeepSeek 作为 AI 模型提供商时，会遇到以下错误：

```
DeepSeek v4 thinking mode: OpenAIChatModel rejects thinking kwarg,
and ThinkingBlock in context causes 400 on subsequent calls
```

### 根本原因

1. **Android Studio 以 Gemini API 格式发送请求**，而 DeepSeek 只提供 OpenAI 兼容接口，两者格式不同。
2. **Android Studio 在请求中注入了 `thinking` 参数**（即 `thinking` kwarg），但 DeepSeek 的 OpenAI 兼容接口（`OpenAIChatModel`）**不识别 `thinking` 参数**，导致 400 错误。
3. **后续请求中携带了 `ThinkingBlock` 上下文**，同样被 DeepSeek 拒绝，造成连续 400 失败。

## 解决方案

本代理服务器运行在本地，充当 **Android Studio ↔ DeepSeek 中间层**：

- **协议转换**：将 Gemini API 格式的请求（`generateContent` / `streamGenerateContent`）自动转换为 OpenAI 兼容格式，再转发到 DeepSeek
- **参数过滤**：在转换过程中剔除不兼容的 `thinking` 参数，避免 DeepSeek 拒接
- **400 自动重试**：若上游返回 400（如 `reasoning_content` 被拒），自动禁用 `thinking` 参数后重试
- **reasoning_content 缓存**：智能缓存工具调用消息的推理内容，避免连锁 400

## 快速开始

### 前提

- Python 3.9+
- 有效的 DeepSeek API Key

### 安装依赖

```bash
pip install flask requests
```

### 启动

```bash
python deepseek_proxy.py
```

默认监听 `http://127.0.0.2:8081`。

### Android Studio 配置

| 字段 | 值 |
|------|-----|
| **Description** | `deepseek-proxy` |
| **URL** | `http://127.0.0.2:8081` |
| **URL Schema** | `OpenAI-compatible` ✅ |
| **API Key** | 你的 DeepSeek API Key |
| **Model** | `deepseek-v4-flash` |

**配置路径**：`File → Settings → Tools → AI → Model Providers` → 添加 Provider

### 模型选择

| 模型 | 用途 |
|------|------|
| `deepseek-v4-flash` | 快速对话（推荐日常使用） |
| `deepseek-v4-pro` | 深度推理（复杂任务） |

### Gemini 原生支持

如果你在 Android Studio 中选择的是 **Gemini API 模式**（而非 OpenAI 兼容模式），本代理同样支持：

- URL 填 `http://127.0.0.2:8081`
- 无需额外配置，自动完成 Gemini ↔ OpenAI 协议双向转换

## 验证运行状态

访问 `http://127.0.0.2:8081/v1/models` 查看可用模型列表。

访问 `http://127.0.0.2:8081/stats` 查看缓存命中统计。

## 日志

日志文件生成在代理脚本同目录下的 `deepseek_proxy.log`，包含详细的请求/响应调试信息。

## 项目结构

```
deepseek_proxy/
├── deepseek_proxy.py   # 代理服务器主程序
├── README.md           # 本文件
└── deepseek_proxy.log  # 运行时日志（自动生成）
```
