# 部署到公网（手机可访问 + 简历可放链接）

目标：拿到一个永久 https 地址，面试官点开就能试。

推荐 **Streamlit Community Cloud**：免费、永久、自动 https、不需要服务器。

---

## 这个项目比普通项目多一个要点：API Key 怎么处理

上一个项目（market_pulse）部署时只需传数据文件。这个项目要调模型，涉及密钥。

**三条铁律：**

| 规则 | 原因 |
|---|---|
| 密钥**绝不写进代码** | 代码推到公开仓库，等于把 key 送给全世界爬虫 |
| 密钥**绝不提交到 git** | 即使后来删掉，git 历史里仍然能翻出来 |
| 网页上让**用户自己填** key | 你不需要替所有人付模型费用 |

本项目的做法：

- 密钥只存在 `st.session_state`（浏览器会话内存），刷新即失效
- 不写日志、不落盘、不进任何持久化存储
- 用户在自己的界面填自己的 key，你只提供工具

**这对简历反而是加分项**：说明你知道密钥该怎么管。

### 如果你想自己预置一个 key（不推荐）

Streamlit Cloud 支持 Secrets 管理，但你一旦预置：

- 任何访问者都能消耗你的额度（有人会扫端口刷 key）
- 免费额度几小时就能被刷光

真要做公开演示，正确姿势是**限流 + 限额**，那需要额外后端，超出这个项目范围。所以让它"用户带 key"最省事也最安全。

---

## 第一步：确认本地能跑

```powershell
cd C:\Users\28456\Desktop\deepseek\screenplay_agent
.\.venv\Scripts\streamlit.exe run app.py
```

浏览器打开 <http://localhost:8501>，应该看到：

- 左侧边栏：服务商选择、模型名、API Key 输入框
- 主区域：①输入剧本 ②解析，两个步骤

先不填 key 点「开始解析」试试——它会走离线模式并**明确警告**情绪与镜头提示会全是「未判定」。这是有意设计：让用户立刻理解离线模式的局限。

---

## 第二步：推到 GitHub

```powershell
cd C:\Users\28456\Desktop\deepseek\screenplay_agent

git init
git add .
git status --short          # 确认没有 .venv、out/*.xlsx 被加进来
git commit -m "feat: 剧本结构化解析 Agent - 网页版"
```

在 GitHub 建一个**公开**仓库，名字建议 `screenplay-agent`，**不要勾选** README/gitignore 初始化（会与本地冲突）。

```powershell
git remote add origin https://github.com/<你的用户名>/screenplay-agent.git
git branch -M main
git push -u origin main
```

> ⚠️ **推送前必须检查**：`git status` 里不能出现 `.venv/`、`out/*.xlsx`、`.env`。
> `.gitignore` 已经配好，但养成看一眼的习惯。

---

## 第三步：部署

1. 打开 <https://share.streamlit.io>
2. **Create app** → **Deploy a public app from GitHub**
3. 填写：

   | 字段 | 值 |
   |---|---|
   | Repository | `<你的用户名>/screenplay-agent` |
   | Branch | `main` |
   | **Main file path** | **`streamlit_app.py`** |
   | App URL | 如 `screenplay-agent` |

4. 点 **Deploy**，等 2–4 分钟

成功后地址形如：

```
https://screenplay-agent.streamlit.app
```

手机浏览器直接打开即可使用。这个地址可以写进简历。

---

## 第四步：验证清单

部署完成后**自己先用手机打开**，逐项确认：

| 检查项 | 预期 |
|---|---|
| 页面能打开 | 显示「🎬 剧本结构化解析」 |
| 侧边栏有 API Key 输入框 | 是密码框（输入显示为圆点） |
| 点「示例剧本」标签 → 使用这个示例 | 文本填入 |
| 不填 key 点解析 | 出现黄色警告，结果里情绪全是「未判定」 |
| 填自己的 key 再解析 | 1–3 分钟后出结果，情绪和镜头提示有内容 |
| 质控区漏抽率 | 通常 0%–5% |
| 勾选「我已复核」后 | 三个下载按钮出现 |

**如果卡在"解析中"很久**：免费版 CPU 较弱，逐场调用模型确实慢。示例剧本 3 场约 30–60 秒，80 场长剧本可能需要 10 分钟以上——这是免费版限制，不是 bug。

---

## 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 构建失败 `ModuleNotFoundError` | requirements 漏包 | 检查是否纯 ASCII；本项目的已知坑见下 |
| `pydantic` 报错 | 版本差异 | 项目已做 v1/v2 兼容层，不应出现 |
| 解析超时 | 免费版 CPU 限制 | 减少剧本长度，或本地跑 |
| 页面提示"质控未通过" | 所有场次都抽取失败 | 多为 key 无效或额度耗尽，换 key 试 |
| 手机上传文件失败 | 文件过大 | Streamlit 默认限 200MB，正常剧本远小于此 |

### 一个踩过的坑：requirements.txt 的编码

上一个项目部署时踩过：中文注释被写成乱码，且注释和第一条依赖粘在同一行，导致 Linux 上 pip 报 `Invalid requirement`。

本项目 `requirements.txt` **保持纯 ASCII**，并且不要用中文注释。改依赖时请照做。

---

## 备选：局域网访问（零配置）

手机连同一 WiFi，打开：

```
http://<你电脑的局域网IP>:8501
```

查自己的 IP：

```powershell
python -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('8.8.8.8',80));print(s.getsockname()[0])"
```

打不开多半是防火墙：

```powershell
# 管理员 PowerShell
New-NetFirewallRule -DisplayName "Streamlit 8501" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```

**注意**：局域网 IP 会随网络环境变化（本项目开发期间就从 `10.10.205.79` 变成过 `10.135.195.232`），所以它只适合临时用，不适合写简历。

---

## 关于成本

以 DeepSeek 为例，示例剧本（967 字、3 场）的消耗：

- 分场：跳过（规则识别到了场标）
- 逐场抽取：3 次调用
- 人物归一：1 次
- 抽样复核：3 次

约 7 次调用，合计约 2 万 token。DeepSeek 定价约 ¥1/百万 token，**单次成本不到 1 分钱**。

一个 80 场的完整短剧约 40–60 次调用，成本约 3–5 分钱。所以让面试官自己带 key 完全合理，你也可以放心在简历上放链接。
