# Cyber-angel

一个AI助手，角色模拟布洛妮娅（出自《崩坏3》），支持Porcupine唤醒词检测、Whisper语音转文字、OpenAI响应及GPT-SoVITS语音合成。

本文也可阅读于(推荐，github上内容可能会忘了更新)：[https://blog.drigrow.us.kg/](https://blog.drigrow.us.kg/2025/01/19/%e7%bb%8f%e9%aa%8c%e5%88%86%e4%ba%ab%e8%b5%9b%e5%8d%9a%e9%b8%ad%e9%b8%ad/)

[English version](https://github.com/Drigrow/cyber-angel/blob/main/README.md) | 中文

## 如何部署
* 概述：部署 [gpt-sovits](https://github.com/RVC-Boss/GPT-SoVITS)，安装gpt-sovits的布洛妮娅模型，设置自定义唤醒词并获取 [picovoice.ai](https://console.picovoice.ai/) 的访问密钥，获取OpenAI API密钥，安装所有依赖并开始使用。

* 详细步骤：
  
1. 部署 [gpt-sovits](https://github.com/RVC-Boss/GPT-SoVITS)

   打开：https://github.com/RVC-Boss/GPT-SoVITS
   
   
   按照上述仓库中的指南完成部署。
   

2. 安装布洛妮娅GPT/SoVITS模型
   

   `GPT模型`：[下载](https://img.0071126.xyz/bronya-e10.ckpt)
   
   
   `SoVITS模型`：[下载](https://img.0071126.xyz/bronya_e10_s320.pth)
   
   
   `参考音频`：[下载](https://img.0071126.xyz/%E5%97%AF...%E5%95%8A%EF%BC%81%E5%86%8D%E8%83%A1%E9%97%B9%E7%9A%84%E8%AF%9D%EF%BC%8C%E4%B8%8B%E6%AC%A1%E6%88%91%E5%B0%B1%E4%B8%8D%E7%BB%99%E4%BD%A0%E5%8D%87%E7%BA%A7%E7%B3%BB%E7%BB%9F%E4%BA%86%E5%93%A6%E3%80%82.wav)
   

   或访问原作者网站：[https://www.ai-hobbyist.com/thread-551-1-1.html](https://www.ai-hobbyist.com/thread-551-1-1.html)
   

   下载完成后，将 `GPT模型`（以 `.ckpt` 结尾）移动到您部署路径中的 `GPT_weights_v2` 文件夹，将 `SoVITS模型`（以 `.pth` 结尾）移动到 `SoVITS_weights_v2` 文件夹。
   

3. 从 [picovoice.ai](https://console.picovoice.ai/) 获取访问密钥和自定义唤醒词
 

   参考：[https://picovoice.ai/blog/console-tutorial-custom-wake-word](https://picovoice.ai/blog/console-tutorial-custom-wake-word)，以下为概要：
   

   * 创建一个账户或登录 [https://console.picovoice.ai/](https://console.picovoice.ai/)
   
   
   * 在 [https://console.picovoice.ai/](https://console.picovoice.ai/) 获取您的访问密钥并下载自定义唤醒词文件（`.ppn`），唤醒词为“Bronya”（注意，“Bronya” 不是有效唤醒词，因此使用类似“bro knee ah”的发音代替）。
  

4. 获取您的OpenAI API密钥

   可以通过 [https://platform.openai.com/](https://platform.openai.com/) 获取，以下为概要：

   * 创建一个账户或登录 [https://platform.openai.com/](https://platform.openai.com/)

   * 前往 [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)，生成您的API密钥（可能需要添加支付方式并充值少量金额）。

5. 安装所有Python依赖

   这是一个简单步骤，您可以复制以下命令到命令行运行：

   `pip install --upgrade gradio pvporcupine pyaudio openai==0.28 openai-whisper numpy==1.26.1 wave simpleaudio`

   可能需要额外安装依赖，按需下载。

6. 最后一步

   在开始使用之前，请调试/测试您的gpt-sovits以执行文本转语音操作，确保您的互联网连接正常并且现有代理（如果有）不会代理本地流量（例如 127.0.0.1）。在GPT-SoVITS的WebUI中切换GPT/SoVITS模型为布洛妮娅，修改 `main.py` 中参考音频的路径为您自己的路径。您可能需要调整参数。

7. 开始使用

   将 `main.py` 文件下载到包含 `.ppn` 文件(自定义唤醒词)的文件夹中。双击 main.py 或通过 cmd 运行它都可以。记得预先打开你的 tts 服务（检查 `localhost:9872` 是否可用）

   玩得开心！^^

## 如何使用

运行 `main.py` 并稍等片刻完成初始化。初始化完成后，你可以说“Bronya, xxx”来激活程序。程序会自动检测到你说“Bronya”并开始录音（可能会有轻微延迟，因此建议在说“Bronya”后等待约 0.5 秒再开始讲话）。录音将在检测到 0.8 秒的静音后自动停止，但你可以根据需要调整此设置。几秒钟后，转录结果会显示出来，随后程序会生成 GPT 的回复以及 TTS（文本转语音）的结果，TTS 结果将自动播放。在 Bronya 回复结束后，你可以通过再次说“Bronya, xxx2”开始新的对话。默认情况下，程序会保留最近 4 条对话记录，你可以根据需要修改这个设置。

## 常见问题

以下是我个人遇到的一些问题，将根据用户反馈进一步更新：

1. 语音合成（TTS）问题

   您可以检查您的语音合成（GPT-SoVITS）是否部署成功（是否可通过WebUI访问），以及本地流量是否未被代理。如果您的GPT-SoVITS部署失败，可以在 [https://github.com/RVC-Boss/GPT-SoVITS/issues](https://github.com/RVC-Boss/GPT-SoVITS/issues) 找到解决方案。

2. 开发中...

## 致谢

感谢以下用户/服务：

[RVC-Boss](https://github.com/RVC-Boss)

[OpenAI](https://openai.com/)

以及所有为此项目或其依赖项目做出贡献的开发者。
