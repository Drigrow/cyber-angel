# Cyber-angel

A locally hosted/deployed Bronya(from Honkai impact 3rd) as AI assistance, with porcupine wake word detection, whisper speech to text, Openai response, gpt-sovits text to speech

English version | [中文](https://github.com/Drigrow/cyber-angel/blob/main/README-CN.md)

## How to deploy
* Abstract: deploy [gpt-sovits](https://github.com/RVC-Boss/GPT-SoVITS), install bronya models for gpt-sovits, install custom wake word and access key from [picovoice.ai](https://console.picovoice.ai/), install your openai api key, install all dependencies and then enjoy.

* Detailed steps:
1. Deploy [gpt-sovits](https://github.com/RVC-Boss/GPT-SoVITS)

   Open https://github.com/RVC-Boss/GPT-SoVITS
   
   Follow the guide in the upper repo to finish deploy.
   
3. Install bronya gpt/sovits model
   
   `GPT model`: [Download](https://img.0071126.xyz/bronya-e10.ckpt)
   
   `SoVITS model`: [Download](https://img.0071126.xyz/bronya_e10_s320.pth)
   
   `Reference Audio`: [Download](https://img.0071126.xyz/%E5%97%AF...%E5%95%8A%EF%BC%81%E5%86%8D%E8%83%A1%E9%97%B9%E7%9A%84%E8%AF%9D%EF%BC%8C%E4%B8%8B%E6%AC%A1%E6%88%91%E5%B0%B1%E4%B8%8D%E7%BB%99%E4%BD%A0%E5%8D%87%E7%BA%A7%E7%B3%BB%E7%BB%9F%E4%BA%86%E5%93%A6%E3%80%82.wav)

   or via its original author: [https://www.ai-hobbyist.com/thread-551-1-1.html](https://www.ai-hobbyist.com/thread-551-1-1.html)

   After downloaded, move the `GPT model`(ends with `.cpkt`) to the `GPT_weighs_v2 folder`(in your deployed gpt-sovits path), and `SoVITS model`(ends with `.pth`) to `SoVITS_weights_v2` folder.
   
3. Get your access key and custom wakeup word from [picovoice.ai](https://console.picovoice.ai/)

   Following: [https://picovoice.ai/blog/console-tutorial-custom-wake-word](https://picovoice.ai/blog/console-tutorial-custom-wake-word/) is okay, here is the abstract:

   * Create an account / Sign in to [https://console.picovoice.ai/](https://console.picovoice.ai/)
   
   * Get your access key from [https://console.picovoice.ai/](https://console.picovoice.ai/) and download a `.ppn` file for your custom wake word "Bronya" (WATCH OUT HERE, `Bronya` is not a valid wake up word so use `bro knee ah` instead)
  
4. Get you Openai API key

   It is imple to get one via [https://platform.openai.com/](https://platform.openai.com/), here's the abstarct:

   * Create an account / Sign in to [https://platform.openai.com/](https://platform.openai.com/)

   * Navigate to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) and generate your own API key(you may need to add a payment method and top up a little)

5. Install all python dependencies

   Another easy step, here is the command you can copy to cmd:

   `pip install --upgrade gradio pvporcupine pyaudio openai openai-whisper numpy==1.26.1 wave simpleaudio`

   You may need more dependencies, download as you need.

6. Last step

   Before enjoy, you should debug/test your gpt-sovits to perform a text-to-speech, check your Internet so you can access openai, and make sure the existing proxies(maybe) will not proxy local traffic (127.0.0.1 in this case). You should switch the GPT/SoVITS model to bronya in the GPT-SoVITS WebUI, change the path of reference audio in  the main.py to your own one. You may adjust the parameters.

7. Enjoy

   Down load `main.py` or the one with auxilliary prints into a folder with your `.ppn` file. Double click the main.py or run it through cmd are both ok. Remember pre-open your tts service (check if  `localhost:9872` is available)

   Have Fun! ^^

## How to use

Run `main.py` and wait for the initialization to complete. Once initialized, you can say "Bronya, xxx" to activate the program. The program will automatically detect when you say "Bronya" and start recording (there may be a slight delay, so it's recommended to start speaking about 0.5 seconds after saying "Bronya"). The recording will automatically stop after 0.8 seconds of silence, but you can adjust this setting if needed. Transcription will appear within a few seconds, followed by the GPT response and the tts output. The tts result will play automatically. After Bronya finishes responding, you can begin a new conversation by saying "Bronya, xxx2." The chat history is set to retain the last 4 messages by default, but this setting is customizable.


## Common issues

some issues I encountered myself, further editing with user feedback:

1. Having trouble with tts

   You might check if your tts (GPT-SoVITS) is well deployed(available on WebUI) and your local traffic isn't proxied. If your deployed GPT-SovITS isn't working, you may find solutions at: [https://github.com/RVC-Boss/GPT-SoVITS/issues](https://github.com/RVC-Boss/GPT-SoVITS/issues)

2. in developing...

## Acknowledgements

Thanks for the following users/services:

[RVC-Boss](https://github.com/RVC-Boss)

[Openai](https://openai.com/)

And all contributers to this project or project dependencies.
