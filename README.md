### Audio Encoder
using music2latent
#### Usage
1. download music2latent.pt at https://huggingface.co/SonyCSLParis/music2latent/blob/main/music2latent.pt
2. pip install requirements.txt
3. 更改 retrieval.py 第32行至第36行的路徑
4. python retrieval.py

 ### Music To Text
 #### Usage
 1. run in Colab https://colab.research.google.com/drive/1UDzBzIA_3PpXHJ7ITkV8LrKsA67LmbXT?usp=sharing

### Text To Music
#### Usage
1. run in Colab https://colab.research.google.com/drive/1KbPPzMvpJ7_WGcW9lXbxJ3Wjgng1ZNOS?usp=sharing

### Estimation
#### Usage
##### CLAP
1. pip install requirements.txt
2. 更改 clap.py 第72至77行的路徑
3. python clap.py

##### AudioBox
1. pip install audiobox_aesthetics
2. 至 https://huggingface.co/facebook/audiobox-aesthetics 下載 checkpoint.pt
3. 更改 audiobox.py 第4至5行的路徑
4. python audiobox.py  生成要檢測的音樂路徑.jsonl檔
5. CUDA_VISIBLE_DEVICES="" audio-aes "audio_paths.jsonl" --batch-size 4 > "output.jsonl" --ckpt "checkpoint.pt路徑"

### File
#### mtt
提供所有生成出來的prompt

#### Prompt_short_gen
MTT以short prompt得出來的結果，放入TTM生出來的音樂

#### Prompt_normal_gen
MTT以normal prompt得出來的結果，放入TTM生出來的音樂

#### Prompt_detail_gen
MTT以detail prompt得出來的結果，放入TTM生出來的音樂
 
