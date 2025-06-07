# PiCASo

**PiCASo** Pi-Integrated Conversational AI for Social Robot with Optimized LLMs. 

As the demand for private, intelligent assistants grows in education and accessibility domains, deploying large language models (LLMs) on low-power devices remains a key challenge. PiCASo addresses this by delivering a fully offline, real-time voice-to-voice conversational AI system that runs entirely on a Raspberry Pi 5.

The pipeline integrates Whisper for transcription, a quantized LLM for response generation, and Piper for voice synthesis, all operating locally without internet or cloud support. Using uniform post-training quantization (2-, 4-, 6-, and 8-bit weights with 8-bit activations), PiCASo achieves up to 4× memory savings and faster inference with minimal accuracy loss.

Integrated into ABii, a commercially available Pi-powered social robot, PiCASo enables private, intelligent interaction in K–12 classrooms and demonstrates the potential of scalable AI at the edge.

PiCASo addresses this by integrating a lightweight voice pipeline consisting of:

* **Whisper ASR** for speech-to-text transcription
* **Quantized LLMs** for local response generation
* **Piper TTS** for voice-based responses
* **NUBIA Score** for semantic response quality evaluation

---

## Paper and Demo

* **Paper Title:** *LLMPi: Optimizing LLMs for High-Throughput on Raspberry Pi*
* **Paper Link:** [https://arxiv.org/abs/2504.02118](https://arxiv.org/abs/2504.02118)
* **Authors:** Mahsa Ardakani, Jinendra Malekar, Ramtin Zand
* **YouTube Demo:** [Watch the Demo](https://www.youtube.com/watch?v=IEH4i1Cv31I)

---

## Installation Instructions

### 1. Clone the Repository (with Submodules)

```bash
git clone --recursive https://github.com/yourusername/PiCASo.git
cd PiCASo
```

---

### 2. Build llama.cpp

```bash
cd llama.cpp
make
cd ..
```

---

### 3. Build whisper.cpp

```bash
cd whisper.cpp
make
cd ..
```

---

### 4. Set Up Python Virtual Environment

```bash
python3.9 -m venv env_name
source env_name/bin/activate
```

---

### 5. Set Up Piper (Text-to-Speech)

Clone and install Piper by following the instructions in its official repository:
🔗 [https://github.com/rhasspy/piper](https://github.com/rhasspy/piper)

---

### 6. Set Up Nubia (Semantic Evaluation)

```bash
git clone https://github.com/wl-research/nubia.git
cd nubia
```

**Edit `requirements.txt`** to include:

```
boto3==1.34.79
botocore==1.34.79
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

### 7. Install Fairseq (required for Nubia)

```bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install .
cd ..
```

---

### 8. Launch LLaMA Server

```bash
cd llama.cpp/build
./bin/llama-server -m /path/to/model --port PORT_NUMBER
```

---

### 9. Run Nubia Server

In a **new terminal**, activate the same virtual environment and run:

```bash
cd PiCASo/nubia
python nubia_server.py
```

> **Note**: If you encounter the following PyTorch error:

```
In PyTorch 2.6, the default for `weights_only` changed from False to True.
```

Edit the following file:

```
./env_name/lib/python3.9/site-packages/fairseq/checkpoint_utils.py
```

Change:

```python
state = torch.load(f, map_location=torch.device("cpu"))
```

To:

```python
state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
```

---

### 10. Run the Main Pipeline

In another terminal, while everything else is running:

```bash
python DACDemo25.py
```


