#!/usr/bin/env python3
import logging
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import sys
import subprocess
import re
import json
import time
import requests
import os

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
sys.path.append('/home/icas/Downloads/nubia/nubia_score')

MIC_RECORD_SECONDS = 7
RECORDED_AUDIO = "/home/icas/Desktop/mic_input.wav"
WHISPER_MODEL = "/home/icas/Desktop/whisper.cpp/models/ggml-tiny.en.bin"
WHISPER_MAIN = "/home/icas/Desktop/whisper.cpp/main"
GEOGRAPHY_FILE = "/home/icas/Downloads/selected_questions.json"
MATH_FILE = "/home/icas/Downloads/selected_metamath_questions.json"
PIPER_EXECUTABLE = "/home/icas/Downloads/piper/piper"
PIPER_VOICE = "/home/icas/Downloads/piper/voices/en_US/amy-low/model.onnx"
PIPER_CONFIG = "/home/icas/Downloads/piper/voices/en_US/amy-low/config.json"

MAX_TOKENS = 170
TEMPERATURE = 0.2
STOP_SEQUENCES = ["\n", ".", "?"]
CONTEXT_LIMIT = 2000
TIMEOUT_DURATION = 90

model_map = {
    "8080": "phi-3_Q8_0",
    "8081": "phi-3_Q2_k",
    "8082": "phi-3_FP",
    "8083": "phi-3_Q4_0"
}

def load_context(path, question):
    with open(path) as f:
        data = json.load(f)
    question_keywords = set(question.lower().split())
    for item in data:
        if item.get("query", "").lower() == question.lower():
            return item.get("context", item.get("query", ""))
    best_match = None
    highest_score = 0
    for item in data:
        context_keywords = set(item.get("context", "").lower().split())
        score = len(question_keywords & context_keywords)
        if score > highest_score:
            best_match = item.get("context", item.get("query", ""))
            highest_score = score
    return best_match or data[0].get("context", data[0].get("query", ""))

def speak_with_piper(text):
    try:
        result = subprocess.run(
            [PIPER_EXECUTABLE, "--model", PIPER_VOICE, "--config", PIPER_CONFIG],
            input=text.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        for line in result.stdout.splitlines():
            decoded = line.decode()
            if decoded.endswith(".wav") and os.path.exists(decoded):
                subprocess.run(["aplay", decoded], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(decoded)
                break
    except Exception as e:
        print(f"Piper TTS error: {e}")

def record_microphone_safe():
    try:
        subprocess.run(["arecord", "-D", "plughw:2,0", "-f", "cd", "-t", "wav", "-d", str(MIC_RECORD_SECONDS),
                        "-r", "16000", "-c", "1", RECORDED_AUDIO])
        return True
    except Exception as e:
        print(f"Microphone error: {e}")
        return False

def transcribe_audio(audio_file):
    print("Transcribing audio ...")
    try:
        result = subprocess.run([WHISPER_MAIN, "-m", WHISPER_MODEL, "-f", audio_file, "-t", "4"],
                                capture_output=True, text=True, timeout=60)
        return result.stdout
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def clean_transcription(raw_text):
    if not raw_text:
        return ""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    full_text = ""
    for line in lines:
        clean = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> .*?\]', '', line)
        clean = re.sub(r'^(whisper_|main:|\([^)]*\))', '', clean).strip()
        if clean:
            full_text += clean + " "
    return full_text.strip()

def stream_llama_server(prompt, port):
    url = f"http://127.0.0.1:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
        "stop": STOP_SEQUENCES,
        "repeat_penalty": 1.1
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=TIMEOUT_DURATION) as response:
            buffer = []
            tps = None
            full_response = ""
            for line in response.iter_lines():
               # print("line", line)
                if not line:
                    continue
                line_str = line.decode()
                if line_str.startswith("data:"):
                    line_str = line_str[5:].strip()
                try:
                    data = json.loads(line_str)
                   # print("data",data)
                except Exception:
                    continue
                content = data.get("content", "")
               # if "tps" in data:
               #     tps = data["tps"]
                if content:
                    words = content.split()
                    buffer.extend(words)
                    while len(buffer) >= 4:
                        chunk = " ".join(buffer[:4])
                        print(chunk, end=" ", flush=True)
                        speak_with_piper(chunk)
                        full_response += chunk + " "
                        buffer = buffer[4:]
            tps=data['timings']['predicted_per_second']
            if buffer:
                chunk = " ".join(buffer)
                print(chunk, end=" ", flush=True)
                speak_with_piper(chunk)
                full_response += chunk + " "
            print()
            return full_response.strip(), tps
    except Exception as e:
        print(f"LLM streaming error: {e}")
        return "", None

def get_nubia_score(prompt, response):
    try:
        url = "http://127.0.0.1:9090/score"
        payload = {"text1": prompt, "text2": response}
        r = requests.post(url, json=payload, timeout=100)
        #print("r.json", r.json())
        return r.json().get("score", None)
    except Exception as e:
        print(f"NUBIA server error: {e}")
        return None

def main_for_port(port, question, context):
    model_name = model_map.get(port, "Unknown Model")
    print("============================================")
    print(f"-- Running model ({model_name})")
    print("============================================")
    prompt = f"""<|system|>
Use the given context and answer the question.

<|user|>
Context: {context[:CONTEXT_LIMIT]}
Question: {question}

<|assistant|>"""

    response, tps = stream_llama_server(prompt, port)
    score = get_nubia_score(prompt, response)
    print(f"\nTPS (tokens/sec) reported by server: {tps if tps is not None else 'N/A'}")
    print(f"NUBIA Score: {score}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 demo.py [math|geography] [model_port]")
        sys.exit(1)

    dataset = sys.argv[1].lower()
    port = sys.argv[2]
    if dataset == "math":
        context_path = MATH_FILE
        print("Math context selected.\n")
    elif dataset == "geography":
        context_path = GEOGRAPHY_FILE
        print("Geography context selected.\n")
    else:
        print("Invalid dataset. Use 'math' or 'geography'")
        sys.exit(1)

    if port not in model_map:
        print("Invalid model port.")
        sys.exit(1)

    try:
        if not record_microphone_safe():
            sys.exit(1)

        raw = transcribe_audio(RECORDED_AUDIO)
        question = clean_transcription(raw)

        if not question:
            print("No transcription found.")
            sys.exit(1)

        print(f"\nYou asked: {question}\n")
        with open("last_question.txt", "w") as f:
            f.write(question)
        #print("start")
        #load_context_time_start = time.time()
        context = load_context(context_path, question)
        #load_context_end_time = time.time()
        #print("end", load_context_end_time - load_context_time_start)
        main_for_port(port, question, context)

        if os.path.exists(RECORDED_AUDIO):
            os.remove(RECORDED_AUDIO)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
    finally:
        print("\nLLM response generated.")
        sys.exit(0)

if __name__ == "__main__":
    main()
