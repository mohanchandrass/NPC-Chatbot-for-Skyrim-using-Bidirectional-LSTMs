import json
import numpy as np
import tensorflow as tf
import pickle
import os
import tkinter as tk
from tkinter import ttk, scrolledtext
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageTk
import pygame

# === Initialize pygame mixer for theme music ===
pygame.mixer.init()

# === Constants ===
MAX_SEQUENCE_LENGTH = 30
THEME_SONG_PATH = "skyrim_theme.mp3"  # Optional: comment out if not available

# === Model/Tokenizer Paths ===
MODEL_PATHS = {
    "Baseline BiLSTM": {
        "model": "BiLstm/skyrim_chatbot_final.keras",
        "tokenizer": "BiLstm/tokenizer.pkl"
    },
    "BiLSTM + GloVe": {
        "model": "Glove+LSTM/skyrim_chatbot_glove_final.keras",
        "tokenizer": "Glove+LSTM/tokenizer.pkl"
    },
    # Add more models here if needed
}

# === Globals for model/tokenizer ===
model = None
tokenizer = None

# === Load selected model/tokenizer ===
def load_model_and_tokenizer(selection):
    global model, tokenizer
    model_path = MODEL_PATHS[selection]["model"]
    tokenizer_path = MODEL_PATHS[selection]["tokenizer"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

# === Repeating word checker ===
def has_repeating_words(response_text):
    words = response_text.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return True
    return False

# === Generate response ===
def generate_response(input_text, max_response_length=30, temperature=0.2):
    input_seq = pad_sequences(
        tokenizer.texts_to_sequences([input_text]),
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post'
    )
    predicted_seq = model.predict(input_seq, verbose=0)[0]
    response_text = []

    for i in range(max_response_length):
        word_probs = predicted_seq[i] ** (1 / temperature)
        word_probs /= np.sum(word_probs)
        word_idx = np.random.choice(len(word_probs), p=word_probs)
        word = tokenizer.index_word.get(word_idx, "<UNK>")

        if word in ["<OOV>", "<UNK>", "", "<END>"]:
            break
        response_text.append(word)

    response = " ".join(response_text).strip()
    if has_repeating_words(response) or not response:
        return "I'm sorry, traveler. I do not know much about that."
    return response

class NPCChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Elder Scrolls NPC Chatbot")
        self.root.geometry("600x700")
        self.root.minsize(500, 600)

        # === Theme song ===
        if os.path.exists(THEME_SONG_PATH):
            pygame.mixer.music.load(THEME_SONG_PATH)
            pygame.mixer.music.play(-1)

        # === Layout frame ===
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # === NPC Image ===
        npc_img = Image.open("npc_image.jpeg")
        npc_img = npc_img.resize((300, 300), Image.LANCZOS)
        self.npc_photo = ImageTk.PhotoImage(npc_img)
        self.npc_label = tk.Label(main_frame, image=self.npc_photo)
        self.npc_label.grid(row=0, column=0, pady=(0, 10), sticky="n")

        # === Model dropdown ===
        self.model_dropdown = ttk.Combobox(main_frame, values=list(MODEL_PATHS.keys()), state="readonly")
        self.model_dropdown.grid(row=1, column=0, pady=5, sticky="ew")
        self.model_dropdown.set(list(MODEL_PATHS.keys())[0])
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_change)

        # === Chat box ===
        self.chat_box = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("Arial", 12))
        self.chat_box.grid(row=2, column=0, sticky="nsew", pady=10)
        self.chat_box.insert(tk.END, "NPC: Welcome, traveler! How may I assist you?\n")
        self.chat_box.config(state=tk.DISABLED)

        # === Input + Button frame ===
        input_frame = tk.Frame(main_frame)
        input_frame.grid(row=3, column=0, pady=(0, 10), sticky="ew")
        input_frame.columnconfigure(0, weight=1)

        self.user_input = tk.Entry(input_frame, font=("Arial", 14))
        self.user_input.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.send_button = tk.Button(input_frame, text="Talk", command=self.send_message, font=("Arial", 14))
        self.send_button.grid(row=0, column=1)

        # === Bind enter key ===
        self.root.bind("<Return>", lambda event: self.send_message())

        # === Load default model ===
        self.on_model_change()

    def on_model_change(self, event=None):
        selected = self.model_dropdown.get()
        try:
            load_model_and_tokenizer(selected)
            self.chat_box.config(state=tk.NORMAL)
            self.chat_box.insert(tk.END, f"\n✅ Loaded model: {selected}\n")
            self.chat_box.config(state=tk.DISABLED)
        except Exception as e:
            self.chat_box.config(state=tk.NORMAL)
            self.chat_box.insert(tk.END, f"\n❌ Error loading model: {e}\n")
            self.chat_box.config(state=tk.DISABLED)

    def send_message(self):
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, f"\nYou: {user_text}\n")
        self.user_input.delete(0, tk.END)

        try:
            response = generate_response(user_text)
        except Exception as e:
            response = f"[Error] {str(e)}"
        self.chat_box.insert(tk.END, f"NPC: {response}\n")
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.yview(tk.END)


# === Run the App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = NPCChatbotGUI(root)
    root.mainloop()