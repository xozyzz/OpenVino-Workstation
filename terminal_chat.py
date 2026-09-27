import openvino_genai as ov_genai
import sys
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import openvino as ov

model_path = r"models\gemma-3-4b-it-int4-cw-ov"
device = "NPU"
max_token_limit = 3000

pipe = ov_genai.VLMPipeline(model_path, device)


def main():
    max_token_use = max_token_limit
    if len(sys.argv) > 1:
        max_token_use = int(sys.argv[1])
    history = []
    chat_history = ov_genai.ChatHistory()
    chat_images = []
    while True:
        turn_image = []
        try:
            user_input = input("You: ").strip()
        except EOFError:
            sys.exit()
        if len(user_input) < 1:
            continue
        if "|" in user_input:
            prompt , image = user_input.split("|")
            user_input = prompt.strip()
            image_file = image.strip()
            tensor_image = load_image(image_file)
            chat_images.append(tensor_image)
            turn_image.append(tensor_image)
        add_message(history,chat_history,"user",user_input)
        print("Gemma: ", end="")
        output = pipe.generate(chat_history,images=turn_image, max_new_tokens= max_token_use, streamer=streamer)
        print()
        add_message(history,chat_history,"assistant",output.texts[0])


def streamer(subword: str) -> bool:
    print(subword, end="", flush=True)
    return False

def add_message(history,chat_history,role,content):
    messages = {"role" : role , "content" : content}
    history.append(messages)
    chat_history.append(messages)

def load_image(image):
    if image.startswith("https:") or image.startswith("http:"):
        response = requests.get(image)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
    return ov.Tensor(image_data)


if __name__ == "__main__":
    main()