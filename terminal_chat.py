import openvino_genai as ov_genai
import openvino as ov
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import sys


model_path = r"models\gemma-3-4b-it-int4-cw-ov"
device = "NPU"
max_token_limit = 3000

pipe = ov_genai.VLMPipeline(model_path, device)


def main():

#Tokens to use if specified or not by system arugumets
    max_token_use = max_token_limit
    if len(sys.argv) > 1:
        max_token_use = int(sys.argv[1])

#Storage: a record of everything used is the chat.
    history = []
    chat_history = ov_genai.ChatHistory()
    chat_images = []

#main loop body
    while True:
        turn_image = []                                         #Temporary storage of images for her context
        try:
            user_input = input("You: ").strip()
        except EOFError:                                        #Leave from chat (For Window's ctrl+z), (For Linux ctrl+d)
            sys.exit()
        if len(user_input) < 1:                                 #If user doesn't type anything chat prompt again
            continue
        if "|" in user_input:                                   #If user want's image reasoning use this "|" 
            prompt , image = user_input.split("|")
            user_input = prompt.strip()
            image_file = image.strip()
            tensor_image = load_image(image_file)
            chat_images.append(tensor_image)
            turn_image.append(tensor_image)
        add_message(history,chat_history,"user",user_input)
        print("Gemma: ", end="")

        #OpenVino.VLMpipeline format = prompt, Chat_History = Context recorded, images - vector converted, max_new_tokens - token generation limit for that prompt, streamer - for smooth text generation as gemma generates the tokens. 
        output = pipe.generate(chat_history,images=turn_image, max_new_tokens= max_token_use, streamer=streamer)
        
        print()
        add_message(history,chat_history,"assistant",output.texts[0])


def streamer(subword: str) -> bool:                             #Stole from OpenVino's inference don't know how it works :}. It works for smooth text generation as gemma generates the tokens.
    print(subword, end="", flush=True)
    return False

def add_message(history,chat_history,role,content):             #Gives context and memory to gemma using her own format.
    messages = {"role" : role , "content" : content}
    history.append(messages)
    chat_history.append(messages)

def load_image(image):                                          #Stole from OpenVino but know how it works.
    if image.startswith("https:") or image.startswith("http:"):
        response = requests.get(image)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image).convert("RGB")
                                                                #Image converted to openvino format.
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
    return ov.Tensor(image_data)


if __name__ == "__main__":
    main()