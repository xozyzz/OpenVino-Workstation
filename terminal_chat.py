import openvino_genai as ov_genai
import sys

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
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            sys.exit()
        if len(user_input) < 1:
            continue
        add_message(history,chat_history,"user",user_input)
        print("Gemma: ", end="")
        output = pipe.generate(chat_history, max_new_tokens= max_token_use, streamer=streamer)
        add_message(history,chat_history,"assistant",output.texts[0])


def streamer(subword: str) -> bool:
    print(subword, end="", flush=True)
    return False

def add_message(history,chat_history,role,content):
    messages = {"role" : role , "content" : content}
    history.append(messages)
    chat_history.append(messages)


if __name__ == "__main__":
    main()