import openvino_genai as ov_genai
import requests
import time
from PIL import Image
from io import BytesIO
import numpy as np
import openvino as ov

model_path = r"models\gemma-3-4b-it-int4-cw-ov"
device = "NPU"
pipe = ov_genai.VLMPipeline(model_path, device)

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
    return ov.Tensor(image_data)

prompt = "What is unusual in this picture?"

url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image_tensor = load_image(url)

def streamer(subword: str) -> bool:
    print(subword, end="", flush=True)
    return False

pipe.start_chat()
start_time = time.perf_counter()
output = pipe.generate(prompt, image=image_tensor, max_new_tokens=100, streamer=streamer)
end_time = time.perf_counter()
generation_time = end_time - start_time
print(f"\nGeneration time: {generation_time:.2f} seconds")
print(f"Output type: {type(output)}")
print(f"Generated text: {output.texts}")
print(f"Generated tokens: {output.perf_metrics.get_num_generated_tokens()}")
print(f"Throughput: {output.perf_metrics.get_throughput().mean} tokens/sec")
print(f"Time per token: {output.perf_metrics.get_tpot().mean} sec")
print(f"Time to first token: {output.perf_metrics.get_ttft().mean} sec")
pipe.finish_chat()
