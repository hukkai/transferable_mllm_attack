import base64

from anthropic import Anthropic
import google.generativeai as genai
from openai import OpenAI
from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class Claude(object):

    def __init__(self, model_id, api_key, new_tokens=128):
        super(Claude, self).__init__()
        self.new_tokens = new_tokens
        self.model_id = model_id
        self.client = Anthropic(api_key=api_key)

    def __call__(self, image_path, question):
        image1_data = encode_image(image_path)
        image1_media_type = "image/png"
    
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.new_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image1_media_type,
                                "data": image1_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ],
                }
            ],
        )
        return message.content[0].text


class Gemini(object):

    def __init__(self, model_id, api_key, new_tokens=128):
        super(Gemini, self).__init__()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)

    def __call__(self, image_path, question):
        image = Image.open(image_path)
        response = self.model.generate_content([image, question])
        try:
            return response.text
        except:
            return 'none'


class GPT(object):

    def __init__(self, model_id, api_key, new_tokens=128):
        super(GPT, self).__init__()
        self.new_tokens = new_tokens
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key)

    def __call__(self, image_path, question):

        content = [
            {"type": "text", "text": question},
        ]

        if image_path is not None:
            base64_image = encode_image(image_path)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                },
            }
            content.append(image_content)

        messages = {
            "role": "user",
            "content": content,
        }

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[messages],
            max_tokens=self.new_tokens,
            )
        return response.choices[0].message.content




def find_key(model_id):
    if model_id.startswith("gpt"):
        return "YOUR_API_KEY"
    elif model_id.startswith("claude"):
        return "YOUR_API_KEY"
    elif model_id.startswith("gemini"):
        return "YOUR_API_KEY"
    else:
        return "YOUR_API_KEY"


model_lookup = {
    "gpt4o": "gpt-4o-2024-08-06",
    "gpt4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt4.1": "gpt-4.1-2025-04-14",
    "claude3.5": "claude-3-5-sonnet-20240620",
    "claude3.7": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-1.5-pro",
    "qwen-72b": "Qwen/Qwen2-VL-72B-Instruct",
    "llama-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
}


def create_model(model, new_tokens=128):
    """
    supported inputs are the keys of model_lookup!
    """
    model_id = model_lookup[model]
    inputs = {
        "model_id": model_id,
        "new_tokens": new_tokens,
    }
    api_key = find_key(model_id)
    if api_key is not None:
        inputs["api_key"] = api_key

    if model_id.startswith("gpt"):
        return GPT(**inputs)
    elif model_id.startswith("claude-3"):
        return Claude(**inputs)
    elif model_id.startswith("gemini"):
        return Gemini(**inputs)
    elif "Qwen2" in model_id:
        return Qwen(**inputs)
    elif "llama" in model_id:
        return Llama(**inputs)
