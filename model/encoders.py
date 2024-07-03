class TextEncoder:
    def __init__(self):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def encode(self, x):
        inputs = self.tokenizer(x, padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        outputs = self.model(**inputs)
        text_embeds = outputs.text_embeds
        return text_embeds

class VisionEncoder:
    def __init__(self):
        from transformers import AutoProcessor, CLIPVisionModelWithProjection
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        return image_embeds