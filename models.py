from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM

clip_name = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
clip_model = CLIPModel.from_pretrained(clip_name)
clip_processor = CLIPProcessor.from_pretrained(clip_name)

translation_name = "Helsinki-NLP/opus-mt-ru-en"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_name)
