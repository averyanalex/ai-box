from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM

clip_name = "google/siglip-so400m-patch14-384"
clip_model = AutoModel.from_pretrained(clip_name)
clip_processor = AutoProcessor.from_pretrained(clip_name)

translation_name = "Helsinki-NLP/opus-mt-ru-en"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_name)
