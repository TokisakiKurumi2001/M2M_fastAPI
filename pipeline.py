from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import List, Tuple
from split_sentences import SplitSentence


class TranslationPipeline:
    def __init__(self) -> None:
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "transZ/M2M_Vi_Ba")
        self.tokenizer = M2M100Tokenizer.from_pretrained("transZ/M2M_Vi_Ba")
        self.tokenizer.src_lang = "vi"

    def translate(self, sent: str) -> str:
        vi_text = sent
        encoded_vi = self.tokenizer(vi_text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_vi, forced_bos_token_id=self.tokenizer.get_lang_id("ba"))
        ba_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)[0]
        return ba_text

    async def __call__(self, text: str) -> Tuple[List[str], List[str]]:
        vi_paragraphs = text.split('\n')
        ba_paragraphs = []
        for paragraph in vi_paragraphs:
            sents = SplitSentence(paragraph)
            translated_sentences = [self.translate(
                sent) + "." for sent in sents]
            translated_paragraph = " ".join(translated_sentences)
            ba_paragraphs.append(
                translated_paragraph if paragraph != '' else '')
        return vi_paragraphs, ba_paragraphs
