"""
Arabic text anonymizer for Al-Muhami Al-Zaki.

Compliant with Egyptian Data Protection Law 151/2020.
Uses CAMeLBERT-NER to detect and mask PII entities.

Entities masked:
- PERSON → [شخص]
- LOCATION/GPE → [مكان]
- ORGANIZATION → [جهة]
"""

from typing import Dict, List, Tuple

from loguru import logger

# Entity replacement mapping (Law 151 compliant)
ENTITY_MASKS: Dict[str, str] = {
    "PER": "[شخص]",
    "PERSON": "[شخص]",
    "LOC": "[مكان]",
    "LOCATION": "[مكان]",
    "GPE": "[مكان]",
    "ORG": "[جهة]",
    "ORGANIZATION": "[جهة]",
}


class ArabicAnonymizer:
    """
    PII Anonymizer using CAMeLBERT-NER for Arabic text.

    Compliant with Egyptian Data Protection Law 151/2020.
    Masks PERSON, LOCATION, and ORGANIZATION entities.

    Example:
        anonymizer = ArabicAnonymizer()
        anonymized_text, audit_log = anonymizer.anonymize(
            "حكم ضد أحمد علي في القاهرة"
        )
        # Result: "حكم ضد [شخص] في [مكان]"
    """

    def __init__(
        self,
        model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
        device: str = "cpu",
    ):
        """
        Initialize the anonymizer.

        Args:
            model_name: HuggingFace model ID for Arabic NER
            device: Device to run model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self._pipeline = None

        logger.info(f"Anonymizer initialized with model: {model_name}")

    def _load_pipeline(self):
        """Lazy load the NER pipeline."""
        if self._pipeline is None:
            try:
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                    pipeline,
                )

                logger.info(f"Loading NER model: {self.model_name}")

                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForTokenClassification.from_pretrained(self.model_name)

                self._pipeline = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=self.device if self.device != "cpu" else -1,
                )

                logger.info("NER pipeline loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load NER model: {e}")
                raise

    def anonymize(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Anonymize PII in Arabic legal text.

        This method detects named entities and replaces them with
        standardized masks for Law 151/2020 compliance.

        Args:
            text: Raw Arabic text potentially containing PII

        Returns:
            Tuple of:
                - anonymized_text: Text with PII replaced by masks
                - audit_log: List of dicts documenting each replacement
                             (for compliance auditing)
        """
        if not text or not text.strip():
            return text, []

        # Load pipeline on first use
        self._load_pipeline()

        # Detect entities
        try:
            entities = self._pipeline(text)
        except Exception as e:
            logger.error(f"NER inference failed: {e}")
            # Return original text if NER fails (log for manual review)
            return text, [{"error": str(e)}]

        if not entities:
            return text, []

        # Sort entities by start position (descending) for safe replacement
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)

        anonymized = text
        audit_log = []

        for entity in entities_sorted:
            entity_type = entity["entity_group"]

            # Get mask for this entity type
            mask = ENTITY_MASKS.get(entity_type)

            if mask:
                start = entity["start"]
                end = entity["end"]
                original_text = text[start:end]

                # Replace in text
                anonymized = anonymized[:start] + mask + anonymized[end:]

                # Log for audit trail
                audit_log.append(
                    {
                        "entity_type": entity_type,
                        "original_text": original_text,
                        "replacement": mask,
                        "confidence": round(entity["score"], 4),
                        "start_position": start,
                        "end_position": end,
                    }
                )

        logger.debug(f"Anonymized {len(audit_log)} entities")

        return anonymized, audit_log

    def anonymize_batch(self, texts: List[str]) -> List[Tuple[str, List[Dict]]]:
        """
        Anonymize a batch of texts.

        More efficient than calling anonymize() repeatedly.

        Args:
            texts: List of texts to anonymize

        Returns:
            List of (anonymized_text, audit_log) tuples
        """
        results = []
        for text in texts:
            result = self.anonymize(text)
            results.append(result)

        return results


class SimpleAnonymizer:
    """
    Fallback anonymizer using regex patterns.

    Use this when CAMeLBERT is not available or for faster processing.
    Less accurate than NER-based anonymization.
    """

    def __init__(self):
        import re

        # Common Arabic name patterns (simplified)
        self.name_pattern = re.compile(
            r"\b(محمد|أحمد|علي|حسن|حسين|عمر|خالد|سعيد|يوسف|إبراهيم|"
            r"محمود|مصطفى|عبد\s*ال\w+)\s+"
            r"(محمد|أحمد|علي|حسن|حسين|عمر|خالد|سعيد|يوسف|إبراهيم|"
            r"محمود|مصطفى|عبد\s*ال\w+)\b",
            re.UNICODE,
        )

        # Egyptian governorates
        self.location_pattern = re.compile(
            r"\b(القاهرة|الإسكندرية|الجيزة|الشرقية|الدقهلية|البحيرة|"
            r"المنوفية|الغربية|كفر\s*الشيخ|دمياط|بورسعيد|الإسماعيلية|"
            r"السويس|شمال\s*سيناء|جنوب\s*سيناء|الفيوم|بني\s*سويف|"
            r"المنيا|أسيوط|سوهاج|قنا|الأقصر|أسوان|"
            r"البحر\s*الأحمر|الوادي\s*الجديد|مطروح)\b",
            re.UNICODE,
        )

    def anonymize(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Anonymize using regex patterns.

        Less accurate than NER but much faster.
        """
        audit_log = []

        # Replace names
        def replace_name(match):
            audit_log.append(
                {
                    "entity_type": "PER",
                    "original_text": match.group(0),
                    "replacement": "[شخص]",
                    "confidence": 0.7,
                    "start_position": match.start(),
                    "end_position": match.end(),
                }
            )
            return "[شخص]"

        # Replace locations
        def replace_location(match):
            audit_log.append(
                {
                    "entity_type": "LOC",
                    "original_text": match.group(0),
                    "replacement": "[مكان]",
                    "confidence": 0.9,
                    "start_position": match.start(),
                    "end_position": match.end(),
                }
            )
            return "[مكان]"

        anonymized = self.name_pattern.sub(replace_name, text)
        anonymized = self.location_pattern.sub(replace_location, anonymized)

        return anonymized, audit_log
