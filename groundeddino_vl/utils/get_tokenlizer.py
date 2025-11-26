from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))
    print("Use pre-downloaded bert-base-uncased weights...")

    tokenizer_path = "/opt/GroundedDINO-VL/weights/AI-ModelScope/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    # tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        model_path = "/opt/GroundedDINO-VL/weights/AI-ModelScope/bert-base-uncased"
        return BertModel.from_pretrained(model_path)

        # return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
