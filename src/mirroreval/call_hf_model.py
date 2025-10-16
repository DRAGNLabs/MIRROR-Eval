from transformers import pipeline


def call_hf_model(model_name, input_text):
    pipe = pipeline("text-generation", model=model_name)
    return pipe(input_text, max_length=50, num_return_sequences=1)
