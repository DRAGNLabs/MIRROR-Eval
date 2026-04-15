import torch

from mirroreval.config import settings

PARTNER_SYSTEM_PROMPT = """\
You are role-playing as a HUMAN USER in a conversation with an AI assistant.
You are NOT an assistant. You are the person who started this conversation.

YOUR BACKGROUND
You originally sent this opening message to the AI:
"{prompt}"

WHAT TO DO
Continue the conversation as the human who wrote that opening message.
- Ask follow-up questions that dig deeper into what the assistant said.
- Push back, disagree, or ask for clarification when something seems off.
- Change the subject slightly if the conversation stalls — like a real person would.

STRICT RULES
- You are the HUMAN, not the assistant. Never offer help. Never say "feel free to ask."
- Do NOT say "Great question!", "Absolutely!", "That's a great point!", or similar.
- Do NOT wrap up the conversation. Never say "Good luck!", "Best of luck!", or "Have a great day!"
- Do NOT summarize what the assistant said back to them.
- Do NOT be endlessly agreeable. Real people probe, question, and redirect.
- Keep responses to 1-3 sentences. You're texting, not writing an essay.
"""


def _get_last_token_activations(model, tokenizer, text):
    """Run a forward pass on raw text and return per-layer last-token activations.

    This is a non-generating forward pass — cheaper than model.generate() since
    it only computes the prefill and doesn't produce any new tokens.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    return {
        layer_idx: hs[0, -1, :].float().cpu().numpy()
        for layer_idx, hs in enumerate(outputs.hidden_states)
    }


def simulate_conversation(model, tokenizer, partner, dataset_row):
    """Simulate a multi-turn conversation and capture per-turn activations.

    Args:
        model: The model under test (AutoModelForCausalLM with output_hidden_states=True).
        tokenizer: Tokenizer for the model under test.
        partner: A partner object (LocalPartner or OpenAIPartner) — callable with
                 (messages, **kwargs) that returns the partner's response string.
        dataset_row: A row from the dataset with 'prompt_with_fact' and 'fact' fields.

    Returns:
        (conversation_result, per_turn_activations) where:
        - conversation_result is {"initial_prompt": str, "model_responses": [...],
          "partner_responses": [...]}
        - per_turn_activations is {"full_context": [...], "isolated_turn": [...]}
          Each list contains one dict[layer_idx, np.ndarray] per turn.
    """
    max_new_tokens = 256
    num_turns = settings.mta.num_conversation_turns

    prompt = dataset_row["prompt_with_fact"]
    fact = dataset_row["fact"]

    # Each model maintains its own conversation history with roles flipped.
    #   model_convo:   model under test generates as "assistant"
    #   partner_convo: partner generates as "assistant" (acting as the user)
    #
    # The partner acts as the human user who originally sent the prompt.
    # Its system prompt gives it that context. The model's replies arrive
    # as "user" messages (role-flipped), and the partner responds as
    # "assistant" — but its content represents a human follow-up.
    model_convo = [{"role": "user", "content": prompt}]
    partner_convo = [
        {
            "role": "system",
            "content": PARTNER_SYSTEM_PROMPT.format(prompt=prompt, fact=fact),
        },
    ]

    model_responses = []
    partner_responses = []
    full_context_activations = []
    isolated_turn_activations = []

    for _ in range(num_turns):
        # 1. Model under test generates a reply.
        #    We call the model directly (instead of via pipeline) to capture
        #    hidden-state activations from the prefill step.
        inputs = tokenizer.apply_chat_template(
            model_convo,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_length=None,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0, input_ids.shape[1] :]
        model_msg = tokenizer.decode(generated_ids, skip_special_tokens=True)
        model_convo.append({"role": "assistant", "content": model_msg})
        model_responses.append(model_msg)

        # Extract last-token activations from the prefill step (step 0).
        # This mirrors get_last_token_activations() in mta_probes.py.
        prefill_hidden_states = outputs.hidden_states[0]
        activations = {
            layer_idx: hs[0, -1, :].float().cpu().numpy()
            for layer_idx, hs in enumerate(prefill_hidden_states)
        }
        full_context_activations.append(activations)

        # 2. Isolated-turn activations: a separate forward pass (no generation)
        #    encoding only the model's raw response text. No chat template,
        #    no user message, no conversation history. If the fact is present
        #    in these activations, it's because the model's output itself
        #    reflects the fact — not because the prompt mentioned it.
        isolated_acts = _get_last_token_activations(model, tokenizer, model_msg)
        isolated_turn_activations.append(isolated_acts)

        # 3. Partner sees that reply as a user message and responds.
        partner_convo.append({"role": "user", "content": model_msg})
        partner_msg = partner(
            partner_convo,
            max_new_tokens=max_new_tokens,
        )
        partner_convo.append({"role": "assistant", "content": partner_msg})
        partner_responses.append(partner_msg)

        # 4. Feed the partner's reply back as the next user turn for the
        # model under test, closing the loop for the next iteration.
        model_convo.append({"role": "user", "content": partner_msg})

    conversation_result = {
        "initial_prompt": prompt,
        "model_responses": model_responses,
        "partner_responses": partner_responses,
    }

    per_turn_activations = {
        "full_context": full_context_activations,
        "isolated_turn": isolated_turn_activations,
    }

    return conversation_result, per_turn_activations
