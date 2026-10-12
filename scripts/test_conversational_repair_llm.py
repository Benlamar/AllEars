"""Evaluates gemma4_4b on conversational repair of the Calligraphy failure sequence.

Compares:
1. Baseline: Current system prompt in live_session.py (naive text history).
2. Repair-Aware: System prompt equipped with phonetic error awareness and conversational correction rules.
"""

from audio_agent.providers.llm import OpenAICompatibleLLMProvider

ENDPOINT = "http://192.168.1.11:8001/v1"
MODEL = "gemma4_4b"

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful, conversational voice assistant. Speak naturally in concise sentences. "
    "Avoid markdown headers, asterisks, bullet points, and code blocks."
)

REPAIR_AWARE_SYSTEM_PROMPT = (
    "You are a helpful, conversational voice assistant speaking with a user over a microphone. "
    "Guidelines:\n"
    "1. Keep answers concise (1-3 sentences) and speak naturally without bullet points or markdown.\n"
    "2. Speech recognition may occasionally produce phonetic errors on uncommon words or in background noise.\n"
    "3. If the user repeats a single word or phrase with emphasis (e.g. 'Calligraphy!'), recognize that they are correcting "
    "a previous misunderstanding. Immediately re-evaluate the conversation under the corrected topic and address their intent directly.\n"
    "4. Do not dwell on the transcription error; seamlessly answer their intended question."
)

DIALOGUE_SEQUENCE = [
    # Turn 1
    {"role": "user", "content": "Things on Candy Brackett and Lee Cribby."},
    # Assistant asks naive question
    {"role": "assistant", "content": "Could you clarify what you mean by Candy Brackett and Lee Cribby?"},
    # Turn 2: clarification also misrecognized
    {"role": "user", "content": "Coffee and Cabbie Grumpy."},
    # Assistant continues confusion
    {"role": "assistant", "content": "I am not sure how coffee connects to Candy Brackett. What are you looking for?"},
    # Turn 3: User emphasizes the correction
    {"role": "user", "content": "Calligraphy!"},
]


def test_conversational_repair():
    llm = OpenAICompatibleLLMProvider(endpoint=ENDPOINT, model_name=MODEL, temperature=0.3)
    if not llm.is_available():
        print(f"[!] Endpoint {ENDPOINT} is unavailable.")
        return

    print("=" * 80)
    print("Testing Turn 3: User says 'Calligraphy!' after prior misunderstood turns")
    print("=" * 80)

    # 1. Baseline System Prompt
    print("\n--- 1. Baseline System Prompt ---")
    messages_baseline = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}] + DIALOGUE_SEQUENCE
    resp_baseline = llm.generate(messages_baseline)
    print(f"Agent Response:\n{resp_baseline}\n")

    # 2. Repair-Aware System Prompt
    print("\n--- 2. Repair-Aware System Prompt ---")
    messages_repair = [{"role": "system", "content": REPAIR_AWARE_SYSTEM_PROMPT}] + DIALOGUE_SEQUENCE
    resp_repair = llm.generate(messages_repair)
    print(f"Agent Response:\n{resp_repair}\n")

    # 3. With Confidence / Uncertainty Annotation
    print("\n--- 3. Repair-Aware Prompt + Low-Confidence Metadata ---")
    annotated_dialogue = [
        {"role": "user", "content": "[Uncertain ASR]: Things on Candy Brackett and Lee Cribby."},
        {"role": "assistant", "content": "Could you clarify what you mean?"},
        {"role": "user", "content": "[Uncertain ASR]: Coffee and Cabbie Grumpy."},
        {"role": "assistant", "content": "Could you clarify?"},
        {"role": "user", "content": "Calligraphy! How good should a person be at calligraphy?"},
    ]
    messages_annotated = [{"role": "system", "content": REPAIR_AWARE_SYSTEM_PROMPT}] + annotated_dialogue
    resp_annotated = llm.generate(messages_annotated)
    print(f"Agent Response:\n{resp_annotated}\n")


if __name__ == "__main__":
    test_conversational_repair()
