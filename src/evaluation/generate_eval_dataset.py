import pandas as pd
import anthropic
import json
import time


SYSTEM_PROMPT = """You are an expert in Quranic exegesis (tafsir).
Given a Quranic ayah and its tafsir commentary, generate evaluation question-answer pairs.

Rules:
- Questions must be answerable ONLY from the provided tafsir text — no outside knowledge.
- Answers must be concise, factual, and directly grounded in the tafsir.
- Vary question types: factual, interpretive, referential (who said X), contextual.
- Return ONLY valid JSON — no preamble, no markdown fences.

Output format:
[
  {"question": "...", "answer": "..."},
  ...
]"""


def build_prompt(surah_n: int, ayah_n: int, ayah_en: str, tafsir: str, n_questions: int = 3) -> str:
    return f"""Surah {surah_n}, Ayah {ayah_n}

English translation:
{ayah_en}

Tafsir commentary:
{tafsir}

Generate {n_questions} diverse question-answer pairs based strictly on the tafsir above."""


def generate_qa_pairs(
    client: anthropic.Anthropic,
    surah_n: int,
    ayah_n: int,
    ayah_en: str,
    tafsir: str,
    n_questions: int = 3,
    retries: int = 2,
) -> list[dict]:
    prompt = build_prompt(surah_n, ayah_n, ayah_en, tafsir, n_questions)

    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            pairs = json.loads(raw)
            return pairs

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt < retries:
                time.sleep(1)

        except anthropic.APIError as e:
            print(f"  [ERROR] API error (attempt {attempt + 1}): {e}")
            if attempt < retries:
                time.sleep(2)

    return []


def run(
    input_csv: str,
    output_csv: str,
    n_questions_per_row: int = 3,
    tafsir_max_chars: int = 4000,
):
    df = pd.read_csv(input_csv)

    client = anthropic.Anthropic()

    records = []

    for _, row in df.iterrows():
        surah_n = int(row["surah_n"])
        ayah_n = int(row["ayah_n"])
        ayah_en = str(row["ayah_text"]).strip()
        tafsir = str(row["ayah_tafsir"]).strip()

        # Truncate very long tafsirs to stay within token budget
        tafsir_truncated = tafsir[:tafsir_max_chars]

        print(f"Processing Surah {surah_n}:{ayah_n} ...")

        pairs = generate_qa_pairs(
            client=client,
            surah_n=surah_n,
            ayah_n=ayah_n,
            ayah_en=ayah_en,
            tafsir=tafsir_truncated,
            n_questions=n_questions_per_row,
        )

        for pair in pairs:
            records.append({
                "surah_n": surah_n,
                "ayah_n": ayah_n,
                "question": pair.get("question", ""),
                "answer": pair.get("answer", ""),
                "source_text": tafsir_truncated,
            })

        print(f"  -> {len(pairs)} pairs generated")
        time.sleep(0.5)

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(out_df)} QA pairs to {output_csv}")


if __name__ == "__main__":
    run(
        input_csv="../../data/gold/test_dataset.csv",
        output_csv="../../data/gold/eval_dataset.csv",
        n_questions_per_row=3,
    )
