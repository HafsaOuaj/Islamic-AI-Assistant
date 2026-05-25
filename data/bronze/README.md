# Quran Tafsir (Ibn Kathir) JSONL Dataset

This dataset provides the Ibn Kathir (Abridged) tafsir for every verse of the Quran as newline-delimited JSON (`.jsonl`). Each surah (chapter) is stored in its own file within `data/quran/`.

## Contents

- `data/quran/surah_XXX.jsonl` – Tafsir entries for surah `XXX` (`001`–`114`). Every line is one JSON object describing a verse and its commentary.

Common keys inside each JSON object:

- `surah`: Surah number (1–114)
- `ayah`: Ayah number within the surah
- `verse_key`: Canonical identifier formatted as `<surah>:<ayah>`
- `resource_name`, `resource_id`, `language_id`, `slug`, `translated_name`: Source metadata
- `text_html`: Tafsir text in HTML format (UTF-8, may include inline markup classes)

## Usage Notes

- Parse with streaming JSON readers such as Python `json.loads` per line, `pandas.read_json(..., lines=True)`, or `jq`.
- `text_html` preserves the HTML provided by the source. Sanitize or render as needed.
- Files are UTF-8 encoded.

## License / Source

Replace this section with the license or attribution required by the original data source before publishing on Kaggle.

## Changelog

- Initial release: Direct export from the local scraper project without modification.
