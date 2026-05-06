# Islamic-AI-Assistant


This is a multi-agents AI assistant aiming at helping Muslims to organize their life. The assistance consists of:

* **Mufasir agent**: help understanding the Quran
* **Fatwa agent**: help understanding religion by answer questions from Quran, Sunah, ... .
* **Planner agent**: help creating weekly plans based on the to-do lists and the prayer time



# Development
 

## Setup of the project
For the development part we will use `[Poetry](https://python-poetry.org/docs/basic-usage/)` to organize the labraries and modules of the application.


### Data set 
We have used the [gurgutan/quran-tafseer-qurancom](https://huggingface.co/datasets/gurgutan/quran-tafseer-qurancom?utm_source=chatgpt.com) dataset:
@dataset{quran_tafsir, title = {QuranDataset: A Dataset of Quran Verses and Tafseer}, url = {https://escape-team.tech/}, author = {Slepovichev Ivan}, email = {gurgutan@yandex.ru}, month = {March}, year = {2025} }



since there are missing tafassir for `4341` ayah. we have to use another data from the same book tafsir:[Kaggle dataset](https://www.kaggle.com/datasets/oyilmaztekin/quran-tafsir-ibn-kathir-jsonl?utm_source=chatgpt.com)

