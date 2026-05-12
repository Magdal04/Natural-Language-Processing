# Natural-Language-Processing
Uni work

## QA notebook

Notebookul `LAB02_Sistem_QA.ipynb` implementează un sistem QA extractiv (RO) peste un snapshot Wikipedia.

### Setup (local / Colab)

- Instalează dependențele: `pip install -r requirements.txt`
- (Dacă e nevoie) instalează modelul spaCy: `python -m spacy download ro_core_news_sm`

### Notă despre reproducibilitate / offline

- Notebookul folosește `data/taylor_swift_wikipedia_ro.txt` ca snapshot local.
- Dacă snapshot-ul există, nu mai face request către Wikipedia.
