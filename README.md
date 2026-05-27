# Interní AI asistent KÚSK

Lokální Streamlit aplikace pro interní dotazy nad znalostní bází a pro práci s PDF dokumenty. Projekt spojuje původní RAG asistenta `ai-asistent-kusk` a dokumentového asistenta `ai-pdf-assistant` do jedné aplikace.

## Co aplikace umí

- odpovídat česky podle interní znalostní báze v `data.json`,
- nahrát PDF dokument a ptát se na jeho obsah,
- vyhledat relevantní části PDF pomocí lokálních embeddings,
- zobrazit použité úryvky dokumentu,
- stáhnout poslední odpověď jako textový soubor,
- běžet lokálně přes Ollama bez posílání dokumentů do cloudu.

## Technologie

- Streamlit pro webové rozhraní,
- Ollama pro lokální jazykové modely,
- `nomic-embed-text` pro embeddings,
- NumPy pro vyhledávání podobných částí dokumentu,
- pypdf pro čtení PDF.

## Instalace

1. Nainstalujte Ollama:

   <https://ollama.com>

2. Stáhněte potřebné modely:

   ```powershell
   ollama pull llama3
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. Připravte Python prostředí:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

4. Spusťte aplikaci:

   ```powershell
   streamlit run app.py
   ```

## Volitelné nastavení modelů

Modely je možné změnit přes proměnné prostředí:

```powershell
$env:CHAT_MODEL = "llama3"
$env:PDF_MODEL = "llama3.2"
$env:EMBEDDING_MODEL = "nomic-embed-text"
$env:OLLAMA_URL = "http://localhost:11434"
streamlit run app.py
```

## Struktura projektu

```text
.
├── app.py
├── data.json
├── requirements.txt
└── README.md
```

## Další kroky

- Doplnit reálnou interní znalostní bázi.
- Přidat trvalé ukládání indexů pro větší sadu dokumentů.
- Doplnit autentizaci a role pro produkční nasazení.
- Po ověření sloučené aplikace archivovat nebo odstranit původní repozitář `ai-pdf-assistant`.
