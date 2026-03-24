🏢 Interní AI Asistent (RAG Prototyp)

Prototyp interního AI asistenta postaveného na architektuře RAG (Retrieval-Augmented Generation), který umožňuje efektivní a bezpečné vyhledávání informací v rámci organizace.

Aplikace běží plně lokálně, bez nutnosti odesílání citlivých dat do externích služeb.

🎯 Hlavní přínosy
🔍 Rychlé vyhledávání informací
(např. dovolená, interní směrnice, IT podpora)
🔒 Ochrana dat
Veškeré zpracování probíhá lokálně (GDPR-friendly)
⚡ Snížení administrativní zátěže
Odlehčení HR a IT oddělení od rutinních dotazů
🧠 Jak systém funguje (RAG)

Aplikace využívá princip Retrieval-Augmented Generation:

Uživatel zadá dotaz
Systém vyhledá relevantní informace v lokální znalostní bázi
Vybraná data jsou předána jazykovému modelu jako kontext
Model vygeneruje odpověď na základě těchto informací

Díky tomu jsou odpovědi přesnější a méně náchylné k halucinacím.

🏗️ Architektura
LLM: Llama 3 (běží lokálně přes Ollama)
Frontend: Streamlit (jednoduché webové UI)
Znalostní báze: JSON (strukturovaná data)
Komunikace: lokální inference bez API
⚙️ Klíčové vlastnosti
💬 Chat rozhraní s historií (session_state)
🎯 Nízká temperature (0.1) pro konzistentní odpovědi
📄 Export odpovědí do .txt
🧩 Modulární struktura (připraveno na rozšíření)
📸 Ukázka

(doporučeno doplnit screenshot aplikace)

🚀 Instalace a spuštění
1. Instalace Ollama

https://ollama.com

2. Stažení modelu

ollama pull llama3

3. Klonování repozitáře

git clone https://github.com/RoganTB/ai-asistent-kusk
cd ai-asistent-kusk

4. Virtuální prostředí

python -m venv venv
venv\Scripts\activate # Windows

source venv/bin/activate # Linux / macOS
5. Instalace závislostí

pip install streamlit ollama

6. Spuštění aplikace

streamlit run app.py

📂 Struktura projektu (příklad)

.
├── app.py
├── data.json
├── requirements.txt
└── README.md

📈 Možnosti rozšíření
📄 Podpora PDF dokumentů
🧠 Vector databáze (FAISS, ChromaDB)
🔎 Embeddings-based vyhledávání
🔐 Role-based access control
🎨 UI branding
⚠️ Omezení
Jednoduchá znalostní báze (JSON)
Bez embeddings (zatím)
Proof of concept (ne produkční řešení)
📌 Status projektu

🧪 Prototyp / experimentální řešení

🤝 Možné využití
státní správa
interní knowledge base
HR / IT asistenti
malé organizace
