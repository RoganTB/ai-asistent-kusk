import streamlit as st
import json
import ollama

# 1. ZÁKLADNÍ NASTAVENÍ STRÁNKY
st.set_page_config(page_title="Interní AI Asistent", page_icon="🤖", layout="centered")

# Funkce pro načtení firemních dat ze souboru data.json
def load_data():
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Soubor data.json nebyl nalezen. Vytvořte ho prosím."}

data = load_data()

# 2. INICIALIZACE PAMĚTI (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# --- SIDEBAR (Boční panel) ---
with st.sidebar:
    st.title("⚙️ Nastavení systému")
    st.info("Běží na lokálním modelu Llama 3.")
    st.write("Tento asistent neposílá data do cloudu.")
    st.divider()
    if st.button("🗑️ Vymazat historii chatu"):
        st.session_state.messages = []
        st.session_state.last_answer = None
        st.rerun()

st.title("🤖 Firemní Smart Asistent")
st.caption("Ptejte se na dovolenou, kontakty nebo vnitřní pravidla úřadu.")

# 3. ZOBRAZENÍ HISTORIE CHATU
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. HLAVNÍ LOGIKA DOTAZOVÁNÍ
if prompt := st.chat_input("Jak vám mohu pomoci?"):
    # Uložení dotazu uživatele
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # SYSTEM PROMPT: Definice role a striktní češtiny
    system_context = f"""
    Jsi profesionální interní asistent na Krajském úřadě. 
    Mluv VŽDY A POUZE ČESKY. Piš gramaticky správně.
    
    Zde jsou tvoje znalosti z interní databáze: {json.dumps(data, ensure_ascii=False)}
    
    PRAVIDLA:
    1. Odpovídej věcně na základě poskytnutých dat.
    2. Pokud se uživatel ptá na "pravidla", cituj přesně text z dat.
    3. Pokud informaci v datech nemáš, slušně to přiznej česky a odkaž na HR.
    4. Nepoužívej angličtinu.
    """

    # Generování odpovědi
    with st.chat_message("assistant"):
        response_area = st.empty()
        full_response = ""
        
        try:
            # Volání modelu s nastavením nízké teploty pro vyšší přesnost
            stream = ollama.chat(
                model='llama3', 
                messages=[
                    {'role': 'system', 'content': system_context},
                    *st.session_state.messages
                ],
                stream=True,
                options={
                    'temperature': 0.1,  # Snižuje chyby a nesmysly v češtině
                    'top_p': 0.9
                }
            )

            for chunk in stream:
                content = chunk['message']['content']
                full_response += content
                response_area.markdown(full_response + "▌")
            
            response_area.markdown(full_response)
            
            # Uložení výsledku
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.last_answer = full_response
            
            # Refresh pro zobrazení tlačítka ke stažení
            st.rerun()

        except Exception as e:
            st.error(f"Chyba při komunikaci s AI: {e}")

# 5. TLAČÍTKO PRO EXPORT (Zobrazí se po vygenerování odpovědi)
if st.session_state.last_answer:
    st.divider()
    st.download_button(
        label="📥 Stáhnout poslední odpověď jako potvrzení",
        data=st.session_state.last_answer,
        file_name="potvrzeni_ai.txt",
        mime="text/plain"
    )