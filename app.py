from dotenv import load_dotenv
# Carica immediatamente le variabili d'ambiente prima di qualsiasi altra importazione
load_dotenv()

import streamlit as st
import json
import logging
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any

# Import della tua pipeline
from main import PipelineLezioniRAG

# Configurazione pagina
st.set_page_config(
    page_title="RAG Pipeline - Lezioni Universitarie",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inizializzazione session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'risultati' not in st.session_state:
    st.session_state.risultati = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def inizializza_pipeline(config: Dict) -> PipelineLezioniRAG:
    """Inizializza la pipeline con la configurazione fornita"""
    try:
        pipeline = PipelineLezioniRAG(
            llm_model=config['llm_model'],
            embedding_model=config['embedding_model'],
            embedding_api_provider=config.get('embedding_api_provider'),
            pg_connection=config.get('pg_connection'),
            collection_name=config['collection_name']
        )
        return pipeline
    except Exception as e:
        st.error(f"Errore nell'inizializzazione della pipeline: {e}")
        return None

def main():
    # Header
    st.markdown('<p class="main-header">🎓 RAG Pipeline - Lezioni Universitarie</p>', unsafe_allow_html=True)
    st.markdown("**Trasforma trascrizioni di lezioni in knowledge base ricercabile**")
    st.markdown("---")
    
    # Sidebar - Configurazione
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        # Configurazione LLM
        st.subheader("🤖 Modello LLM")
        llm_model = st.selectbox(
            "Seleziona modello",
            [
                "gemini/gemini-2.5-flash",
                "gemini/gemini-2.0-flash-exp",
                "deepseek/deepseek-chat",
                "anthropic/claude-sonnet-4-5",
                "openai/gpt-4o"
            ],
            help="Modello per analisi e segmentazione"
        )
        
        # Gestione API Key basata sul provider
        provider = llm_model.split("/")[0]
        api_key_labels = {
            "gemini": "🔑 Google AI Studio API Key",
            "deepseek": "🔑 DeepSeek API Key",
            "anthropic": "🔑 Anthropic API Key",
            "openai": "🔑 OpenAI API Key"
        }
        
        api_key_help = {
            "gemini": "Ottieni la tua key su: https://aistudio.google.com/apikey",
            "deepseek": "Ottieni la tua key su: https://platform.deepseek.com/api_keys",
            "anthropic": "Ottieni la tua key su: https://console.anthropic.com/",
            "openai": "Ottieni la tua key su: https://platform.openai.com/api-keys"
        }
        
        env_var_names = {
            "gemini": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY"
        }
        
        # Check se API key è già nelle variabili d'ambiente
        import os
        env_var = env_var_names.get(provider)
        existing_key = os.getenv(env_var)
        
        if existing_key:
            st.success(f"✅ API Key trovata in variabile d'ambiente ({env_var})")
            api_key = existing_key
        else:
            api_key = st.text_input(
                api_key_labels.get(provider, "🔑 API Key"),
                type="password",
                help=api_key_help.get(provider, "Inserisci la tua API key"),
                placeholder="sk-..." if provider == "openai" else "Inserisci API key..."
            )
            
            if api_key:
                # Salva temporaneamente nella variabile d'ambiente per la sessione
                os.environ[env_var] = api_key
                st.success("✅ API Key configurata per questa sessione")
            else:
                st.warning(f"⚠️ Inserisci la tua API Key o configurala come variabile d'ambiente `{env_var}`")
        
        # Configurazione Embedding
        st.subheader("🔢 Modello Embedding")
        
        # Info box per aiutare la scelta
        with st.expander("ℹ️ Quale tipo di embedding scegliere?"):
            st.markdown("""
            **🖥️ Locale (Raccomandato per iniziare)**
            - ✅ Completamente gratuito
            - ✅ Privacy totale (nessun dato inviato online)
            - ✅ Ottima qualità per italiano
            - ⚠️ Richiede 2-8 GB RAM
            - ⚠️ Download iniziale del modello (~1-2 GB)
            
            **☁️ API Provider**
            - ✅ Più veloce (no download modello)
            - ✅ Meno RAM necessaria
            - ❌ Costo per utilizzo (~$0.01-0.05/lezione)
            - ⚠️ Dati inviati al provider
            
            **💡 Consiglio**: Inizia con Locale BAAI/bge-m3 (gratuito)
            """)
        
        embedding_provider = st.radio(
            "Tipo di embedding",
            ["🖥️ Locale (Sentence Transformers)", "☁️ API Provider"],
            help="Locale: gratuito ma richiede GPU/CPU. API: più veloce ma a pagamento"
        )
        
        if embedding_provider == "🖥️ Locale (Sentence Transformers)":
            embedding_model = st.selectbox(
                "Seleziona modello locale",
                [
                    "BAAI/bge-m3",
                    "intfloat/multilingual-e5-large-instruct",
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "BAAI/bge-large-en-v1.5",
                    "sentence-transformers/all-MiniLM-L6-v2"
                ],
                help="Modelli eseguiti localmente (gratuito)"
            )
            embedding_api_provider = None
            embedding_api_key = None
            
        else:  # API Provider
            embedding_api_provider = st.selectbox(
                "Seleziona provider API",
                [
                    "openai",
                    "cohere",
                    "voyage",
                    "mistral",
                    "google"
                ],
                help="Provider per embeddings via API"
            )
            
            # Modelli disponibili per provider
            embedding_models_by_provider = {
                "openai": [
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                    "text-embedding-ada-002"
                ],
                "cohere": [
                    "embed-multilingual-v3.0",
                    "embed-english-v3.0",
                    "embed-multilingual-light-v3.0"
                ],
                "voyage": [
                    "voyage-large-2",
                    "voyage-2",
                    "voyage-lite-02-instruct"
                ],
                "mistral": [
                    "mistral-embed"
                ],
                "google": [
                    "models/embedding-001",
                    "models/text-embedding-004"
                ]
            }
            
            embedding_model = st.selectbox(
                "Seleziona modello",
                embedding_models_by_provider.get(embedding_api_provider, []),
                help=f"Modelli disponibili per {embedding_api_provider}"
            )
            
            # API Key per embeddings
            embedding_env_vars = {
                "openai": "OPENAI_API_KEY",
                "cohere": "COHERE_API_KEY",
                "voyage": "VOYAGE_API_KEY",
                "mistral": "MISTRAL_API_KEY",
                "google": "GEMINI_API_KEY"
            }
            
            embedding_api_help = {
                "openai": "Usa la stessa key di OpenAI LLM",
                "cohere": "Ottieni key su: https://dashboard.cohere.com/api-keys",
                "voyage": "Ottieni key su: https://dash.voyageai.com/",
                "mistral": "Ottieni key su: https://console.mistral.ai/",
                "google": "Usa la stessa key di Gemini"
            }
            
            embedding_env_var = embedding_env_vars.get(embedding_api_provider)
            existing_embedding_key = os.getenv(embedding_env_var)
            
            if existing_embedding_key:
                st.success(f"✅ API Key per embeddings trovata ({embedding_env_var})")
                embedding_api_key = existing_embedding_key
            else:
                embedding_api_key = st.text_input(
                    f"🔑 {embedding_api_provider.title()} API Key",
                    type="password",
                    help=embedding_api_help.get(embedding_api_provider, "Inserisci API key"),
                    key="embedding_api_key_input"
                )
                
                if embedding_api_key:
                    os.environ[embedding_env_var] = embedding_api_key
                    st.success("✅ API Key embeddings configurata")
                else:
                    st.warning(f"⚠️ Inserisci API Key o configura `{embedding_env_var}`")
        
        # Configurazione Database
        st.subheader("💾 Database PostgreSQL")
        usa_vector_store = st.checkbox("Salva in Vector Store", value=True)
        
        if usa_vector_store:
            pg_host = st.text_input("Host", value=os.getenv("PGHOST", "localhost"))
            pg_port = st.text_input("Port", value=os.getenv("PGPORT", "5432"))
            pg_user = st.text_input("Username", value=os.getenv("PGUSER", "postgres"))
            pg_password = st.text_input("Password", type="password", value=os.getenv("PGPASSWORD", ""))
            pg_database = st.text_input("Database", value=os.getenv("PGDATABASE", "pandino"))
            collection_name = st.text_input("Collection", value=os.getenv("COLLECTION_NAME", "Compass"))
            
            pg_connection = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        else:
            pg_connection = None
            collection_name = "Compass"
        
        # Parametri avanzati
        st.subheader("🔧 Parametri Avanzati")
        batch_size = st.slider("Batch Size", 8, 64, 32, 8, help="Dimensione batch per embeddings")
        
        st.markdown("---")
        
        # Test API Key (opzionale)
        with st.expander("🧪 Test Connessioni"):
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                st.write("**Test LLM**")
                if api_key and st.button("Testa LLM", key="test_llm"):
                    with st.spinner("Test in corso..."):
                        try:
                            import litellm
                            test_response = litellm.completion(
                                model=llm_model,
                                messages=[{"role": "user", "content": "Rispondi solo con: OK"}],
                                max_tokens=10
                            )
                            st.success("✅ LLM funzionante!")
                            st.code(test_response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"❌ Errore: {str(e)}")
            
            with col_test2:
                if embedding_provider != "🖥️ Locale (Sentence Transformers)":
                    st.write("**Test Embeddings**")
                    if embedding_api_key and st.button("Testa Embeddings", key="test_embed"):
                        with st.spinner("Test in corso..."):
                            try:
                                from main import APIEmbeddings
                                test_embedder = APIEmbeddings(embedding_api_provider, embedding_model)
                                test_emb = test_embedder.embed_query("test")
                                st.success(f"✅ Embeddings funzionanti!")
                                st.caption(f"Dimensione: {len(test_emb) if test_emb else 'N/A'} dim")
                            except Exception as e:
                                st.error(f"❌ Errore: {str(e)}")
                else:
                    st.write("**Embeddings Locali**")
                    st.info("ℹ️ Verranno caricati al primo utilizzo")
                
                # Test PostgreSQL if vector store is enabled
                if usa_vector_store:
                    st.write("**Test Database**")
                    if st.button("Testa PostgreSQL", key="test_db"):
                        with st.spinner("Connessione al database..."):
                            try:
                                # Create the connection string in the same format used by the pipeline
                                connection_string = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
                                # Basic validation - check if all required fields are present
                                if all([pg_user, pg_password, pg_host, pg_database]):
                                    st.success("✅ Stringa connessione valida!")
                                else:
                                    st.error("❌ Parametri di connessione incompleti")
                            except Exception as e:
                                st.error(f"❌ Errore connessione DB: {str(e)}")
        
        # Info
        st.info("💡 **Tip**: Per iniziare usa Gemini + Embeddings Locali (100% gratuito)")
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Risultati", "🔍 Ricerca"])
    
    # TAB 1: Upload e Processing
    with tab1:
        st.header("Carica e Processa Lezione")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Upload file
            uploaded_file = st.file_uploader(
                "Carica trascrizione (TXT, MD)",
                type=['txt', 'md'],
                help="Carica il file contenente la trascrizione della lezione"
            )
            
            source_name = st.text_input(
                "Nome Lezione",
                value="Lezione - " + datetime.now().strftime("%Y-%m-%d"),
                help="Nome identificativo per questa lezione"
            )
        
        with col2:
            st.metric("Modello LLM", llm_model.split("/")[-1])
            
            # Display embedding info
            if embedding_provider == "🖥️ Locale (Sentence Transformers)":
                st.metric("Embedding", "🖥️ Locale")
                st.caption(embedding_model.split("/")[-1])
            else:
                st.metric("Embedding", f"☁️ {embedding_api_provider.title()}")
                st.caption(embedding_model)
            
            st.metric("Vector Store", "✅ Attivo" if usa_vector_store else "❌ Disattivo")
            
            # Indicatore API Key
            if api_key:
                st.metric("LLM API Key", "✅ Configurata")
            else:
                st.metric("LLM API Key", "❌ Mancante")
            
            # Indicatore API Key Embedding (solo se API provider)
            if embedding_provider != "🖥️ Locale (Sentence Transformers)":
                if embedding_api_key:
                    st.metric("Embed API Key", "✅ Configurata")
                else:
                    st.metric("Embed API Key", "❌ Mancante")
        
        if uploaded_file is not None:
            # Mostra preview
            with st.expander("👀 Anteprima Trascrizione"):
                content = uploaded_file.read().decode('utf-8')
                st.text_area("Contenuto", content[:1000] + "...", height=200, disabled=True)
                st.info(f"📝 Lunghezza totale: {len(content)} caratteri")
                uploaded_file.seek(0)  # Reset file pointer
            
            # Pulsante di elaborazione
            st.markdown("---")
            
            # Validazione API Keys
            if not api_key:
                st.error("❌ Configura prima l'API Key LLM nella sidebar!")
                st.stop()
            
            if embedding_provider != "🖥️ Locale (Sentence Transformers)" and not embedding_api_key:
                st.error("❌ Configura l'API Key per embeddings nella sidebar!")
                st.stop()
            
            if st.button("🚀 Avvia Elaborazione", type="primary", use_container_width=True):
                # Salva file temporaneamente
                temp_path = Path("temp")
                temp_path.mkdir(exist_ok=True)
                temp_file = temp_path / uploaded_file.name
                
                with open(temp_file, 'wb') as f:
                    f.write(uploaded_file.read())
                
                # Configurazione pipeline
                config = {
                    'llm_model': llm_model,
                    'embedding_model': embedding_model,
                    'embedding_api_provider': embedding_api_provider if embedding_provider != "🖥️ Locale (Sentence Transformers)" else None,
                    'pg_connection': pg_connection,
                    'collection_name': collection_name
                }
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Inizializza pipeline
                    status_text.text("🔄 Inizializzazione pipeline...")
                    progress_bar.progress(10)
                    pipeline = inizializza_pipeline(config)
                    
                    if pipeline is None:
                        st.error("❌ Errore nell'inizializzazione della pipeline")
                        st.stop()
                    
                    st.session_state.pipeline = pipeline
                    
                    # Step 1: Caricamento
                    status_text.text("📖 Caricamento trascrizione...")
                    progress_bar.progress(20)
                    
                    # Step 2: Analisi strutturale
                    status_text.text("🔍 Analisi struttura lezione...")
                    progress_bar.progress(30)
                    
                    # Step 3-5: Processing completo
                    status_text.text("⚙️ Elaborazione in corso (può richiedere alcuni minuti)...")
                    progress_bar.progress(40)
                    
                    risultati = pipeline.processa_lezione(
                        file_trascrizione=str(temp_file),
                        source_name=source_name,
                        output_dir="output",
                        salva_vector_store=usa_vector_store,
                        batch_size=batch_size
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Elaborazione completata!")
                    
                    st.session_state.risultati = risultati
                    
                    # Success message
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("🎉 **Lezione processata con successo!**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Metriche risultato
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        num_unita = len(risultati['unita_didattiche'].get('unita_concettuali', []))
                        st.metric("📚 Unità Didattiche", num_unita)
                    with col2:
                        st.metric("🔢 Embeddings", risultati.get('num_embeddings', 0))
                    with col3:
                        concetti = len(risultati['analisi_struttura'].get('concetti_chiave', []))
                        st.metric("💡 Concetti Chiave", concetti)
                    
                    st.info("👉 Vai al tab **Risultati** per vedere i dettagli")
                    
                    # Cleanup
                    temp_file.unlink()
                    
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.text("")
                    st.error(f"❌ Errore durante l'elaborazione: {str(e)}")
                    logger.error(f"Errore: {e}", exc_info=True)
        else:
            st.info("👆 Carica una trascrizione per iniziare")
    
    # TAB 2: Risultati
    with tab2:
        st.header("Risultati Elaborazione")
        
        if st.session_state.risultati is None:
            st.warning("⚠️ Nessun risultato disponibile. Elabora prima una lezione nel tab 'Upload & Process'")
        else:
            risultati = st.session_state.risultati
            
            # Overview
            st.subheader("📋 Overview")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Lezione:** {risultati['source_name']}")
                st.write(f"**File:** {Path(risultati['file_trascrizione']).name}")
            with col2:
                if st.button("💾 Esporta JSON Completo"):
                    json_str = json.dumps(risultati, ensure_ascii=False, indent=2)
                    st.download_button(
                        "⬇️ Download JSON",
                        json_str,
                        file_name=f"risultati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            st.markdown("---")
            
            # Analisi Strutturale
            with st.expander("🔍 Analisi Strutturale", expanded=True):
                analisi = risultati['analisi_struttura']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Titolo:** {analisi.get('titolo_lezione', 'N/A')}")
                    st.write(f"**Argomento:** {analisi.get('argomento_generale', 'N/A')}")
                with col2:
                    st.write(f"**Durata stimata:** {analisi.get('durata_stimata', 'N/A')}")
                
                # Struttura macro
                if 'struttura_macro' in analisi:
                    st.subheader("📚 Struttura Macro")
                    for sez in analisi['struttura_macro']:
                        st.markdown(f"**{sez['sezione']}. {sez['titolo']}**")
                        st.write(f"- Argomento: {sez['argomento']}")
                        st.write(f"- Durata: {sez.get('durata_relativa', 'N/A')}")
                
                # Concetti chiave
                if 'concetti_chiave' in analisi:
                    st.subheader("💡 Concetti Chiave")
                    concetti_df = pd.DataFrame(analisi['concetti_chiave'])
                    if not concetti_df.empty:
                        st.dataframe(concetti_df, use_container_width=True)
            
            # Unità Didattiche
            with st.expander("📖 Unità Didattiche", expanded=True):
                unita_list = risultati['unita_didattiche'].get('unita_concettuali', [])
                
                for unita in unita_list:
                    with st.container():
                        st.markdown(f"### {unita['id']}: {unita['titolo_unita']}")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write("**Concetti principali:**", ", ".join(unita.get('concetti_principali', [])))
                            st.info(f"📌 {unita.get('sintesi_punto_chiave', 'N/A')}")
                        with col2:
                            num_domande = len(unita.get('domande_studente_tipiche', []))
                            st.metric("❓ Domande", num_domande)
                        
                        # Testo riformulato
                        with st.expander("📝 Testo Completo"):
                            st.write(unita.get('testo_riformulato', 'N/A'))
                        
                        # Domande
                        if unita.get('domande_studente_tipiche'):
                            with st.expander("❓ Domande Tipiche"):
                                for i, domanda in enumerate(unita['domande_studente_tipiche'], 1):
                                    st.write(f"{i}. {domanda}")
                        
                        st.markdown("---")
    
    # TAB 3: Ricerca
    with tab3:
        st.header("🔍 Ricerca nel Vector Store")
        
        if not usa_vector_store:
            st.warning("⚠️ Vector Store non attivo. Abilita nelle impostazioni e riprocessa la lezione.")
        elif st.session_state.pipeline is None:
            st.warning("⚠️ Pipeline non inizializzata. Elabora prima una lezione.")
        else:
            st.info("🎯 Cerca contenuti nelle lezioni processate")
            
            # Query input
            query = st.text_input(
                "Inserisci la tua domanda",
                placeholder="Es: Come si misurano gli indicatori qualitativi?"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                k_results = st.slider("Numero risultati", 1, 10, 3)
            
            if st.button("🔍 Cerca", type="primary") and query:
                try:
                    with st.spinner("Ricerca in corso..."):
                        pipeline = st.session_state.pipeline
                        # Use similarity_search_with_score to get both documents and similarity scores
                        results = pipeline.vector_store.similarity_search_with_score(query, k=k_results)
                    
                    st.success(f"✅ Trovati {len(results)} risultati")
                    
                    # Mostra risultati
                    for i, (doc, score) in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"### 📄 Risultato {i}")
                            
                            # Calculate similarity from distance (score is typically a distance)
                            # For cosine similarity, similarity = 1 - distance
                            similarity = 1 - score
                            similarity_percent = similarity * 100

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                            with col2:
                                tipo = doc.metadata.get('tipo', 'N/A')
                                tipo_emoji = {
                                    'testo_principale': '📝',
                                    'sintesi': '📌',
                                    'domanda': '❓'
                                }.get(tipo, '📄')
                                st.write(f"**Tipo:** {tipo_emoji} {tipo}")
                            with col3:
                                st.metric("**Similarità**", f"{similarity_percent:.1f}%")
                            
                            st.markdown("**Contenuto:**")
                            st.info(doc.page_content)
                            
                            with st.expander("🔗 Testo Originale Completo"):
                                st.write(doc.metadata.get('text', 'N/A'))
                            
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Errore durante la ricerca: {str(e)}")
            
            elif query and not st.button:
                st.info("👆 Clicca su 'Cerca' per avviare la ricerca")

if __name__ == "__main__":
    main()
