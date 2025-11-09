# 🎓 RAG Pipeline - Lezioni Universitarie

Trasforma trascrizioni di lezioni universitarie in una knowledge base ricercabile usando RAG (Retrieval-Augmented Generation).

## ✨ Funzionalità

- 📤 **Upload trascrizioni** (TXT, MD)
- 🤖 **Analisi automatica** con LLM (Gemini, Claude, GPT-4, DeepSeek)
- 📚 **Segmentazione** in unità didattiche discrete
- 🔢 **Generazione embeddings** multilingue
- 💾 **Vector Store** PostgreSQL/pgvector (opzionale)
- 🔍 **Ricerca semantica** nei contenuti
- 📊 **Visualizzazione risultati** interattiva
- 💾 **Export JSON** completo

## 🚀 Quick Start

### 1. Installazione

```bash
# Clone repository (o crea nuova directory)
mkdir rag-pipeline && cd rag-pipeline

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Configurazione API Keys

**Opzione A: File .env (Raccomandato)**
```bash
# Copia il file di esempio
cp .env.example .env

# Modifica .env con le tue chiavi
nano .env  # o usa il tuo editor preferito
```

**Opzione B: Interfaccia Streamlit**
- Le API keys possono essere inserite direttamente nell'interfaccia
- Valide solo per la sessione corrente

### 3. Avvio

```bash
streamlit run app.py
```

L'app si aprirà automaticamente su `http://localhost:8501`

## 🔑 Ottenere le API Keys

### LLM Models

#### Google Gemini (Gratuito)
1. Vai su https://aistudio.google.com/apikey
2. Clicca "Create API Key"
3. Copia la chiave in `.env` come `GEMINI_API_KEY`

#### DeepSeek
1. Registrati su https://platform.deepseek.com/
2. Vai su API Keys
3. Copia la chiave in `.env` come `DEEPSEEK_API_KEY`

#### Anthropic Claude
1. Registrati su https://console.anthropic.com/
2. Vai su API Keys
3. Copia la chiave in `.env` come `ANTHROPIC_API_KEY`

#### OpenAI
1. Vai su https://platform.openai.com/api-keys
2. Crea una nuova API key
3. Copia la chiave in `.env` come `OPENAI_API_KEY`

### Embedding Models (opzionale)

#### 🖥️ Locale (Sentence Transformers) - RACCOMANDATO
- **Gratuito** e **Privacy-first**
- Nessuna API key necessaria
- Eseguito sulla tua macchina
- Buona qualità per italiano
- **Modelli consigliati**: BAAI/bge-m3, multilingual-e5-large

#### ☁️ API Providers

**OpenAI Embeddings**
- Stessa API key di OpenAI LLM
- Modelli: text-embedding-3-small, text-embedding-3-large
- Costo: ~$0.02 per 1M tokens

**Cohere Embeddings**
1. Registrati su https://dashboard.cohere.com/
2. Ottieni API key
3. Configura come `COHERE_API_KEY`
- Modelli: embed-multilingual-v3.0 (ottimo per italiano)

**Voyage AI**
1. Registrati su https://dash.voyageai.com/
2. Ottieni API key
3. Configura come `VOYAGE_API_KEY`
- Modelli: voyage-large-2, voyage-2

**Mistral**
- Usa la chiave Mistral
- Modello: mistral-embed

## 📦 Struttura Files

```
rag-pipeline/
├── app.py                          # Streamlit app
├── pipeline_completa_lezioni.py    # Pipeline RAG core
├── requirements.txt                # Dipendenze Python
├── .env                           # Configurazione (non committare!)
├── .env.example                   # Template configurazione
├── output/                        # Output JSON generati
│   ├── *_analisi.json
│   └── *_unita.json
└── temp/                          # File temporanei upload
```

## 🎯 Come Usare

### 🆚 Embedding: Locale vs API - Quale scegliere?

| Criterio | 🖥️ Locale (Sentence Transformers) | ☁️ API (OpenAI/Cohere/Voyage) |
|----------|-----------------------------------|-------------------------------|
| **Costo** | ✅ Gratuito | ❌ A pagamento (~$0.01-0.05/lezione) |
| **Privacy** | ✅ Tutto locale, nessun dato inviato | ⚠️ Testi inviati al provider |
| **Velocità** | ⚠️ Dipende da CPU/GPU | ✅ Molto veloce |
| **Qualità** | ✅ Ottima per italiano (bge-m3) | ✅ Eccellente |
| **Setup** | ⚠️ Richiede download modello (1-2 GB) | ✅ Immediato con API key |
| **RAM** | ⚠️ 2-8 GB | ✅ Minima |
| **Internet** | ✅ Solo per download iniziale | ❌ Necessaria sempre |

**Raccomandazioni:**
- 🎓 **Studenti/Ricercatori**: Locale (gratuito, privacy)
- 🏢 **Aziende con budget**: API (veloce, scalabile)
- 🔒 **Dati sensibili**: SEMPRE locale
- ⚡ **Produzione/grandi volumi**: API (più veloce)

### Step 1: Configurazione Sidebar
1. **Seleziona modello LLM** (es: Gemini 2.5 Flash)
2. **Inserisci API Key LLM** (o carica da .env)
3. **Test connessione LLM** (opzionale ma raccomandato)
4. **Scegli tipo embedding**:
   - 🖥️ **Locale**: Gratuito, privacy-first, eseguito sulla tua macchina
   - ☁️ **API**: Più veloce, richiede API key e crediti
5. **Configura embedding model**
6. **Opzionale:** Abilita Vector Store PostgreSQL

### Step 2: Upload & Process
1. Carica file trascrizione (.txt o .md)
2. Dai un nome alla lezione
3. Clicca "🚀 Avvia Elaborazione"
4. Attendi completamento (può richiedere alcuni minuti)

### Step 3: Visualizza Risultati
- **Analisi strutturale**: Overview, concetti, struttura
- **Unità didattiche**: Testo, concetti, domande
- **Export JSON**: Salva risultati completi

### Step 4: Ricerca (se Vector Store attivo)
1. Vai al tab "🔍 Ricerca"
2. Inserisci domanda in linguaggio naturale
3. Ottieni risultati semanticamente rilevanti

## ⚙️ Configurazione PostgreSQL (Opzionale)

Se vuoi usare il Vector Store per ricerca semantica:

```bash
# Installa PostgreSQL con pgvector
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib

# macOS:
brew install postgresql

# Crea database
createdb pandino

# Abilita estensione pgvector
psql pandino -c "CREATE EXTENSION vector;"
```

Configura in `.env`:
```bash
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=your_password
PGDATABASE=pandino
```

## 🧪 Test Rapido

```bash
# Test senza Vector Store (più veloce)
# 1. Inserisci API key Gemini nella sidebar
# 2. Carica una trascrizione breve
# 3. NON abilitare Vector Store
# 4. Avvia elaborazione

# Tempo stimato: 2-5 minuti per lezione di 5000 parole
```

## 📊 Output Generati

### 1. Analisi Strutturale
```json
{
  "titolo_lezione": "...",
  "argomento_generale": "...",
  "struttura_macro": [...],
  "concetti_chiave": [...],
  "terminologia_tecnica": [...]
}
```

### 2. Unità Didattiche
```json
{
  "unita_concettuali": [
    {
      "id": "U001",
      "titolo_unita": "...",
      "testo_riformulato": "...",
      "concetti_principali": [...],
      "domande_studente_tipiche": [...]
    }
  ]
}
```

## 🔧 Parametri Avanzati

### Batch Size
- **8-16**: Più lento, meno memoria
- **32** (default): Bilanciato
- **64**: Più veloce, più memoria

### Modelli Embedding
- **BAAI/bge-m3** (Locale): Ottimo per italiano, veloce, GRATUITO
- **multilingual-e5-large** (Locale): Più accurato, più lento, GRATUITO
- **text-embedding-3-small** (OpenAI): Veloce, economico (~$0.02/1M tokens)
- **embed-multilingual-v3.0** (Cohere): Eccellente per italiano
- **voyage-large-2** (Voyage): Alta qualità, costoso

### Modelli LLM
- **Gemini 2.5 Flash**: Veloce, economico, buona qualità
- **Claude Sonnet**: Eccellente qualità, più costoso
- **GPT-4o**: Ottimo bilanciamento
- **DeepSeek**: Economico, buono per task semplici

## ❗ Troubleshooting

### "API Key non valida"
- Verifica che la chiave sia copiata correttamente (no spazi)
- Controlla di avere crediti disponibili
- Usa il test connessione nella sidebar

### "Out of memory"
- Riduci batch_size (es: 16 o 8)
- Usa modello embedding più leggero
- Processa trascrizioni più brevi

### "Elaborazione troppo lenta"
- Usa Gemini Flash invece di Claude/GPT-4
- Aumenta batch_size se hai RAM disponibile
- Verifica connessione internet

### "Vector Store non funziona"
- Verifica che PostgreSQL sia in esecuzione
- Controlla che l'estensione pgvector sia installata
- Verifica credenziali database

## 💡 Best Practices

1. **Primo test**: Usa Gemini Flash + embeddings locali (GRATUITO)
2. **Trascrizioni pulite**: Rimuovi header/footer non necessari
3. **Nomi descrittivi**: Usa nomi lezione chiari (es: "Statistica - Lezione 3")
4. **Backup JSON**: Salva sempre i risultati prima di chiudere
5. **API Keys**: Non condividere mai le chiavi, usa .env
6. **Embeddings**: Inizia con locali (gratis), passa ad API solo se serve velocità
7. **Costi**: Embedding locali = $0, OpenAI ~$0.02/lezione, Cohere ~$0.01/lezione

## 📝 TODO / Roadmap

- [ ] Supporto batch processing multiple lezioni
- [ ] Export in Markdown/PDF
- [ ] Grafici analisi temporale lezioni
- [ ] Integrazione con LMS (Moodle, Canvas)
- [ ] Generazione automatica quiz
- [ ] Confronto tra lezioni

## 🤝 Contributi

PRs benvenute! Per modifiche importanti, apri prima un issue.

## 📄 Licenza

MIT License - vedi LICENSE file

## 🆘 Supporto

- 📧 Email: [tua-email@example.com]
- 🐛 Issues: GitHub Issues
- 💬 Discussioni: GitHub Discussions

---

**Fatto con ❤️ per l'educazione universitaria**
