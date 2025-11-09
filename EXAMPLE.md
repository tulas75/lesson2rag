# 🎯 Configurazioni Esempio

Ecco alcune configurazioni ottimali per diversi casi d'uso.

---

## 💰 Configurazione "Zero Costi"

**Perfetta per: studenti, test, uso personale**

```yaml
LLM:
  Provider: Gemini
  Modello: gemini-2.5-flash
  API Key: Gratuita da https://aistudio.google.com/apikey

Embeddings:
  Tipo: 🖥️ Locale
  Modello: BAAI/bge-m3
  API Key: Non necessaria

Vector Store: Disabilitato (per test rapidi)
Batch Size: 32
```

**Costo totale: $0.00**
**Tempo elaborazione**: ~3-5 minuti per lezione (5000 parole)
**RAM necessaria**: ~4 GB

---

## ⚡ Configurazione "Velocità Massima"

**Perfetta per: produzione, grandi volumi, demo live**

```yaml
LLM:
  Provider: Gemini o OpenAI
  Modello: gemini-2.5-flash o gpt-4o-mini
  API Key: Necessaria

Embeddings:
  Tipo: ☁️ API Provider
  Provider: OpenAI
  Modello: text-embedding-3-small
  API Key: Stessa di OpenAI LLM

Vector Store: Abilitato (PostgreSQL + pgvector)
Batch Size: 64
```

**Costo stimato**: ~$0.05-0.10 per lezione
**Tempo elaborazione**: ~1-2 minuti per lezione (5000 parole)
**RAM necessaria**: ~1 GB

---

## 🏢 Configurazione "Aziendale Pro"

**Perfetta per: aziende, università, dati sensibili con budget**

```yaml
LLM:
  Provider: Anthropic
  Modello: claude-sonnet-4-5
  API Key: Necessaria

Embeddings:
  Tipo: ☁️ API Provider
  Provider: Cohere
  Modello: embed-multilingual-v3.0
  API Key: Necessaria

Vector Store: Abilitato (PostgreSQL Cloud)
Batch Size: 32
```

**Costo stimato**: ~$0.15-0.30 per lezione
**Tempo elaborazione**: ~2-3 minuti per lezione (5000 parole)
**Qualità**: Massima per analisi complesse

---

## 🔒 Configurazione "Privacy First"

**Perfetta per: dati sensibili, compliance GDPR, ricerca confidenziale**

```yaml
LLM:
  Provider: DeepSeek (meno privacy) o Locale (Ollama)
  Modello: deepseek-chat o llama3.1:70b (Ollama)
  API Key: Necessaria per DeepSeek

Embeddings:
  Tipo: 🖥️ Locale
  Modello: BAAI/bge-m3 o multilingual-e5-large
  API Key: Non necessaria

Vector Store: Locale (PostgreSQL self-hosted)
Batch Size: 16
```

**Costo**: Variabile (DeepSeek economico)
**Privacy**: Massima (embeddings 100% locali)
**Tempo elaborazione**: ~4-6 minuti per lezione

---

## 🎓 Configurazione "Ricerca Accademica"

**Perfetta per: tesi, pubblicazioni, analisi approfondite**

```yaml
LLM:
  Provider: Anthropic o OpenAI
  Modello: claude-sonnet-4-5 o gpt-4o
  API Key: Necessaria

Embeddings:
  Tipo: 🖥️ Locale
  Modello: intfloat/multilingual-e5-large-instruct
  API Key: Non necessaria

Vector Store: Abilitato
Batch Size: 16 (più accurato)
```

**Costo stimato**: ~$0.10-0.20 per lezione
**Qualità**: Massima per entrambi LLM e embeddings
**Tempo elaborazione**: ~5-7 minuti per lezione

---

## 💻 Configurazione "Laptop con GPU"

**Perfetta per: chi ha GPU NVIDIA con 8+ GB VRAM**

```yaml
LLM:
  Provider: Gemini
  Modello: gemini-2.5-flash
  API Key: Gratuita

Embeddings:
  Tipo: 🖥️ Locale
  Modello: BAAI/bge-m3
  API Key: Non necessaria
  Note: Usa automaticamente GPU se disponibile

Vector Store: Abilitato
Batch Size: 64 (sfrutta GPU)
```

**Costo**: $0.00
**Tempo elaborazione**: ~1-2 minuti per lezione
**Vantaggio**: Velocità API con costi zero

---

## 🌍 Configurazione "Multi-lingua"

**Perfetta per: lezioni in più lingue**

```yaml
LLM:
  Provider: Anthropic o Gemini
  Modello: claude-sonnet-4-5 o gemini-2.5-flash
  API Key: Necessaria

Embeddings:
  Tipo: 🖥️ Locale
  Modello: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  API Key: Non necessaria

Vector Store: Abilitato
Batch Size: 32
```

**Lingue supportate**: 50+ lingue
**Costo**: $0.00-0.10 per lezione
**Qualità**: Buona per tutte le lingue europee

---

## 📊 Tabella Comparativa

| Configurazione | Costo/Lezione | Velocità | Qualità | Privacy | Uso RAM |
|----------------|---------------|----------|---------|---------|---------|
| Zero Costi | $0.00 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~4 GB |
| Velocità Max | $0.05-0.10 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ~1 GB |
| Aziendale Pro | $0.15-0.30 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~2 GB |
| Privacy First | Variabile | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~4 GB |
| Ricerca | $0.10-0.20 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~6 GB |
| GPU | $0.00 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~2 GB |
| Multi-lingua | $0.00-0.10 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~3 GB |

---

## 🎯 Come Scegliere

**Domande da farsi:**

1. **Budget disponibile?**
   - Zero → "Zero Costi" o "GPU"
   - Limitato → "Velocità Massima"
   - Illimitato → "Aziendale Pro"

2. **Priorità principale?**
   - Velocità → "Velocità Massima"
   - Qualità → "Ricerca Accademica"
   - Privacy → "Privacy First"
   - Costi → "Zero Costi"

3. **Hardware disponibile?**
   - Solo CPU → "Zero Costi" o API
   - GPU NVIDIA → "GPU"
   - Server Cloud → "Velocità Massima"

4. **Tipo di dati?**
   - Pubblici → Qualsiasi
   - Sensibili → "Privacy First"
   - Confidenziali → "Privacy First" + Locale

5. **Volumi?**
   - Poche lezioni → "Zero Costi"
   - Molte lezioni → "Velocità Massima"
   - Produzione → "Aziendale Pro"

---

## 💡 Tips Finali

- **Inizia sempre con "Zero Costi"** per testare
- **Passa ad API solo se** la velocità è critica
- **Per tesi/ricerca**: usa Claude o GPT-4 (qualità massima)
- **Per didattica standard**: Gemini + locale è perfetto
- **Embeddings locali** sono quasi sempre sufficienti
- **Vector Store** è utile solo se fai ricerche frequenti
