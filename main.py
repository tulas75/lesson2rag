import litellm
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_postgres import PGVector
from langchain_core.embeddings import Embeddings

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper per SentenceTransformer compatibile con LangChain"""
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


class APIEmbeddings(Embeddings):
    """Wrapper per embeddings API tramite litellm compatibile con LangChain"""
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import litellm
        embeddings = []
        for text in texts:
            try:
                # Use litellm to get embeddings
                response = litellm.embedding(
                    model=self.model_name,
                    input=[text],
                    custom_llm_provider=self.provider
                )
                embeddings.append(response.data[0]['embedding'])
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")
                # Fallback to a zero vector with typical dimension
                embeddings.append([0.0] * 384)  # Common embedding dimension
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        import litellm
        try:
            response = litellm.embedding(
                model=self.model_name,
                input=[text],
                custom_llm_provider=self.provider
            )
            return response.data[0]['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * 384  # Common embedding dimension


class PipelineLezioniRAG:
    """Pipeline completa per processare lezioni e creare vector store"""
    
    def __init__(
        self,
        llm_model: str = "gemini/gemini-2.5-flash",
        embedding_model: str = "BAAI/bge-m3",
        embedding_api_provider: str = None,
        pg_connection: str = None,
        collection_name: str = "Compass"
    ):
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model
        self.embedding_api_provider = embedding_api_provider
        self.pg_connection = pg_connection
        self.collection_name = collection_name
        
        # Inizializza componenti
        self.embedding_model = None
        self.vector_store = None
        
    def _carica_file(self, filepath: str) -> str:
        """Carica contenuto da file"""
        return Path(filepath).read_text(encoding='utf-8')
    
    def _salva_json(self, data: Dict, filepath: str):
        """Salva dati in formato JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Salvato: {filepath}")
    
    # ========== STEP 1: ANALISI STRUTTURALE ==========
    
    def analizza_struttura(self, trascrizione: str) -> Dict:
        """Analizza la struttura della lezione"""
        prompt = f"""Sei un esperto di analisi didattica. Analizza questa trascrizione di una lezione universitaria in italiano e fornisci una struttura dettagliata del contenuto.

TRASCRIZIONE:
"{trascrizione}"

Fornisci l'output in formato JSON strutturale come segue:

{{
  "titolo_lezione": "...",
  "argomento_generale": "...",
  "durata_stimata": "...",
  
  "struttura_macro": [
    {{
      "sezione": 1,
      "titolo": "...",
      "argomento": "...",
      "posizione_approssimativa": "inizio/metà/fine o parole di riferimento",
      "durata_relativa": "breve/media/lunga"
    }}
  ],
  
  "concetti_chiave": [
    {{
      "concetto": "nome del concetto",
      "definizione_breve": "...",
      "importanza": "fondamentale/importante/secondario",
      "prima_menzione": "contesto o parole chiave dove appare",
      "prerequisiti": ["concetto X", "concetto Y"]
    }}
  ],
  
  "esempi_e_casi_studio": [
    {{
      "esempio": "descrizione breve",
      "concetto_illustrato": "...",
      "posizione": "indicazioni per trovarlo nella trascrizione"
    }}
  ],
  
  "transizioni_principali": [
    {{
      "da": "argomento X",
      "a": "argomento Y",
      "marker_testuale": "frase o parole che segnalano il cambio",
      "tipo_transizione": "logico/temporale/causale/contrasto"
    }}
  ],
  
  "collegamenti_concettuali": [
    {{
      "concetto_A": "...",
      "relazione": "causa/prerequisito/esempio/contrasto/estensione",
      "concetto_B": "..."
    }}
  ],
  
  "domande_retoriche_o_stimoli": [
    "domanda 1 posta dal docente",
    "domanda 2..."
  ],
  
  "terminologia_tecnica": [
    {{
      "termine": "...",
      "contesto_primo_uso": "...",
      "frequenza": "alta/media/bassa"
    }}
  ],
  
  "riassunto_progressione_didattica": "Descrizione narrativa di come il docente sviluppa l'argomento dall'inizio alla fine"
}}

IMPORTANTE:
- Sii preciso nell'identificare le transizioni
- Marca chiaramente i prerequisiti concettuali
- Identifica pattern ricorrenti (es: "il docente tende a dare prima la teoria poi l'esempio" o "usa spesso analogie")
"""
        
        logger.info("→ Analisi strutturale in corso...")
        response = litellm.completion(
            model=self.llm_model,
            messages=[{"content": prompt, "role": "user"}],
        )
        
        content = response.choices[0].message.content
        # Pulisci markdown se presente
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    
    # ========== STEP 2: SEGMENTAZIONE IN UNITÀ ==========
    
    def crea_unita_didattiche(self, trascrizione: str, analisi_struttura: Dict) -> Dict:
        """Segmenta la trascrizione in unità didattiche"""
        prompt = f"""Sei un esperto di analisi didattica. Analizza questa trascrizione di una lezione universitaria in italiano e segmentala in unità didattiche discrete.

CONTESTO DELLA LEZIONE:
{json.dumps(analisi_struttura, ensure_ascii=False, indent=2)}

TRASCRIZIONE:
"{trascrizione}"

CRITERI DI SEGMENTAZIONE:
1. Ogni unità deve spiegare un concetto completo o un sotto-argomento coerente
2. Ogni unità deve essere comprensibile standalone (con minime integrazioni)
3. Lunghezza target: 300-800 parole (flessibile in base alla coerenza concettuale)
4. Preferisci dividere ai confini naturali (transizioni, cambio argomento)
5. Se un esempio è strettamente legato alla spiegazione, tienili insieme
6. Se un esempio è generico, può essere un'unità separata

Fornisci l'output in formato JSON:

{{
  "unita_concettuali": [
    {{
      "id": "U001",
      "titolo_unita": "Titolo descrittivo dell'unità",
      "testo_riformulato": "Testo riformulato di questa porzione, senza interazioni con partecipanti",
      "concetti_principali": ["concetto X", "concetto Y"],
      "sintesi_punto_chiave": "In 1-2 frasi, cosa impara lo studente da questa unità",
      "domande_studente_tipiche": [
        "Domanda che uno studente potrebbe fare su questo contenuto",
        "Altra domanda..."
      ]
    }}
  ],
  "note_segmentazione": "Eventuali considerazioni sulla divisione effettuata"
}}

IMPORTANTE:
- Riformula il testo tralasciando le interazioni con i partecipanti e focalizzando sui contenuti
- Se incontri una transizione debole, segnalala e spiega la scelta
- Risolvi TUTTI i riferimenti impliciti tipo "questo", "quello", "come detto prima"
- Genera 10-15 domande per ogni unità concettuale
"""
        
        logger.info("→ Segmentazione in unità didattiche...")
        response = litellm.completion(
            model=self.llm_model,
            messages=[{"content": prompt, "role": "user"}],
        )
        
        content = response.choices[0].message.content
        # Pulisci markdown se presente
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    
    # ========== STEP 3: CREAZIONE EMBEDDINGS ==========
    
    def _inizializza_embedding_model(self):
        """Inizializza il modello di embedding"""
        if not self.embedding_model:
            if self.embedding_api_provider:
                # Use API-based embeddings
                logger.info(f"→ Inizializzazione embedding API: {self.embedding_api_provider}/{self.embedding_model_name}")
                self.embedding_model = APIEmbeddings(self.embedding_api_provider, self.embedding_model_name)
            else:
                # Use local embeddings
                logger.info(f"→ Caricamento modello embedding locale: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformerEmbeddings(self.embedding_model_name)
            logger.info("✓ Modello embedding caricato")
    
    def _estrai_voci_per_embedding(
        self, 
        unita_didattiche: List[Dict[str, Any]], 
        source_name: str
    ) -> List[Dict[str, Any]]:
        """Estrae tutte le voci da embedddare dalle unità didattiche"""
        voci = []
        
        for unita in unita_didattiche:
            testo_riformulato = unita.get("testo_riformulato", "")
            id_unita = unita.get("id", "UNKNOWN")
            
            # 1. Testo riformulato principale
            if testo_riformulato:
                voci.append({
                    "testo": testo_riformulato,
                    "metadata": {
                        "text": testo_riformulato,
                        "source": source_name,
                        "id_unita": id_unita,
                        "tipo": "testo_principale",
                        "mimetype": "text/txt"
                    }
                })
            
            # 2. Sintesi punto chiave
            if sintesi := unita.get("sintesi_punto_chiave"):
                voci.append({
                    "testo": sintesi,
                    "metadata": {
                        "text": testo_riformulato,
                        "source": source_name,
                        "id_unita": id_unita,
                        "tipo": "sintesi",
                        "mimetype": "text/txt"
                    }
                })
            
            # 3. Domande studente tipiche
            if domande := unita.get("domande_studente_tipiche"):
                for domanda in domande:
                    voci.append({
                        "testo": domanda,
                        "metadata": {
                            "text": testo_riformulato,
                            "source": source_name,
                            "id_unita": id_unita,
                            "tipo": "domanda",
                            "mimetype": "text/txt"
                        }
                    })
        
        logger.info(f"✓ Estratte {len(voci)} voci per embedding")
        return voci
    
    def _genera_embeddings_batch(
        self, 
        voci: List[Dict[str, Any]], 
        batch_size: int = 32
    ) -> List[Tuple[str, str, List[float], Dict[str, Any]]]:
        """Genera embeddings in batch e prepara documenti"""
        if not voci:
            return []
        
        self._inizializza_embedding_model()
        
        testi = [voce['testo'] for voce in voci]
        all_embeddings = []
        
        logger.info(f"→ Generazione embeddings per {len(testi)} testi...")
        
        # Genera embeddings in batch
        for i in range(0, len(testi), batch_size):
            batch_texts = testi[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"  Processati {min(i + batch_size, len(testi))}/{len(testi)}")
        
        # Prepara documenti per vector store
        documents = []
        for voce, embedding in zip(voci, all_embeddings):
            documents.append((
                voce['testo'],
                voce['metadata']['source'],
                embedding,
                voce['metadata']
            ))
        
        logger.info(f"✓ Generati {len(documents)} embeddings")
        return documents
    
    # ========== STEP 4: SALVATAGGIO IN VECTOR STORE ==========
    
    def _inizializza_vector_store(self):
        """Inizializza il vector store PGVector"""
        if not self.vector_store:
            self._inizializza_embedding_model()
            logger.info(f"→ Inizializzazione vector store: {self.collection_name}")
            self.vector_store = PGVector(
                connection=self.pg_connection,
                collection_name=self.collection_name,
                embeddings=self.embedding_model
            )
            logger.info("✓ Vector store inizializzato")
    
    def _salva_in_vector_store(
        self, 
        documents: List[Tuple[str, str, List[float], Dict[str, Any]]]
    ):
        """Salva documenti nel vector store"""
        if not documents:
            logger.warning("Nessun documento da salvare")
            return
        
        self._inizializza_vector_store()
        
        texts = [doc[0] for doc in documents]
        embeddings = [doc[2] for doc in documents]
        metadatas = [doc[3] for doc in documents]
        
        logger.info(f"→ Salvataggio {len(documents)} documenti nel vector store...")
        self.vector_store.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"✓ Salvati {len(documents)} documenti")
    
    # ========== PIPELINE COMPLETA ==========
    
    def processa_lezione(
        self,
        file_trascrizione: str,
        source_name: str,
        output_dir: str = "output",
        salva_vector_store: bool = True,
        batch_size: int = 32
    ) -> Dict:
        """
        Esegue l'intera pipeline su una lezione.
        
        Args:
            file_trascrizione: Path al file della trascrizione
            source_name: Nome identificativo della lezione
            output_dir: Directory per salvare i file intermedi
            salva_vector_store: Se True, salva nel vector store
            batch_size: Dimensione batch per embedding
            
        Returns:
            Dizionario con tutti i risultati della pipeline
        """
        logger.info("=" * 80)
        logger.info(f"INIZIO PIPELINE: {source_name}")
        logger.info("=" * 80)
        
        # Crea directory output
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        risultati = {
            "source_name": source_name,
            "file_trascrizione": file_trascrizione
        }
        
        try:
            # STEP 1: Carica trascrizione
            logger.info("\n[1/5] Caricamento trascrizione...")
            trascrizione = self._carica_file(file_trascrizione)
            logger.info(f"✓ Caricati {len(trascrizione)} caratteri")
            
            # STEP 2: Analisi strutturale
            logger.info("\n[2/5] Analisi strutturale...")
            analisi = self.analizza_struttura(trascrizione)
            risultati["analisi_struttura"] = analisi
            
            output_analisi = output_path / f"{Path(file_trascrizione).stem}_analisi.json"
            self._salva_json(analisi, str(output_analisi))
            
            # STEP 3: Creazione unità didattiche
            logger.info("\n[3/5] Creazione unità didattiche...")
            unita = self.crea_unita_didattiche(trascrizione, analisi)
            risultati["unita_didattiche"] = unita
            
            output_unita = output_path / f"{Path(file_trascrizione).stem}_unita.json"
            self._salva_json(unita, str(output_unita))
            
            num_unita = len(unita.get("unita_concettuali", []))
            logger.info(f"✓ Create {num_unita} unità didattiche")
            
            # STEP 4: Generazione embeddings
            logger.info("\n[4/5] Generazione embeddings...")
            voci = self._estrai_voci_per_embedding(
                unita.get("unita_concettuali", []), 
                source_name
            )
            documents = self._genera_embeddings_batch(voci, batch_size)
            risultati["num_embeddings"] = len(documents)
            
            # STEP 5: Salvataggio in vector store
            if salva_vector_store and self.pg_connection:
                logger.info("\n[5/5] Salvataggio in vector store...")
                self._salva_in_vector_store(documents)
            else:
                logger.info("\n[5/5] Salvataggio vector store saltato (configurazione)")
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ PIPELINE COMPLETATA CON SUCCESSO")
            logger.info("=" * 80)
            logger.info(f"Unità create: {num_unita}")
            logger.info(f"Embeddings generati: {len(documents)}")
            
            return risultati
            
        except Exception as e:
            logger.error(f"✗ Errore durante la pipeline: {e}", exc_info=True)
            raise
    
    def test_ricerca(self, query: str, k: int = 3):
        """Test di ricerca nel vector store"""
        if not self.vector_store:
            self._inizializza_vector_store()
        
        logger.info(f"\n→ Test ricerca: '{query}'")
        results = self.vector_store.similarity_search(query, k=k)
        
        print("\n" + "=" * 80)
        print(f"RISULTATI PER: '{query}'")
        print("=" * 80)
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.metadata.get('source', 'N/A')}")
            print(f"Tipo: {doc.metadata.get('tipo', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...")
        print("=" * 80)


# ========== ESEMPIO DI UTILIZZO ==========

if __name__ == "__main__":
    # Configurazione
    CONFIG = {
        "llm_model": "gemini/gemini-2.5-flash",
        "embedding_model": "BAAI/bge-m3",
        "pg_connection": "postgresql+psycopg://user:pwd@host/db?sslmode=require&channel_binding=require",
        "collection_name": "Test"
    }
    
    # Inizializza pipeline
    pipeline = PipelineLezioniRAG(**CONFIG)
    
    # Processa una lezione
    risultati = pipeline.processa_lezione(
        file_trascrizione="120anni.txt",
        source_name="120anni",
        output_dir="output",
        salva_vector_store=True,
        batch_size=32
    )
    
    # Test ricerca (opzionale)
    pipeline.test_ricerca("Come si misurano gli indicatori qualitativi?", k=3)
