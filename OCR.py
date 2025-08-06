# ocr_module.py
import pandas as pd
import numpy as np
import easyocr
import faiss
from sentence_transformers import SentenceTransformer
import os
from config import DB_FILE, CACHE_FILE


class OCRManager:
    def __init__(self):
        self.database = self.load_database(DB_FILE)
        if self.database is not None:
            self.model_sbert, self.index = self.precompute_embeddings(self.database, CACHE_FILE)
        else:
            self.model_sbert, self.index = None, None
        self.reader = easyocr.Reader(['en'])

    def load_database(self, file_path):
        """Load medicine database from Excel file"""
        try:
            df = pd.read_excel(file_path, usecols=['name', 'short_composition1'], engine='openpyxl')
            return df
        except Exception as e:
            print(f"Error loading database: {e}")
            return None

    def precompute_embeddings(self, df, cache_file):
        """Precompute embeddings for medicine names"""
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        if os.path.exists(cache_file):
            print("Loading cached embeddings...")
            embeddings = np.load(cache_file)
        else:
            print("Computing embeddings...")
            medicine_names = df['name'].tolist()
            embeddings = np.array(model.encode(medicine_names, convert_to_tensor=False))
            np.save(cache_file, embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return model, index

    def extract_text_from_frame(self, frame):
        """Extract text from frame using OCR"""
        result = self.reader.readtext(frame)
        return ' '.join([text[1] for text in result]).strip()

    def find_closest_match(self, extracted_text):
        """Find closest medicine match for extracted text"""
        if not extracted_text or self.model_sbert is None:
            return "No text detected"

        query_embedding = np.array(self.model_sbert.encode([extracted_text], convert_to_tensor=False))
        _, nearest_idx = self.index.search(query_embedding, 1)
        closest_match = self.database.iloc[nearest_idx[0][0]]['name']
        return closest_match

    def process_frame(self, frame):
        """Process frame for OCR and medicine detection"""
        extracted_text = self.extract_text_from_frame(frame)
        closest_match = self.find_closest_match(extracted_text)
        return {
            "status": "ocr_complete",
            "extracted_text": extracted_text,
            "closest_match": closest_match
        }
