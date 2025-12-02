from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
from docx import Document
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Permitir requisições do frontend

def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"Erro ao extrair texto do PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    """Extrai texto de um arquivo Word (.docx)"""
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Erro ao extrair texto do Word: {e}")
        return ""

def extract_text_from_txt(txt_file):
    """Extrai texto de um arquivo TXT"""
    try:
        content = txt_file.read()
        # Tentar decodificar com UTF-8, se falhar tentar latin-1
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
        return text.strip()
    except Exception as e:
        print(f"Erro ao extrair texto do TXT: {e}")
        return ""

def extract_text_from_file(file, filename):
    """Extrai texto baseado no tipo de arquivo"""
    file_lower = filename.lower()
    
    if file_lower.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file_lower.endswith('.docx'):
        return extract_text_from_docx(file)
    elif file_lower.endswith('.doc'):
        # .doc antigo não tem suporte direto, mas tentamos como docx
        return extract_text_from_docx(file)
    elif file_lower.endswith('.txt'):
        return extract_text_from_txt(file)
    else:
        return ""

def calculate_similarity(texts):
    """Calcula similaridade entre textos usando TF-IDF e Cosine Similarity"""
    if len(texts) < 2:
        return []
    
    # Criar vetores TF-IDF
    vectorizer = TfidfVectorizer(
        min_df=1,
        stop_words=None,  # Pode adicionar stop words em português se quiser
        ngram_range=(1, 2)  # Usar unigramas e bigramas
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        # Calcular similaridade de cosseno
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except Exception as e:
        print(f"Erro ao calcular similaridade: {e}")
        return []

@app.route('/')
def home():
    return jsonify({
        "message": "CaughtTweaking API está rodando!",
        "status": "online"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint principal para análise de arquivos"""
    
    # Verificar se arquivos foram enviados
    if 'files' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    files = request.files.getlist('files')
    
    if len(files) < 2:
        return jsonify({"error": "Envie pelo menos 2 arquivos"}), 400
    
    if len(files) > 100:
        return jsonify({"error": "Máximo de 100 arquivos permitidos"}), 400
    
    # Extrair texto de todos os arquivos
    documents = []
    file_names = []
    
    print(f"Processando {len(files)} arquivos...")
    
    for file in files:
        file_extension = file.filename.lower().split('.')[-1]
        
        if file_extension in ['pdf', 'docx', 'doc', 'txt']:
            try:
                # Ler arquivo em memória
                file_content = io.BytesIO(file.read())
                text = extract_text_from_file(file_content, file.filename)
                
                if text:
                    documents.append(text)
                    file_names.append(file.filename)
                    print(f"✓ {file.filename} processado ({file_extension.upper()})")
                else:
                    print(f"✗ {file.filename} - sem texto extraído")
            except Exception as e:
                print(f"✗ Erro ao processar {file.filename}: {e}")
    
    if len(documents) < 2:
        return jsonify({"error": "Não foi possível extrair texto de arquivos suficientes"}), 400
    
    # Calcular similaridade
    print("Calculando similaridades...")
    similarity_matrix = calculate_similarity(documents)
    
    if len(similarity_matrix) == 0:
        return jsonify({"error": "Erro ao calcular similaridades"}), 500
    
    # Preparar resultados
    results = []
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            similarity_score = similarity_matrix[i][j] * 100  # Converter para porcentagem
            
            # Só incluir se similaridade >= 40%
            if similarity_score >= 40:
                results.append({
                    "file1": file_names[i],
                    "file2": file_names[j],
                    "similarity": round(similarity_score, 2)
                })
    
    # Ordenar por similaridade (maior primeiro)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"Análise concluída! {len(results)} pares com similaridade >= 40%")
    
    return jsonify({
        "success": True,
        "total_files": len(file_names),
        "comparisons": len(results),
        "results": results
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar saúde da API"""
    return jsonify({
        "status": "healthy",
        "message": "Backend operacional"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
