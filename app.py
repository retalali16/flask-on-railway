from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import os

# Initialize Flask app and CORS
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load BERT tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
except Exception as e:
    print(f"[ERROR] Failed to load BERT model: {e}")
    raise

# Define Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

    def forward(self, input1, input2):
        return torch.nn.functional.cosine_similarity(input1, input2, dim=1)

# Generate BERT embeddings
def get_bert_embeddings(text):
    try:
        if isinstance(text, str):
            text = [text]
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        raise

# Compare questions
def compare_questions(q1, q2):
    try:
        emb1 = get_bert_embeddings(q1)
        emb2 = get_bert_embeddings(q2)
        similarity = siamese_network(emb1, emb2)
        return similarity.item()
    except Exception as e:
        print(f"[ERROR] Failed to compare questions: {e}")
        return 0.0

# Instantiate Siamese network
try:
    siamese_network = SiameseNetwork()
    siamese_network.eval()
except Exception as e:
    print(f"[ERROR] Failed to initialize Siamese network: {e}")
    raise

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({"error": "No message received"}), 400

        predefined_answers = {
            "What are the admission requirements for Zewail City?": "Applicants must have a high school diploma (Thanaweya Amma or equivalent) with a strong academic record in relevant subjects. Additional requirements include passing entrance exams and an interview.",
            "How can I apply to Zewail City?": "You can apply through the official Zewail City website by filling out the online application form, uploading required documents, and paying the application fee.",
            "What are the available programs at Zewail City?": "Zewail City offers programs in Engineering, Computer Science, Nanotechnology, Biomedical Sciences, and other advanced scientific fields.",
            "Is there a scholarship program at Zewail City?": "Yes, Zewail City offers merit-based and need-based scholarships for outstanding students who meet specific academic and financial criteria.",
            "What is the tuition fee for undergraduate programs?": "Tuition fees vary by program. The latest fee structure is available on the official Zewail City website.",
            "What is the deadline for admission applications?": "Admission deadlines are announced on the university's website and social media channels. It is advised to apply early.",
            "Do international students qualify for admission?": "Yes, international students can apply and must meet equivalent academic and English proficiency requirements.",
            "What entrance exams are required for admission?": "Students may be required to take an aptitude test in mathematics, physics, or other subjects depending on their chosen program.",
            "How do I contact the admission office?": "You can contact the admission office via email at admissions@zewailcity.edu.eg or by phone through the numbers listed on the university website.",
            "Is there student accommodation available?": "Yes, Zewail City provides on-campus accommodation for students, subject to availability."
        }

        best_answer = "Sorry, I didn’t understand that."
        highest_score = -1

        for question, answer in predefined_answers.items():
            score = compare_questions(user_message, question)
            if score > highest_score:
                highest_score = score
                best_answer = answer

        return jsonify({"reply": best_answer})
    except Exception as e:
        print(f"[ERROR] Chat endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    try:
        if not os.path.exists(os.path.join('static', path)):
            return jsonify({"error": f"Static file {path} not found"}), 404
        return send_from_directory('static', path)
    except Exception as e:
        print(f"[ERROR] Static file error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Serve the frontend
@app.route('/')
def serve_frontend():
    return render_template('index.html')

# Health check
@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True)