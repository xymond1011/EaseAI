from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer  # sentenceBERT for information retrieval
from sentence_transformers.util import cos_sim  # for getting cosine_similarity between query and corpus
from docx import Document  # *.docx reading
import os, sys, torch, shutil, fitz, traceback, werkzeug.utils  # For secure filenames

# --- Configuration ---
# Determine base directory
if getattr(sys, 'frozen', False):
    basedir = sys._MEIPASS
else:
    basedir = os.path.abspath(os.path.dirname(__file__))

template_dir = os.path.join(basedir, 'templates')
static_dir = os.path.join(basedir, 'static')
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Model Loading ---
LLM_MODEL_NAME = "models/Qwen2.5-CM-DaVinci-Final"
RETRIEVAL_MODEL_NAME = "hkunlp/instructor-large"  # Using instructor model for RAG

tokenizer = None
llm_model = None  # Renamed to llm_model for clarity
retrieval_model = None
models_loaded = False
model_load_error = None

# In-memory store for RAG (stores chunks)
document_knowledge_base = []  # Stores dicts: {"filename": str, "chunk_id": int, "content": str}


def load_models_once(llm_name, retrieval_name):
    global tokenizer, llm_model, retrieval_model, models_loaded, model_load_error
    if not models_loaded and model_load_error is None:
        print(f"Attempting to load models...")
        try:
            print(f"Loading LLM tokenizer from: {llm_name}")
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            print(f"Loading LLM model from: {llm_name}")
            llm_model = AutoModelForCausalLM.from_pretrained(llm_name)
            if torch.cuda.is_available():
                llm_model.to('cuda')
                print("LLM model moved to GPU.")
            else:
                llm_model.to('cpu')
                print("LLM model moved to CPU.")
            llm_model.eval()
            print("LLM model and tokenizer loaded successfully.")

            print(f"Loading Retrieval model: {retrieval_name}")
            retrieval_model = SentenceTransformer(retrieval_name)
            if torch.cuda.is_available():
                retrieval_model.to('cuda')  # SentenceTransformer can also use GPU
                print("Retrieval model moved to GPU.")
            else:
                print("Retrieval model on CPU.")
            print("Retrieval model loaded successfully.")

            models_loaded = True
            print("All models loaded successfully.")
        except Exception as e:
            model_load_error = str(e)
            print(f"FATAL: Error loading models: {e}")
            traceback.print_exc()
    return tokenizer, llm_model, retrieval_model


# --- RAG - Document Processing ---
def extract_text_from_file(filepath):
    filename = os.path.basename(filepath)
    try:
        if filename.lower().endswith(('.txt', '.md')):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif filename.lower().endswith('.pdf'):
            pdf_file = fitz.open(filepath)
            content = ""
            for page_num in range(pdf_file.page_count):  # Corrected loop
                page = pdf_file.load_page(page_num)  # Corrected method call
                content += page.get_text("text") + "\n"
            pdf_file.close()
            return content
        elif filename.lower().endswith(('.doc', '.docx')):
            doc = Document(filepath)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
        else:
            print(f"Unsupported file type for text extraction: {filename}")
            return f"[Content of non-text file: {filename}]"  # Or return None
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        traceback.print_exc()
        return None


def split_text_into_chunks(text, chunk_size=200, chunk_overlap=120):  # Words, approx.
    """
    Simple text chunking. For production, consider more sophisticated methods.
    This is a basic splitter, you might want to use libraries like Langchain's
    RecursiveCharacterTextSplitter for better semantic chunking.
    """
    if not text:
        return []
    words = text.split()  # Simple split by space
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = min(current_pos + chunk_size, len(words))
        chunks.append(" ".join(words[current_pos:end_pos]))
        current_pos += chunk_size - chunk_overlap
        if current_pos >= end_pos and end_pos < len(words):  # ensure progress if overlap is large
            current_pos = end_pos
    return [chunk for chunk in chunks if chunk.strip()]


def add_document_to_knowledge_base(filepath):
    global document_knowledge_base
    text = extract_text_from_file(filepath)
    if text:
        chunks = split_text_into_chunks(text)
        if not chunks:
            print(f"No chunks generated for '{os.path.basename(filepath)}'. Text might be too short or empty.")
            return False
        for i, chunk_content in enumerate(chunks):
            document_knowledge_base.append({
                "filename": os.path.basename(filepath),
                "chunk_id": f"{os.path.basename(filepath)}_chunk_{i}",  # Unique ID for chunk
                "content": chunk_content
            })
        print(f"Added {len(chunks)} chunks from '{os.path.basename(filepath)}' to knowledge base.")
        # In a real system, you would now create embeddings for these chunks and store them in a vector DB.
        return True
    else:
        print(f"Could not extract text from '{os.path.basename(filepath)}'.")
    return False


def get_relevant_context_from_kb(query, top_k=3):
    global document_knowledge_base, retrieval_model
    if not document_knowledge_base or not retrieval_model:
        print("Knowledge base is empty or retrieval model not loaded.")
        return ""

    corpus_chunks_data = document_knowledge_base  # List of dicts
    if not corpus_chunks_data:
        return ""

    corpus_contents = [doc["content"] for doc in corpus_chunks_data]

    # Embed query and corpus chunks
    # Using prompts specific to instructor-large model
    query_instruction = "Represent the user's question for retrieving relevant event planning document sections: "
    corpus_instruction = "Represent the event planning document section for retrieval: "

    query_embedding = retrieval_model.encode(query_instruction + query, convert_to_tensor=True)
    corpus_embeddings = retrieval_model.encode([corpus_instruction + content for content in corpus_contents],
                                               convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_scores = cos_sim(query_embedding, corpus_embeddings)

    # Get top_k results
    # Ensure k is not greater than the number of chunks
    actual_top_k = min(top_k, len(corpus_contents))
    if actual_top_k == 0:
        return ""

    top_results = torch.topk(cosine_scores.squeeze(0), k=actual_top_k)

    relevant_chunks_text = []
    print(f"\n--- Top {actual_top_k} Relevant Chunks ---")
    for score, idx in zip(top_results.values, top_results.indices):
        chunk_info = corpus_chunks_data[idx]
        relevant_chunks_text.append(chunk_info["content"])
        print(f"  Score: {score:.4f}, File: {chunk_info['filename']}, Chunk ID: {chunk_info['chunk_id']}")
        # print(f"  Content: {chunk_info['content'][:100]}...") # For debugging
    print("----------------------------")

    return "\n\n---\n\n".join(relevant_chunks_text)  # Concatenate content of relevant chunks


# --- Generate Response Function (RAG-aware) ---
def generate_response(prompt, context="", max_new_tokens=512, temperature=0.7, top_k=30, top_p=0.95):
    global tokenizer, llm_model, models_loaded, model_load_error

    if model_load_error:
        return f"Error: Models are not loaded due to a previous error ({model_load_error}). Please check server logs."
    if not models_loaded or not tokenizer or not llm_model:
        return "Error: Models are not loaded. Please check server logs."

    try:
        # Determine model_max_length
        model_max_input_length = getattr(tokenizer, 'model_max_length', 2048)

        # Construct prompt with context if available
        if context:
            full_prompt = (
                "You are EaseAI, created by Team Vamonos & Balmry - Saint Louis University Baguio. "
                "You are an intelligent AI assistant for event planning, designed to assist clients. "
                "Strictly use ONLY the information from the 'Context from documents' provided below to answer the 'User's question'. "
                "Do NOT repeat the user's question or the provided context in your answer. Provide a direct and concise answer. "
                "If the information needed to answer the question is not present in the context, clearly state that the information is not available in the provided documents.\n\n"
                "Context from documents:\n"
                f"{context}\n\nUser's question: {prompt}\n\nAnswer:"
            )
        else:
            full_prompt = (
                "You are EaseAI, created by Team Vamonos & Balmry - Saint Louis University Baguio. "
                "You are an intelligent AI assistant for event planning, designed to assist clients. "
                "Answer the user's question directly and concisely. Do not repeat the question in your answer.\n\n"
                f"User's question: {prompt}\n\nAnswer:"
            )

        # Truncate context if full_prompt is too long (very basic approach)
        # A more sophisticated approach would involve token-aware truncation of context.
        # This is a simple string length check, actual token length is what matters.
        # We rely on the tokenizer's truncation for final fit.
        MAX_PROMPT_CHAR_LEN = model_max_input_length * 3  # Heuristic: average 3 chars per token
        if len(full_prompt) > MAX_PROMPT_CHAR_LEN:
            print(
                f"Warning: Full prompt string length ({len(full_prompt)}) is very long. May be truncated by tokenizer.")

        print(f"\n--- Full prompt to model (approx {len(full_prompt)} chars) ---")
        # print(full_prompt) # Uncomment for debugging full prompt
        print("---------------------------------------------------")

        encoded_input = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,  # Crucial: truncates if too long for the model
            max_length=model_max_input_length - max_new_tokens,  # Reserve space for generation
            padding="longest"  # Pad to the longest sequence in the batch (or max_length if single)
        )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        print(f"Number of tokens in input_ids to LLM: {input_ids.shape[1]}")
        if input_ids.shape[1] >= model_max_input_length - max_new_tokens:
            print(
                f"WARNING: Input to LLM is at or near max_length allowed for input based on tokenizer.model_max_length ({model_max_input_length}) minus max_new_tokens.")

        input_ids = input_ids.to(llm_model.device)
        attention_mask = attention_mask.to(llm_model.device)

        with torch.no_grad():
            # The output_ids will contain the input_ids followed by the generated_ids
            output_ids_full_sequence = llm_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,  # Important for temperature, top_k, top_p to have effect
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Slice to get only the generated tokens (excluding the input prompt)
        input_token_length = input_ids.shape[1]
        # output_ids_full_sequence[0] contains all tokens (input + generated)
        generated_token_ids = output_ids_full_sequence[0][input_token_length:]

        response = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        print(f"\n--- Model's Generated Text (after slicing prompt) ---\n{response}\n---------------------------------")

        # Optional: Further clean-up for incomplete sentences (might be less necessary now)
        if response and response[-1] not in ['.', '!', '?', '"', "'", ')', ';', ':']:
            last_sentence_end = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_sentence_end > -1 and last_sentence_end < len(response) - 5:  # Check if it's a real end
                # response = response[:last_sentence_end+1] # Be careful with this, could cut off valid short answers
                pass

        return response if response else "I received your prompt, but I couldn't generate a specific answer based on the provided documents."

    except Exception as e:
        print(f"Error during response generation: {e}")
        traceback.print_exc()
        return "Sorry, I encountered an error while generating a response."


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_documents", methods=["POST"])
def handle_upload_documents():
    if 'files[]' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    files = request.files.getlist('files[]')

    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400

    successful_uploads = 0
    errors = []

    for file_obj in files:  # Renamed 'file' to 'file_obj' to avoid conflict
        if file_obj and file_obj.filename:
            try:
                filename = werkzeug.utils.secure_filename(file_obj.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_obj.save(save_path)

                if add_document_to_knowledge_base(save_path):
                    successful_uploads += 1
                else:
                    errors.append(f"Could not process (extract text or chunk) {filename}.")

            except Exception as e:
                errors.append(f"Error saving or processing {file_obj.filename}: {str(e)}")
                print(f"Error processing file {file_obj.filename}: {e}")
                traceback.print_exc()

    if successful_uploads > 0 and not errors:
        return jsonify({"message": f"{successful_uploads} document(s) uploaded and processed successfully."}), 200
    elif successful_uploads > 0 and errors:
        return jsonify({
            "message": f"{successful_uploads} document(s) uploaded. Some errors occurred.",
            "errors": errors
        }), 207
    else:
        return jsonify({
            "error": "Failed to upload or process any documents.",
            "details": errors
        }), 500


@app.route("/chat", methods=["POST"])
def chat():
    global models_loaded, model_load_error
    if model_load_error:
        return jsonify({"error": f"Models not loaded: {model_load_error}"}), 500
    if not models_loaded:
        return jsonify({"error": "Models are not loaded. Cannot process chat request."}), 500

    data = request.get_json()
    user_prompt = data.get("prompt", "").strip()

    if not user_prompt:
        return jsonify({"error": "Empty prompt"}), 400

    print(f"\nReceived user prompt: '{user_prompt}'")

    # RAG Step: Get relevant context
    retrieved_context = get_relevant_context_from_kb(user_prompt, top_k=3)  # Get top 3 chunks
    if retrieved_context:
        print(
            f"\n--- Retrieved Context for RAG (concatenated from chunks) ---\n{retrieved_context[:500]}...\n--------------------------------------------------\n")
    else:
        print("No specific context retrieved from knowledge base for this query.")

    response_text = generate_response(user_prompt, context=retrieved_context)
    print(f"\n '{response_text}'")
    return jsonify({"response": response_text})


@app.route('/clear_knowledge_base', methods=['POST'])
def clear_knowledge_base_route():
    global document_knowledge_base
    # Clear in-memory knowledge base
    document_knowledge_base = []
    print("In-memory knowledge base (chunks) cleared.")

    # Clear uploaded files from disk
    upload_dir = app.config['UPLOAD_FOLDER']  # Use configured upload folder
    try:
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"Uploaded files in '{upload_dir}' cleared.")
        else:
            print(f"Upload directory '{upload_dir}' does not exist. No files to clear from disk.")
        # Recreate if it was removed (though above loop only removes contents)
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

    except Exception as e:
        print(f"Error clearing uploaded files: {e}")
        traceback.print_exc()
        return jsonify({"message": "Knowledge base cleared, but error clearing uploaded files."}), 500

    return jsonify({"message": "Knowledge base and uploaded documents cleared."}), 200


# --- Run App ---
if __name__ == "__main__":
    print("-----------------------------------------------------")
    print("Starting EaseAI Assistant backend...")
    load_models_once(LLM_MODEL_NAME, RETRIEVAL_MODEL_NAME)

    if not models_loaded:
        print(f"WARNING: AI Models FAILED TO LOAD. Error: {model_load_error}")
        print("The chat functionality will be severely affected or non-functional.")
    else:
        print("AI Models loaded. Backend is ready.")
    print(f"Uploads will be saved to: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Knowledge base is currently in-memory.")
    print("-----------------------------------------------------")

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)