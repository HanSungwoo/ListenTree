from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os

# Flask 앱 생성
app = Flask(__name__)

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

model="gpt-3.5-turbo"

client = OpenAI(
    api_key = os.environ["OPENAI_API_KEY"]
)
# 사용자 데이터 (간단한 예로 메모리에 저장)
user_data = {
    "conversation_count": 0,
    "tree_stage": "seed",
    "fruit": []
}

# 대화 기능 구현 (GPT-3 호출)
def generate_response(user_input, model, client):
    try:
        messages = [
            {"role": "system", "content": "You are an AI chatbot designed to assist users with a wide range of topics, including but not limited to technology, education, personal advice, and general knowledge. You are friendly, helpful, and professional in your responses. Your goal is to understand the user’s needs, provide clear and concise information, and offer assistance when possible. Always be polite, respectful, and maintain a positive tone. If a user asks for something you are not capable of answering or if it involves personal or sensitive information, inform the user politely and redirect the conversation to topics you can assist with. Make sure your responses are accurate and informative."},
            {"role": "user", "content": {"type": "text", "text": user_input}},
        ]
        answer = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        return answer.choices[0].message['content'].strip()
    
    except Exception as e:
        return f"Error: {str(e)}"

# 나무 성장 단계 결정 함수
def update_tree_stage(conversation_count):
    if conversation_count < 5:
        return "seed"
    elif 5 <= conversation_count < 10:
        return "sapling"
    elif 10 <= conversation_count < 20:
        return "young_tree"
    else:
        return "full_tree"

# 열매 요약 기능 구현 (최근 대화 내용을 요약)
def summarize_conversation(conversations, model, client):
    try:
        combined_text = "\n".join(conversations)
        summary = [
                {"role": "system", "content": "You are an AI chatbot designed to assist users with a wide range of topics, including but not limited to technology, education, personal advice, and general knowledge. You are friendly, helpful, and professional in your responses. Your goal is to understand the user’s needs, provide clear and concise information, and offer assistance when possible. Always be polite, respectful, and maintain a positive tone. If a user asks for something you are not capable of answering or if it involves personal or sensitive information, inform the user politely and redirect the conversation to topics you can assist with. Make sure your responses are accurate and informative."},
                {"role": "user", "content": {"type": "text", "text": f"Summarize this conversation: {combined_text}"}},
            ]
        answer = client.chat.completions.create(
            model=model,
            messages=summary,
            max_tokens=150,
            temperature=0.7
        )
        return answer.choices[0].message['content'].strip()
    
    except Exception as e:
        return f"Error: {str(e)}"

# 라우트 및 로직 설정
@app.route('/')
def index():
    return render_template('index.html', tree_stage=user_data["tree_stage"], fruits=user_data["fruit"])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    response = generate_response(user_input, model, client)
    
    # 대화 횟수 증가 및 나무 성장 업데이트
    user_data["conversation_count"] += 1
    user_data["tree_stage"] = update_tree_stage(user_data["conversation_count"])
    
    # 일정 대화 횟수마다 열매 생성
    if user_data["conversation_count"] % 5 == 0:
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": summarize_conversation([user_input, response], model, client)
        })

    return jsonify({"response": response, "tree_stage": user_data["tree_stage"], "fruits": user_data["fruit"]})

@app.route('/fruit/<int:fruit_id>', methods=['GET'])
def get_fruit_summary(fruit_id):
    fruit = next((f for f in user_data["fruit"] if f["id"] == fruit_id), None)
    if fruit:
        return jsonify({"summary": fruit["summary"]})
    else:
        return jsonify({"error": "Fruit not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
