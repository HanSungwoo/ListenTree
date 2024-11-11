from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os

# Flask 앱 생성
app = Flask(__name__)

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "your-api-key"

model = "gpt-4o-mini"
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

# 사용자 데이터 (간단한 예로 메모리에 저장)
user_data = {
    "conversation_count": 0,
    "tree_stage": "seed",
    "fruit": [],
    "chat_history": []  # 새로운 대화 기록을 저장하는 리스트
}

# 대화 기능 구현 (GPT 호출)
def generate_response(user_input):
    try:
        messages = [
            {"role": "system",
             "content": "You are an empathetic AI chatbot that assists users across diverse topics, responding in Korean with a friendly and professional tone. Keep answers short, under 50 tokens, and focused on understanding and reflecting users' feelings without providing solutions."},
            {"role": "user", "content": user_input},
        ]
        answer = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        return answer.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"


# 나무 성장 단계 결정 함수
def update_tree_stage(conversation_count):
    tree_set = ["seed", "sapling", "young_tree", "mid_tree", "full_tree"]
    return tree_set[conversation_count // 50]


# 열매 요약 기능 구현 (최근 대화 내용을 요약)
def summarize_conversation(conversations):
    try:
        # 사용자 메시지만 추출
        user_messages = [entry['user'] for entry in conversations]
        combined_text = "\n".join(user_messages)
        summary_prompt = f"Summarize these user messages: {combined_text}"
        messages = [
            {"role": "system",
             "content": "You are an AI chatbot designed to assist users with a wide range of topics, including but not limited to technology, education, personal advice, and general knowledge. You can only respond in Korean with a maximum of 100 tokens. You summarize user's text, not provide an answer. "},
            {"role": "user", "content": summary_prompt}
        ]
        answer = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        return answer.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"

# 라우트 및 로직 설정
@app.route('/')
def index():
    return render_template('index.html', tree_stage=user_data["tree_stage"], fruits=user_data["fruit"], chat_history=user_data["chat_history"])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = generate_response(user_input)

    # 대화 기록을 저장
    user_data["chat_history"].append({"user": user_input, "bot": response})

    # 대화 횟수 증가 및 나무 성장 업데이트
    user_data["conversation_count"] += 1
    user_data["tree_stage"] = update_tree_stage(user_data["conversation_count"])

    # 요약 생성 코드 (50회마다 요약)
    if user_data["conversation_count"] % 50 == 0:
        last_5_user_messages = [{"user": entry['user']} for entry in user_data["chat_history"][-5:]]
        fruit_summary = summarize_conversation(last_5_user_messages)
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": fruit_summary  # 요약된 텍스트
        })

    return jsonify({"response": response, "tree_stage": user_data["tree_stage"], "fruits": user_data["fruit"], "chat_history": user_data["chat_history"]})


@app.route('/fruit/<int:fruit_id>', methods=['GET'])
def get_fruit_summary(fruit_id):
    fruit = next((f for f in user_data["fruit"] if f["id"] == fruit_id), None)
    if fruit:
        return jsonify({"summary": fruit["summary"]})
    else:
        return jsonify({"error": "Fruit not found"}), 404


@app.route('/fruit', methods=['GET'])
def get_all_fruit_summaries():
    return jsonify(user_data["fruit"])


if __name__ == '__main__':
    app.run(debug=True)
