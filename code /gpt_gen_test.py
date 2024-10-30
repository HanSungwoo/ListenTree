from flask import Flask, render_template, request, jsonify
import openai

# Flask 앱 생성
app = Flask(__name__)

# GPT API 키 설정 (환경변수로 설정하거나 직접 입력 가능)
openai.api_key = "YOUR_API_KEY"

# 사용자 데이터 (간단한 예로 메모리에 저장)
user_data = {
    "conversation_count": 0,
    "tree_stage": "seed",
    "fruit": []
}

# 대화 기능 구현 (GPT-3 호출)
def generate_response(user_input):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_input,
            max_tokens=150
        )
        return response.choices[0].text.strip()
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
def summarize_conversation(conversations):
    try:
        combined_text = "\n".join(conversations)
        summary = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize this conversation: {combined_text}",
            max_tokens=50
        )
        return summary.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# 라우트 및 로직 설정
@app.route('/')
def index():
    return render_template('index.html', tree_stage=user_data["tree_stage"], fruits=user_data["fruit"])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    response = generate_response(user_input)
    
    # 대화 횟수 증가 및 나무 성장 업데이트
    user_data["conversation_count"] += 1
    user_data["tree_stage"] = update_tree_stage(user_data["conversation_count"])
    
    # 일정 대화 횟수마다 열매 생성
    if user_data["conversation_count"] % 5 == 0:
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": summarize_conversation([user_input, response])
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
