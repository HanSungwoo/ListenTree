from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os


# Flask 앱 생성
app = Flask(__name__)

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "api-key"

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
    if conversation_count < 10:
        return "seed"
    elif conversation_count < 30:
        return "sapling"
    elif conversation_count < 60:
        return "small_tree"
    elif conversation_count < 100:
        return "large_tree"
    elif conversation_count < 150:
        return "fruit_tree"
    else:
        return 'end'


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

def emotion_recognition(user_input):
    # GPT 모델에 입력할 프롬프트 정의
    prompt = f"""
    The following is a sentence: "{user_input}"
    Please determine the emotion expressed in this sentence. Possible emotions are:
    [angry, sad, happy, neutral, surprise, worry]

    Return only the emotion label.
    """

    # OpenAI API 요청
    response = client.chat.completions.create(
        model=model,  # 또는 gpt-4
        messages=[
            {"role": "system", "content": "You are an assistant that labels emotions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10  # 감정 라벨만 반환할 것이므로 짧게 설정
    )

    # 응답에서 감정 라벨 추출.choices[0].message.content.strip()
    emotion = response.choices[0].message.content.strip()
    return emotion

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

    # 열매 나무가 완전히 성장한 후 (end 상태) 열매 수확 및 초기화
    if user_data["tree_stage"] == "end":
        # 첫 번째 열매 (0~49까지의 대화 요약)
        first_fruit_summary = summarize_conversation(
            [{"user": entry['user']} for entry in user_data["chat_history"][:50]])
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": first_fruit_summary
        })

        # 두 번째 열매 (50~99까지의 대화 요약)
        second_fruit_summary = summarize_conversation(
            [{"user": entry['user']} for entry in user_data["chat_history"][50:100]])
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": second_fruit_summary
        })

        # 세 번째 열매 (100~149까지의 대화 요약)
        third_fruit_summary = summarize_conversation(
            [{"user": entry['user']} for entry in user_data["chat_history"][100:150]])
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": third_fruit_summary
        })

        # 모든 열매 수확 후 초기화
        user_data["conversation_count"] = 0
        user_data["tree_stage"] = "seed"

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


# 감정 분석을 수행하는 API 엔드포인트
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    user_input = request.json['message']
    emotion = emotion_recognition(user_input)
    return jsonify({'emotion': emotion})


if __name__ == '__main__':
    app.run(debug=True)
