from flask import Flask, render_template, request, jsonify, url_for
from openai import OpenAI
import json
import os
import pandas as pd
import random

FRUITS_FILE_PATH = "listen_tree/database/fruits.jsonl"
FRUITS_STORAGE_FILE_PATH = "listen_tree/database/fruits_storage_items.jsonl"

fruits_items = ["apple", "grape", "pear"]

# JSONL 데이터를 읽어오는 함수
def load_jsonl_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line.strip()) for line in file if line.strip()]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# JSONL 데이터 로드
fruits_data = load_jsonl_data(FRUITS_FILE_PATH)
fruits_storage_data = load_jsonl_data(FRUITS_STORAGE_FILE_PATH)

diary_example = pd.read_csv('listen_tree/diary.csv', header=None, names=['sentence'])

# 텍스트를 분할하여 저장
diary_text = diary_example['sentence'].tolist()
# Flask 앱 생성
app = Flask(__name__)

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "key"

model = "gpt-4o-mini"
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

# 사용자 데이터 (간단한 예로 메모리에 저장) #테스팅
user_data = {
    "conversation_count": 0,
    "tree_stage": "seed",
    "fruit": [],
    "chat_history": [],
    "fruits": fruits_data,
    "fruits_storage": fruits_storage_data,
}

def sentiment_classification(summary):
    prompt = f"""
    The following is a summary of user messages: "{summary}"
    Please classify the sentiment of this summary as either 'positive' or 'negative'.

    Respond only with 'positive' or 'negative'.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that classifies sentiment."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

# 대화 기능 구현 (GPT 호출)
def generate_response(user_input, model = model, client = client):
    try:
        # 최근 대화 5개를 사용
        # conversation_history = [{"role": "user", "content": user_input}]
        # # 이전 대화가 있으면 추가
        # if len(user_data["chat_history"]) > 0:
        #     for entry in user_data["chat_history"][-5:]:  # 최근 5개의 대화만 사용
        #         conversation_history.append({"role": "assistant", "content": entry['bot']})
        messages = [
            {"role": "system",
             "content": '''당신은 사용자의 이야기를 들어주는 나무입니다. 사용자의 말에 과하지 않은 공감과 반응을 합니다. 모든 응답은 반드시 20토큰 이내로 작성합니다. 이전 응답에서 질문했다면, 다음응답에서 질문하지 않습니다. 이전 응답의 내용을 다음 응답에서 똑같이 얘기하지 않습니다.
- 입력 예시: 오늘은 정말 힘든일이 있었어.
- 출력 예시: 그랬구나, 얘기해볼래?'''},
        ]
        # messages.extend(conversation_history)

        answer = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=30,
            temperature=0.7
        )
        return answer.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"


# 나무 성장 단계 결정 함수
def update_tree_stage(conversation_count):
    if conversation_count < 1:
        return "seed"
    elif conversation_count < 3:
        return "sapling"
    elif conversation_count < 6:
        return "small_tree"
    elif conversation_count < 10:
        return "large_tree"
    elif conversation_count < 15:
        return "fruit_tree"
    else:
        return "end"


# 열매 요약 기능 구현 (최근 대화 내용을 요약)
def summarize_conversation(conversations):
    try:
        # conversations가 문자열 리스트인지 확인
        if isinstance(conversations[0], str):
            user_messages = conversations  # conversations 자체가 문자열 리스트인 경우
        else:
            user_messages = [entry['user'] for entry in conversations]  # 딕셔너리 리스트인 경우

        # 문자열 리스트를 합치기
        combined_text = "\n".join(user_messages)
        summary_prompt = f"Summarize these user messages: {combined_text}"

        messages = [
            {"role": "system",
             "content": "You are an AI chatbot designed to assist users with a wide range of topics, including but not limited to technology, education, personal advice, and general knowledge. You can only respond in Korean with a maximum of 50 tokens. You summarize user's text, not provide an answer. "},
            {"role": "user", "content": summary_prompt}
        ]
        answer = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=50,
            temperature=0.6
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

@app.route('/generate_fruit_images', methods=['POST'])
def generate_fruit_images():
    fruit_summaries = [fruit["summary"] for fruit in user_data["fruit"]]
    results = []
    random_fruit = random.choice(fruits_items)  # 랜덤 과일 선택
    for summary in fruit_summaries: 
        sentiment = sentiment_classification(summary)  # 감정 분류
        # 감정에 따라 이미지 경로 설정
        if sentiment == 'positive':
            image_path = f"/static/fruit/{random_fruit}.png"
        else:
            image_path = f"/static/fruit_bad/{random_fruit}.png"

        results.append({"summary": summary, "image_path": image_path, "sentiment": sentiment})
    
    print("Generated fruit images:", results)  # 데이터 확인
    return jsonify(results)


# 라우트 및 로직 설정
@app.route('/')
def index():
    return render_template(
        'index.html',
        tree_stage=user_data["tree_stage"],
        fruits=user_data["fruit"],
        chat_history=user_data["chat_history"],
    )
@app.route('/fruits', methods=['GET'])
def get_fruits():
    """도감 데이터를 반환"""
    return jsonify(user_data["fruits"])

@app.route('/fruits_storage', methods=['GET'])
def get_fruits_storage():
    """보관함 데이터를 반환"""
    return jsonify(user_data["fruits_storage"])

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
        first_fruit_summary = summarize_conversation(user_data["chat_history"][:5])
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": first_fruit_summary
        })

        second_fruit_summary = summarize_conversation(user_data["chat_history"][5:10])
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": second_fruit_summary
        })

        third_fruit_summary = summarize_conversation(user_data["chat_history"][10:15])
        user_data["fruit"].append({
            "id": len(user_data["fruit"]) + 1,
            "summary": third_fruit_summary
        })

        # 모든 열매 수확 후 초기화

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

@app.route('/save_fruit', methods=['POST'])
def save_fruit():
    fruit_id = request.json['fruit_id']
    # 해당 fruit_id를 보관함에 저장하는 로직을 구현
    return jsonify({"status": "saved", "fruit_id": fruit_id})

@app.route('/reset_tree', methods=['POST'])
def reset_tree():
    # user_data 초기화
    user_data["chat_history"] = []
    user_data["conversation_count"] = 0
    user_data["tree_stage"] = "seed"
    user_data["fruit"] = []
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)