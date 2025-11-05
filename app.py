import streamlit as st
import pandas as pd
import os
import json
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from youtubesearchpython import VideosSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
LOG_FILE = "user_logs.csv"
MAX_QUESTIONS = 6

# -----------------------------
# OPENAI SETUP
# -----------------------------
def setup_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API key not found! Please add OPENAI_API_KEY to your .env file")
        st.info("Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        st.stop()
    return OpenAI(api_key=api_key)

# -----------------------------
# INITIAL LOG FILE SETUP
# -----------------------------
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["topic", "difficulty", "answered_correct", "choice", "question_id"]).to_csv(LOG_FILE, index=False)

# -----------------------------
# LOG USER ANSWERS
# -----------------------------
def log_result(topic, difficulty, answered_correct, choice, question_id):
    df_log = pd.read_csv(LOG_FILE)
    df_log = pd.concat([df_log, pd.DataFrame([{
        "topic": topic,
        "difficulty": difficulty,
        "answered_correct": answered_correct,
        "choice": choice,
        "question_id": question_id
    }])], ignore_index=True)
    df_log.to_csv(LOG_FILE, index=False)

# -----------------------------
# GENERATE 6TH GRADE WORLD WARS QUESTIONS WITH OPENAI
# -----------------------------
def generate_world_wars_question(client, topic, difficulty):
    """Generate age-appropriate World Wars questions for 6th graders"""
    try:
        if difficulty == "Easy":
            complexity = "simple facts, basic dates, and well-known leaders. Use vocabulary suitable for 11-12 year olds."
        elif difficulty == "Medium":
            complexity = "connections between events, basic causes and effects. Keep language clear for 6th graders."
        else:  # Hard
            complexity = "comparisons between wars, understanding consequences. Still use 6th grade appropriate language."
        
        prompt = f"""
        You are creating a history quiz question about {topic} for 6th grade students (ages 11-12).
        
        Create a {difficulty} level multiple choice question that focuses on {complexity}
        
        Topics to focus on:
        - World War I: Basic causes, major countries involved, key dates (1914-1918), famous leaders
        - World War II: Hitler, major battles, countries involved, dates (1939-1945), Holocaust (age-appropriate)
        - Key figures: Roosevelt, Churchill, Hitler, Stalin (basic roles)
        - Important events: Pearl Harbor, D-Day, end of wars
        
        Guidelines:
        - Use simple, clear language appropriate for 6th graders
        - Focus on major facts and well-known events
        - Avoid overly complex political concepts
        - Keep content educational but age-appropriate
        
        Return ONLY a valid JSON object with this exact structure:
        {{
            "question": "Your 6th-grade appropriate question here",
            "option1": "First option",
            "option2": "Second option", 
            "option3": "Third option",
            "option4": "Fourth option",
            "correct_answer": 2,
            "explanation": "Simple explanation suitable for 6th graders"
        }}
        
        The correct_answer should be a number (1-4) indicating which option is correct.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        question_json = response.choices[0].message.content.strip()
        # Clean up any potential markdown formatting
        if question_json.startswith("```"):
            question_json = question_json.replace("```json", "").replace("```", "")
        
        return json.loads(question_json)
    except Exception as e:
        st.error(f"Error generating question: {str(e)}")
        return None

# -----------------------------
# GET 6TH GRADE APPROPRIATE YOUTUBE RECOMMENDATIONS WITH OPENAI
# -----------------------------
def get_educational_youtube_recommendation(client, topic, difficulty, user_performance):
    """Get age-appropriate educational videos for 6th graders"""
    try:
        performance_context = "having some difficulty" if user_performance < 0.5 else "doing well"
        
        prompt = f"""
        A 6th grade student (age 11-12) is {performance_context} with {difficulty} level questions about {topic}.
        
        Generate 3 educational YouTube search queries that would help them learn better.
        Focus on:
        - Educational channels suitable for middle school students
        - Age-appropriate historical content
        - Visual and engaging historical documentaries
        - Simple explanations of {topic}
        
        Make the search queries specific and educational. No inappropriate content.
        
        Return only the search queries, one per line, without numbering.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        queries = response.choices.message.content.strip().split('\n')
        return [q.strip() for q in queries if q.strip()][:3]
    except Exception as e:
        st.error(f"Error getting YouTube recommendations: {str(e)}")
        return [f"{topic} for kids", "World Wars simple explanation", "History for middle school"]

def fetch_youtube_video(query):
    """Fetch YouTube video using search query"""
    try:
        search = VideosSearch(query, limit=1)
        result = search.result()
        if result["result"]:
            return result["result"]["link"]
    except Exception:
        pass
    return None

# -----------------------------
# TRAIN ADAPTIVE MODEL
# -----------------------------
def train_adaptive_model():
    log_data = pd.read_csv(LOG_FILE)
    if len(log_data) < 3 or log_data['answered_correct'].nunique() < 2:
        return None, None
    
    X = log_data[["topic", "difficulty"]]
    y = log_data["answered_correct"]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_enc = enc.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_enc, y)
    return model, enc

# -----------------------------
# SELECT ADAPTIVE DIFFICULTY FOR WORLD WARS
# -----------------------------
def select_adaptive_difficulty(model, enc):
    """Select difficulty level based on student performance"""
    if model is None or enc is None:
        return "World War I", "Easy"  # Start with basics
    
    topics = ["World War I", "World War II", "World Wars General"]
    difficulties = ["Easy", "Medium", "Hard"]
    predictions = {}
    
    for topic in topics:
        for difficulty in difficulties:
            try:
                sample = pd.DataFrame({"topic": [topic], "difficulty": [difficulty]})
                X_enc = enc.transform(sample)
                prob = model.predict_proba(X_enc)[0][1]  # probability of correct answer
                predictions[f"{topic}_{difficulty}"] = prob
            except:
                predictions[f"{topic}_{difficulty}"] = 0.5
    
    # FIX START: proper unpacking
    weakest_key = min(predictions, key=predictions.get)
    topic, difficulty = weakest_key.split("_", 1)
    # FIX END
    return topic, difficulty

# -----------------------------
# GET FINAL RECOMMENDATIONS WITH OPENAI
# -----------------------------
def get_final_recommendations(client, final_score, total_questions):
    """Get personalized study recommendations based on overall performance"""
    try:
        performance_pct = (final_score / total_questions) * 100
        
        if performance_pct >= 80:
            level_desc = "excellent work"
            focus = "advanced World Wars topics and connections between events"
        elif performance_pct >= 60:
            level_desc = "good understanding but room for improvement"
            focus = "reviewing key facts and important events"
        else:
            level_desc = "need to strengthen your foundation"
            focus = "basic World Wars facts, dates, and major figures"
        
        prompt = f"""
        A 6th grade student just completed a World Wars quiz with {final_score}/{total_questions} correct ({performance_pct:.0f}%).
        They show {level_desc} and should focus on {focus}.
        
        Generate 5 specific, educational YouTube search queries that would help this student improve their World Wars knowledge.
        
        Focus on:
        - Age-appropriate educational content for 6th graders
        - Engaging historical videos and documentaries
        - Visual learning materials
        - Content that matches their current level
        
        Return only the search queries, one per line, without numbering.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        queries = response.choices.message.content.strip().split('\n')
        return [q.strip() for q in queries if q.strip()][:5]
    except Exception as e:
        st.error(f"Error getting final recommendations: {str(e)}")
        return [
            "World War 1 for kids",
            "World War 2 simple explanation", 
            "History of World Wars animated",
            "Famous World War leaders for students",
            "World Wars timeline for middle school"
        ]

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(
    page_title="World Wars Quiz for 6th Graders",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for 6th grade friendly design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .final-score {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 World Wars History Quiz</h1>
    <h3>For 6th Grade Students</h3>
    <p>Learn about World War I and World War II with AI-powered questions!</p>
    <p><strong>📝 6 Questions Total</strong></p>
</div>
""", unsafe_allow_html=True)

# Setup OpenAI client
client = setup_openai()

# Session state initialization
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = set()
if "score" not in st.session_state:
    st.session_state.score = 0
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "answered" not in st.session_state:
    st.session_state.answered = False
if "num_questions" not in st.session_state:
    st.session_state.num_questions = 0

# Sidebar with learning tips
st.sidebar.markdown("### 📚 Learning Tips")
st.sidebar.markdown("""
- **Take your time** reading each question
- **Think** about what you learned in class
- **Learn from mistakes** - check explanations!
- **Watch videos** to learn more
""")

st.sidebar.markdown("### 🎯 Quiz Progress")
if st.session_state.num_questions > 0:
    accuracy = (st.session_state.score / st.session_state.num_questions) * 100
    st.sidebar.metric("Current Score", f"{st.session_state.score}/{st.session_state.num_questions}")
    st.sidebar.metric("Accuracy", f"{accuracy:.0f}%")
    progress = st.session_state.num_questions / MAX_QUESTIONS
    st.sidebar.progress(progress)

# -----------------------------
# QUIZ LOGIC
# -----------------------------
if st.session_state.current_question is None and st.session_state.num_questions < MAX_QUESTIONS:
    with st.spinner("🧠 AI is generating your next question..."):
        # Train model and get adaptive difficulty
        model, enc = train_adaptive_model()
        topic, difficulty = select_adaptive_difficulty(model, enc)
        
        # Generate question using OpenAI
        question_data = generate_world_wars_question(client, topic, difficulty)
        
        if question_data:
            question_data["topic"] = topic
            question_data["difficulty"] = difficulty
            question_data["question_id"] = f"{topic}_{difficulty}_{st.session_state.num_questions}"
            st.session_state.current_question = question_data

# Display question
if st.session_state.current_question is not None and st.session_state.num_questions < MAX_QUESTIONS:
    q_data = st.session_state.current_question
    
    # Question header - FIXED: All st.columns() calls have proper arguments
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Question", f"{st.session_state.num_questions + 1}/{MAX_QUESTIONS}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Topic", q_data["topic"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Level", q_data["difficulty"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Score", f"{st.session_state.score}/{st.session_state.num_questions}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Question display
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.markdown(f"### 📖 {q_data['question']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Answer options
    options = [q_data["option1"], q_data["option2"], q_data["option3"], q_data["option4"]]
    options_numbered = [f"{i+1}. {opt}" for i, opt in enumerate(options)]
    
    choice = st.radio("**Choose your answer:**", options_numbered, key=f"q_{st.session_state.num_questions}")

    # Submit button
    col1, col2, col3 = st.columns(3)
    with col2:
        submit_clicked = st.button("📝 Submit Answer", use_container_width=True, type="primary")
    
    if submit_clicked and not st.session_state.answered:
        selected = int(choice.split(".")[0])
        correct_answer = q_data["correct_answer"]
        answered_correct = 1 if selected == correct_answer else 0
        
        if answered_correct:
            st.balloons()
            st.success("🎉 Excellent! You got it right!")
            st.session_state.score += 1
        else:
            st.error(f"🤔 Not quite! The correct answer is: **{correct_answer}. {options[correct_answer-1]}**")
        
        # Show explanation
        with st.expander("📚 Learn More - Click to Expand"):
            st.write("**Explanation:**")
            st.write(q_data["explanation"])
        
        # Log result
        log_result(q_data["topic"], q_data["difficulty"], answered_correct, selected, q_data["question_id"])
        st.session_state.answered = True

        # Educational resources
        st.markdown("### 🎥 Want to Learn More?")
        
        # Calculate performance
        log_data = pd.read_csv(LOG_FILE)
        topic_performance = log_data[
            (log_data["topic"] == q_data["topic"]) & 
            (log_data["difficulty"] == q_data["difficulty"])
        ]["answered_correct"].mean() if not log_data.empty else 0.5
        
        # Get educational videos using OpenAI
        with st.spinner("🔍 AI is finding educational videos for you..."):
            queries = get_educational_youtube_recommendation(
                client, q_data["topic"], q_data["difficulty"], topic_performance
            )
            
            for i, query in enumerate(queries):
                yt_url = fetch_youtube_video(query)
                if yt_url:
                    st.write(f"**📺 Educational Video:** {query}")
                    st.video(yt_url)
                    break  # Show first successful video

    # Next question button
    if st.session_state.answered:
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("➡️ Next Question", use_container_width=True, type="secondary"):
                st.session_state.num_questions += 1
                st.session_state.current_question = None
                st.session_state.answered = False
                st.rerun()

else:
    # Quiz completed
    st.balloons()
    
    # Final results
    final_score = st.session_state.score
    total_questions = MAX_QUESTIONS
    percentage = (final_score / total_questions) * 100
    
    st.markdown(f"""
    <div class="final-score">
        <h1>🎉 Quiz Complete!</h1>
        <h2>Your Final Score: {final_score}/{total_questions}</h2>
        <h3>Accuracy: {percentage:.0f}%</h3>
        <p>Great job learning about World Wars!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance feedback
    if percentage >= 80:
        st.success("🌟 Outstanding work! You have excellent knowledge of World Wars!")
        feedback = "You're ready for more advanced historical topics!"
    elif percentage >= 60:
        st.info("👍 Good job! You have solid understanding with room to grow!")
        feedback = "Keep studying to master all the important details!"
    else:
        st.warning("📚 Keep learning! History takes time to master!")
        feedback = "Focus on the basics and you'll improve quickly!"
    
    st.markdown(f"### 💭 {feedback}")
    
    # YouTube Recommendations Section using OpenAI
    st.markdown("### 🎥 Recommended Learning Videos")
    st.markdown("Click on these links to watch educational videos and improve your World Wars knowledge:")
    
    with st.spinner("🔍 AI is finding the best educational videos for you..."):
        # Get personalized recommendations based on final score using OpenAI
        video_queries = get_final_recommendations(client, final_score, total_questions)
        
        # Display video recommendations as clickable hyperlinks
        video_col1, video_col2 = st.columns(2)
        
        for i, query in enumerate(video_queries):
            yt_url = fetch_youtube_video(query)
            
            with video_col1 if i % 2 == 0 else video_col2:
                if yt_url:
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #4CAF50;">
                        <h5>📺 {query}</h5>
                        <a href="{yt_url}" target="_blank" style="background: #FF0000; color: white; padding: 0.5rem 1rem; text-decoration: none; border-radius: 5px; display: inline-block;">
                            ▶️ Watch Video
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #6c757d;">
                        <h5>🔍 {query}</h5>
                        <p style="color: #666; font-size: 0.9rem;">Search this on YouTube to learn more!</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Take Quiz Again", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                if key.startswith(('score', 'num_questions', 'asked_questions', 'current_question', 'answered')):
                    del st.session_state[key]
            st.rerun()
    
    with action_col2:
        if st.button("📊 Download Report", use_container_width=True):
            if os.path.exists(LOG_FILE):
                df_log = pd.read_csv(LOG_FILE)
                csv = df_log.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name="world_wars_quiz_progress.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with action_col3:
        if st.button("📚 Study Tips", use_container_width=True):
            st.info("""
            **📖 How to Improve Your World Wars Knowledge:**
            - Read your history textbook carefully
            - Watch documentaries about WWI and WWII
            - Visit history museums if possible
            - Ask your teacher questions about confusing topics
            - Practice with timeline activities
            - Learn about key figures like Roosevelt, Churchill, and Hitler
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎓 Made for 6th Grade Students | 📚 Learn History with AI | 🌟 Keep Learning!</p>
</div>
""", unsafe_allow_html=True)
