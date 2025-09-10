# 🚀 AI Adaptive Quiz Platform

> **A next-generation, AI-powered quiz system to transform any learning topic into an adaptive experience!**

---

## 🌟 What is this?

**AI Adaptive Quiz Platform** is a Streamlit-based, fully extensible quiz generator for students, teachers, and self-learners.  
It uses the OpenAI API to create unlimited, curriculum-friendly quizzes and adapts to the learner’s strengths and weaknesses in real time.  
At the end, it recommends personalized YouTube learning videos—making learning truly interactive and fun.

---

## 🎨 Features

- 🤖 **AI-Generated Questions:** Dynamic questions on any topic/subject with GPT-3.5 or GPT-4.
- 🎯 **Adaptive Difficulty:** Learner’s performance drives the challenge every step.
- 📝 **Detailed Explanations:** Each question comes with an AI-generated explanation.
- 📊 **Real-Time Progress Bar:** Track your quiz journey as you go.
- 📺 **YouTube Recommendations:** AI suggests relevant educational videos at the end—click and learn more!
- 📥 **Download Progress:** Export your quiz results for review or record-keeping.
- 💡 **No Hard-Coding:** Change quiz topic or grade by editing a single value or prompt.

---

## 🚦 Quick Start

### 1️⃣ Clone this Repository


### 2️⃣ Install Dependencies


### 3️⃣ Set up the OpenAI API Key

- Create a `.env` file in your project root:
    ```
    OPENAI_API_KEY=your-openai-api-key-here
    ```
- Get your OpenAI key at [platform.openai.com](https://platform.openai.com/).

**Your key is required for all AI features.**  
🚨 Never commit your `.env` file or API key to public repos!

---

## 🧑‍🎓 Use Cases

- 🎒 Classroom quizzes (all subjects, grades 6+)
- 🏡 Self-paced revision for students
- 🌐 EdTech platforms: embed for custom learning
- 🧑‍💼 Professional onboarding/skills validation

---

## 🛠️ Technologies Used

| Tool                  | Purpose                    |
|-----------------------|---------------------------|
| **Python 3.x**        | Programming language      |
| **Streamlit**         | Interactive Web UI        |
| **OpenAI API**        | Question & video query generation |
| **scikit-learn**      | ML for difficulty adaptation   |
| **YouTube Search**    | Video recommendations     |
| **pandas**            | Data handling             |
| **python-dotenv**     | Environment/configuration |

---

## 📚 Customization Tips

- **Change subject**: Alter the "topic" at quiz start or prompt string in `app.py`.
- **Set grade/difficulty**: Tune your prompting or model adaptation to any age/level.
- **UI Themes**: Edit Streamlit or CSS settings for color/branding.
- **Extend reporting**: Save/export more analytics as needed.

---

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)
- [Youtube Search Python](https://github.com/alexmercerind/youtube-search-python)
- All educators inspiring adaptive learning!

---

**Ready to make any topic interactive, personalized and fun? Fork and try it now!** 🌍✨

