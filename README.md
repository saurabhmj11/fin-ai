# FIN-AI Agent Web App

## Overview
The Multi-AI Agent Web App is a Streamlit-powered platform that combines advanced AI models and specialized tools to deliver actionable insights in real-time. This application is designed to handle tasks such as web research, financial analysis, and news aggregation, providing structured and well-formatted results.

---

## Features

### 🔍 Web Research
- Integrated with **DuckDuckGo** for reliable and up-to-date web search results.
- Fetches and summarizes the latest news and information based on user queries.

### 📊 Financial Insights
- Uses **YFinance Tools** to:
  - Retrieve current stock prices.
  - Summarize analyst recommendations.
  - Display company financial fundamentals.
  - Provide the latest company news.

### 💡 Intelligent Query Processing
- Leverages multiple agents for domain-specific tasks:
  - **Web Search Agent** for online information.
  - **Finance Agent** for stock market and financial data.
- Automatically selects the best agent for the task.

### 🎨 User-Friendly Interface
- Built with **Streamlit** for an interactive and visually appealing experience.
- Clean and well-structured outputs using Markdown and tables.

---

## Tech Stack

### Backend
- **AI Models**: Groq `llama-3.1-70b-versatile`
- **Tools**: 
  - DuckDuckGo (for web search)
  - YFinance (for financial data)

### Frontend
- **Streamlit**: Ensures quick deployment and an intuitive user interface.

### Additional
- **Python**: Core programming language for implementation.
- **dotenv**: For secure handling of environment variables.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/multi-ai-agent-web-app.git
   cd multi-ai-agent-web-app
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and configure your environment variables (e.g., API keys).

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

---

## Usage
1. Enter your query in the input field (e.g., "Get the latest news about Tesla").
2. Click on the **Submit Query** button.
3. View the results in a well-formatted table or markdown output.

---

## Example Queries
- "Summarize analyst recommendations and share the latest news for NVDA."
- "What is the current stock price of Tesla?"
- "Get the latest news about ChatGPT."

---

## Contributions
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a new feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For questions or support, reach out to:
- **Email**: [saurabhmj11@gmail.com](mailto:saurabhmj11@gmail.com)
- **LinkedIn**: [Saurabh Lokhande](https://www.linkedin.com/in/saurabh-lokhande/)
- **GitHub**: [saurabhmj11](https://github.com/saurabhmj11)

---

## Acknowledgments
- **Streamlit** for the amazing framework.
- **DuckDuckGo** and **YFinance** for their APIs.
- The OpenAI and Groq teams for cutting-edge AI models.

---

### Let's revolutionize how we access and utilize information! 🌟
