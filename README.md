The Enhanced Multi-Religious Ethics Assessment System is a sophisticated Streamlit web application that analyzes ethical scenarios across multiple dimensions. Here's how it works:

*Core Functionality:
1) Multi-Dimensional Analysis: Evaluates scenarios across 6 religions, 5 ethical frameworks, and 6 scenario types
2) Advanced NLP: Uses TextBlob and VADER for sentiment analysis
3) Contextual Scoring: Applies weighted scoring with contextual modifiers
4) Consensus Measurement: Calculates agreement across different ethical traditions
5) Interactive Visualizations: Creates radar charts, bar charts, pie charts, and gauge displays


*Key Libraries Used:
1) Streamlit: Web interface and user interaction
2) Pandas/NumPy: Data manipulation and numerical calculations
3) TextBlob/VADER: Natural language processing and sentiment analysis
4) Plotly: Interactive visualizations
5) Regular Expressions: Text preprocessing


*If Running on a personal device 
1) Install Dependencies:
bash
pip install streamlit pandas numpy textblob vaderSentiment plotly requests

2) Run Quick Test:
bash
python quick_test_runner.py

3) Launch Application:
bash
streamlit run ethics_analyzer.py
