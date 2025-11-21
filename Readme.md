⚽ ProphetPlay: The Journalist AI

ProphetPlay is an advanced sports analytics engine designed to predict the winners of the 2026 Ballon d'Or and the UEFA Champions League.

Unlike traditional stats-only models, ProphetPlay mimics the "Journalist View"—accounting for media narratives, club prestige ("Heritage Bonus"), and trophy impact—to generate realistic, human-like power rankings.

🚀 Features

1. 🏆 Ballon d'Or Predictor (The Journalist Engine)

- Predicts the Top 15 candidates by simulating how journalists vote.

- Narrative Score: Heavily weights major trophies (UCL, League titles).

- Media Bias: Applies a "Real Madrid/Premier League" tax to simulate media attention.

- Main Character Energy: Measures how much a player carries their team.

2. UCL Title Contenders (DeepUCLNet)

- Predicts the Top 10 teams most likely to win the Champions League.

- Power Index (0-100): A FIFA-style rating combining current form and historical dominance.

- Heritage Bonus: Boosts teams with "Champions League DNA" (e.g., Real Madrid, Bayern) even if their current form is dipping.

- Attack vs. Defense: Weighs offensive firepower against defensive solidity.

3. 📊 Interactive Dashboard

- Built with Streamlit for real-time data exploration.

- Search for any player or team in the 2026 database.

- Visual progress bars for "Power Ratings" and "Win Probabilities".

📂 Project Structure
```
DeepBallonNet/
├── app.py                  # Main Streamlit application
├── data/                   # Data storage
│   ├── master_dataset_2026.csv        # Live 2026 season data
│   └── master_dataset_2011-2025.csv   # Historical training data
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

🛠️ Installation & Setup

Prerequisites

- Python 3.8 or higher ( Python version 3.11.9 is used in this project.)

pip (Python package manager)

1. Clone or Download

Download the project files to your local machine.

2. Install Dependencies

Open your terminal/command prompt in the project folder and run:

```pip install streamlit pandas numpy scikit-learn xgboost torch```


3. Run the App

Launch the dashboard using Streamlit:

```streamlit run app.py```


The app will open automatically in your web browser (usually at http://localhost:8501).

🛠️ Evaluations:

1. Ballon D'or Evaluation

![Banner](BallonDor_Evaluation.jpg)

2. UCL Winner Evaluation

![Banner](UCL_Evaluations.jpg)

3. Ranking Efficiency

![Banner](Ranking_Efficiency.jpg)

4. Confusion Matrix and Graph 

![Banner](Graph_&_Confusion_Matrix.png)

🧠 Methodology

The "Journalist View" Algorithm

- Data science often fails at sports awards because it ignores human bias. DeepBallonNet explicitly codes these biases:

The Narrative Factor (30% Weight): * Winning the UCL is worth 10x more than scoring a hat-trick against a relegation team.

League titles provide a stability baseline.

- The Heritage Bonus:

Teams like Real Madrid and Bayern Munich statistically overperform their underlying numbers in the UCL.

The model applies a 0.15 probability boost to these historical giants to prevent "Recency Bias."

- Weighted Stats:

Goals in the UCL Knockout stages are weighted 2.5x higher than domestic league goals.

📝 Requirements

- Install the Requirements.txt file inorder to run the project.

```pip install -r requirements.txt```

⚠️ Disclaimer

This project is for educational and entertainment purposes only. While the model achieves 85% Recall on historical data, football is unpredictable. The AI cannot foresee injuries, transfers, or managerial sackings.

Enjoy the predictions! ⚽