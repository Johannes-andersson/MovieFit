# 🎬 Movie Fit Checker

A machine learning-powered web application that helps you determine if a movie is a good fit for your taste based on your personal movie rating history. The app uses k-NN (k-Nearest Neighbors) content-based filtering to analyze movie features and compare them against your liked movies.

## ✨ Features

- **Personalized Movie Recommendations**: Uses your existing movie ratings to predict if you'll like a new movie
- **Content-Based Filtering**: Analyzes movie genres, runtime, and year to find similar films
- **OMDb Integration**: Auto-fill movie details using the OMDb API
- **Interactive Dashboard**: Clean, user-friendly interface built with Streamlit
- **Similarity Scoring**: Shows how similar a movie is to your liked movies (0-100%)
- **Nearest Neighbors Explanation**: Displays the most similar movies from your collection

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd moviegrader
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get your OMDb API key** (Required for auto-fill feature)
   - Visit [https://www.omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx)
   - Click "FREE! (1,000 daily limit)"
   - Enter your email address
   - Check your email for the API key
   - Copy the API key (it looks like: `a7b9c2d4`)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal

## 📊 How It Works

### Data Requirements

Your movie data should be in a CSV file named `movies_1989_2025.csv` with these columns:
- `title`: Movie title
- `year`: Release year
- `genres`: Comma-separated genres (e.g., "comedy, action")
- `runtime_min`: Runtime in minutes
- `my_rating`: Your rating (1-5 scale)

### Machine Learning Approach

1. **Feature Engineering**: Converts movie genres into binary vectors and normalizes numerical features (runtime, year)
2. **Similarity Calculation**: Uses cosine similarity to find movies with similar content
3. **Recommendation**: Compares new movies against your "liked" movies (ratings ≥ threshold)
4. **Scoring**: Provides a fit score based on average similarity to your liked movies

### Settings

- **Like Threshold**: Movies with ratings ≥ this value are considered "liked" (default: 4)
- **Neighbors to Show**: Number of similar movies to display for explanation (default: 10)

## 🎯 Usage

1. **Enter Movie Details**:
   - Type the movie title
   - Set the year and runtime
   - Select relevant genres from your library

2. **Auto-fill with OMDb** (Optional):
   - Enter your OMDb API key
   - Click "Auto-fill from OMDb" to automatically populate movie details

3. **Check Fit**:
   - Click "Check fit" to get your personalized score
   - View similar movies from your collection

## 📁 Project Structure

```
moviegrader/
├── app.py                    # Main Streamlit application
├── movies_1989_2025.csv     # Your movie rating data
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .venv/                  # Virtual environment (created during setup)
```

## 🔧 Dependencies

- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `requests`: HTTP library for OMDb API

## 🎨 Customization

### Adding More Movies
Simply add new rows to your CSV file with the required columns. The app will automatically include them in the analysis.

### Adjusting the Algorithm
You can modify the similarity calculation in the `fit_score_vs_likes()` function or change the feature engineering in `build_features()`.

### UI Customization
The Streamlit interface can be customized by modifying the layout, colors, and components in the `app.py` file.

## 🐛 Troubleshooting

### Common Issues

1. **"No movies found" error**:
   - Ensure your CSV file is named `movies_1989_2025.csv`
   - Check that all required columns are present
   - Verify the file is in the same directory as `app.py`

2. **OMDb API errors**:
   - Verify your API key is correct
   - Check your internet connection
   - Ensure you haven't exceeded the daily limit (1,000 requests)

3. **"You have no 'liked' movies" error**:
   - Lower the "like threshold" in the sidebar
   - Add more movies with higher ratings to your dataset

### Getting Help

If you encounter issues:
1. Check the terminal/console for error messages
2. Ensure all dependencies are installed correctly
3. Verify your CSV file format matches the requirements

## 📈 Future Enhancements

- Support for additional movie features (director, cast, plot keywords)
- Collaborative filtering recommendations
- Export recommendations to watchlist
- Rating prediction for unrated movies
- Genre preference analysis

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

---

**Happy movie watching! 🍿**
