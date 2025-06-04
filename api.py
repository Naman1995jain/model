from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import uvicorn
from typing import List, Optional
from datetime import datetime

app = FastAPI(
    title="ML Models API",
    description="API for House Price Prediction and Movie Recommendations",
    version="1.0.0"
)

# Load the models and data
try:
    model = joblib.load('models/gradient_boosting_model.joblib')
    housing_data = pd.read_csv('BostonHousing.csv')
    movies_data = pd.read_csv('Top_rated_movies1.csv')
except Exception as e:
    print(f"Error loading models/data: {e}")

# Initialize scaler
scaler = StandardScaler()
scaler.fit(housing_data.drop('medv', axis=1))

# Pydantic models for request validation
class HousePredictionRequest(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    b: float
    lstat: float

    class Config:
        schema_extra = {
            "example": {
                "crim": 0.00632,
                "zn": 18.0,
                "indus": 2.31,
                "chas": 0,
                "nox": 0.538,
                "rm": 6.575,
                "age": 65.2,
                "dis": 4.09,
                "rad": 1,
                "tax": 296.0,
                "ptratio": 15.3,
                "b": 396.9,
                "lstat": 4.98
            }
        }

class MovieRecommendationRequest(BaseModel):
    movie_id: int
    num_recommendations: Optional[int] = 5

    class Config:
        schema_extra = {
            "example": {
                "movie_id": 168705,
                "num_recommendations": 5
            }
        }

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: dict
    prediction_time: str
    model_version: str

class MovieRecommendation(BaseModel):
    id: int
    title: str
    overview: str
    similarity_score: float
    vote_average: float

class MovieRecommendationsResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    recommendation_time: str

# API endpoints
@app.get("/")
def read_root():
    return {
        "message": "Welcome to ML Models API",
        "endpoints": {
            "/predict/house-price": "Predict house prices",
            "/recommend/movies": "Get movie recommendations",
            "/health": "Check API health",
            "/models/info": "Get models information"
        }
    }

@app.post("/predict/house-price", response_model=PredictionResponse)
def predict_house_price(request: HousePredictionRequest):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Calculate confidence interval (simple example)
        std_dev = 2.5  # This should be calculated based on model validation
        confidence_interval = {
            "lower_bound": prediction - 1.96 * std_dev,
            "upper_bound": prediction + 1.96 * std_dev,
            "confidence_level": 0.95
        }
        
        return PredictionResponse(
            predicted_price=float(prediction),
            confidence_interval=confidence_interval,
            prediction_time=datetime.now().isoformat(),
            model_version="gradient_boosting_v1"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/movies", response_model=MovieRecommendationsResponse)
def recommend_movies(request: MovieRecommendationRequest):
    try:
        # Check if movie exists
        if request.movie_id not in movies_data['id'].values:
            raise HTTPException(status_code=404, detail="Movie ID not found")
        
        # Get movie index
        movie_idx = movies_data[movies_data['id'] == request.movie_id].index[0]
        
        # Calculate similarity (simplified version - you might want to use your existing similarity matrix)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_data['overview'].fillna(''))
        similarity = cosine_similarity(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
        
        # Get top similar movies
        similar_indices = similarity.argsort()[::-1][1:request.num_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            movie = movies_data.iloc[idx]
            recommendations.append(
                MovieRecommendation(
                    id=int(movie['id']),
                    title=str(movie['title']),
                    overview=str(movie['overview']),
                    similarity_score=float(similarity[idx]),
                    vote_average=float(movie['vote_average'])
                )
            )
        
        return MovieRecommendationsResponse(
            recommendations=recommendations,
            recommendation_time=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "house_price_model": model is not None,
            "movies_data": movies_data is not None
        }
    }

@app.get("/models/info")
def get_models_info():
    return {
        "house_price_model": {
            "type": "gradient_boosting",
            "features": list(housing_data.columns[:-1]),
            "target": "medv",
            "data_shape": housing_data.shape
        },
        "movie_recommender": {
            "total_movies": len(movies_data),
            "features_used": ["overview", "vote_average"],
            "recommendation_method": "content-based-filtering"
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
