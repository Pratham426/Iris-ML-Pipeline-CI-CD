import sys
import os
import pytest
import json

# Add the project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_valid_input(client):
    """Test prediction with valid input"""
    test_data = {
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5,
        'petal length (cm)': 1.4,
        'petal width (cm)': 0.2
    }
    response = client.post('/predict', json=test_data)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability' in data
    assert 'probabilities' in data

def test_predict_invalid_input(client):
    """Test prediction with missing fields"""
    test_data = {
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5
    }
    response = client.post('/predict', json=test_data)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_home_route(client):
    """Test the home route"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["API"] == "Iris Flower Classification API"
