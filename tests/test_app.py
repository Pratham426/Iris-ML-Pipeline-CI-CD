import sys
import os

# Add the project root (where app.py is) to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # Now works

import pytest
from app import app
import joblib
import numpy as np
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_valid_input(client):
    """Test prediction with valid input"""
    test_data = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
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
        'sepal_length': 5.1,
        'sepal_width': 3.5
    }
    response = client.post('/predict', json=test_data)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_home_route(client):
    """Test the home route"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Iris Classification API" in response.data
