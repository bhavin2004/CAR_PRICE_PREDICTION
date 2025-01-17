import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200

def test_train_endpoint(client):
    response = client.get('/train')
    assert response.status_code == 200

def test_predict_endpoint(client):
    response = client.get('/predict')
    assert response.status_code == 200
