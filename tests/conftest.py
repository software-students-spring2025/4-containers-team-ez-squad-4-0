import pytest
from unittest.mock import MagicMock
from app import app, socketio

@pytest.fixture
def client():
    """提供一个用于测试的 Flask 客户端"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# ---------- 页面路由相关 ----------

def test_index_route(client):
    """测试主页是否正确渲染"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Voice Command Flappy Game' in response.data
    assert b'voice commands' in response.data.lower()

def test_scores_route(client, monkeypatch):
    """测试分数路由，模拟数据库可用情况"""
    with monkeypatch.context() as m:
        m.setattr('app.scores_collection.find', lambda: [{"score": 100, "timestamp": "2025-04-10T10:00:00"}])
        response = client.get('/scores')
        assert response.status_code == 200
        assert b'High Scores' in response.data
        assert b'100' in response.data

def test_scores_route_no_db(client, monkeypatch):
    """测试分数路由，模拟数据库不可用情况"""
    with monkeypatch.context() as m:
        m.setattr('app.scores_collection', None)
        response = client.get('/scores')
        assert response.status_code == 200
        assert b'Database not connected' in response.data

# ---------- 分数提交 ----------

def test_receive_score(client, monkeypatch):
    """测试分数提交路由，提交一个有效的分数"""
    with monkeypatch.context() as m:
        mock_insert = MagicMock()
        m.setattr('app.scores_collection.insert_one', mock_insert)
        response = client.post('/score', json={"score": 10})
        assert response.status_code == 200
        mock_insert.assert_called_once_with({'score': 10, 'timestamp': datetime.now(timezone.utc)})

# ---------- SocketIO ----------

def test_socket_connect_disconnect(client):
    """测试 Socket.IO 连接和断开事件"""
    socket_client = socketio.test_client(client.application, flask_test_client=client)
    assert socket_client.is_connected()
    socket_client.disconnect()

# ---------- 音频处理 ----------

def test_handle_audio_invalid_base64(client):
    """测试处理无效的 base64 音频数据"""
    invalid_audio = "data:audio/webm;base64,!!!invalidbase64!!!"
    test_client = socketio.test_client(client.application, flask_test_client=client)
    test_client.connect()
    test_client.emit("audio", invalid_audio)
    received = test_client.get_received()
    test_client.disconnect()
    assert any(event["name"] == "command" and event["args"][0] == "stop" for event in received)
