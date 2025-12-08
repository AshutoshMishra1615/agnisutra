import 'package:web_socket_channel/web_socket_channel.dart';
import '../constants.dart';

class SocketService {
  WebSocketChannel? _channel;

  Stream get stream {
    if (_channel == null) {
      connect();
    }
    return _channel!.stream;
  }

  void connect() {
    try {
      final uri = Uri.parse(AppConstants.baseUrl);
      final scheme = uri.scheme == 'https' ? 'wss' : 'ws';

      // Explicitly construct the WebSocket URI
      final wsUri = Uri(
        scheme: scheme,
        host: uri.host,
        port: uri.port,
        path: '/ws/alerts',
      );

      print('Attempting to connect to WebSocket: $wsUri');
      _channel = WebSocketChannel.connect(wsUri);
    } catch (e) {
      print('Error connecting to WebSocket: $e');
    }
  }

  void disconnect() {
    _channel?.sink.close();
    _channel = null;
  }
}
