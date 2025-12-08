import 'dart:async';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../constants.dart';

class SocketService {
  WebSocketChannel? _channel;
  final StreamController _controller = StreamController.broadcast();
  bool _isConnected = false;

  Stream get stream => _controller.stream;

  void connect() {
    if (_isConnected) return;

    try {
      final uri = Uri.parse(AppConstants.baseUrl);
      final scheme = uri.scheme == 'https' ? 'wss' : 'ws';

      final wsUri = Uri(
        scheme: scheme,
        host: uri.host,
        port: uri.port,
        path: '/ws/alerts',
      );

      print('Connecting to WebSocket: $wsUri');
      _channel = WebSocketChannel.connect(wsUri);
      _isConnected = true;

      _channel!.stream.listen(
        (data) {
          _controller.add(data);
        },
        onError: (error) {
          print('WebSocket Error: $error');
          _isConnected = false;
          // _controller.addError(error); // Optional: propagate error
        },
        onDone: () {
          print('WebSocket Closed');
          _isConnected = false;
        },
      );
    } catch (e) {
      print('Connection failed: $e');
      _isConnected = false;
    }
  }

  void disconnect() {
    _channel?.sink.close();
    _channel = null;
    _isConnected = false;
  }

  void dispose() {
    disconnect();
    _controller.close();
  }
}
