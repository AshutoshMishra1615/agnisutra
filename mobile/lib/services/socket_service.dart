<<<<<<< HEAD
=======
import 'dart:async';
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
import 'package:web_socket_channel/web_socket_channel.dart';
import '../constants.dart';

class SocketService {
  WebSocketChannel? _channel;
<<<<<<< HEAD

  Stream get stream {
    if (_channel == null) {
      connect();
    }
    return _channel!.stream;
  }

  void connect() {
=======
  final StreamController _controller = StreamController.broadcast();
  bool _isConnected = false;

  Stream get stream => _controller.stream;

  void connect() {
    if (_isConnected) return;

>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
    try {
      final uri = Uri.parse(AppConstants.baseUrl);
      final scheme = uri.scheme == 'https' ? 'wss' : 'ws';

<<<<<<< HEAD
      // Explicitly construct the WebSocket URI
=======
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
      final wsUri = Uri(
        scheme: scheme,
        host: uri.host,
        port: uri.port,
        path: '/ws/alerts',
      );

<<<<<<< HEAD
      print('Attempting to connect to WebSocket: $wsUri');
      _channel = WebSocketChannel.connect(wsUri);
    } catch (e) {
      print('Error connecting to WebSocket: $e');
=======
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
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
    }
  }

  void disconnect() {
    _channel?.sink.close();
    _channel = null;
<<<<<<< HEAD
=======
    _isConnected = false;
  }

  void dispose() {
    disconnect();
    _controller.close();
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
  }
}
