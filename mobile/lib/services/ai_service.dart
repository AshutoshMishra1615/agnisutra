import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../constants.dart';

class AiService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  Future<Map<String, dynamic>?> chat({
    required String sessionId,
    required String query,
    Map<String, dynamic>? yieldContext,
    String language = 'auto',
  }) async {
    try {
      String? token = await _storage.read(key: 'access_token');

      final response = await _dio.post(
        '${AppConstants.baseUrl}/krishi-sathi/chat',
        data: {
          "session_id": sessionId,
          "query": query,
          "yield_context": yieldContext ?? {},
          "language": language,
        },
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
            'Content-Type': 'application/json',
          },
        ),
      );

      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      print('AI Chat error: $e');
      rethrow;
    }
  }
}
