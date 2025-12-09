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

      if (token == null) {
        print('Error: Access token is null');
        return null;
      }

      print('Sending Chat Request:');
      print('Query: $query');
      print('Language: $language');
      print('Yield Context Keys: ${yieldContext?.keys.toList()}');

      final response = await _dio.post(
        '${AppConstants.baseUrl}/krishi-saathi/chat',
        data: {
          "session_id": token,
          "query": query,
          "yield_context": yieldContext ?? {},
          "language": language,
        },
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
            'Content-Type': 'application/json',
          },
          validateStatus: (status) => status! < 500,
        ),
      );

      if (response.statusCode == 200) {
        print('AI Response: ${response.data}');
        return response.data;
      }

      print('AI Service Error: ${response.statusCode}');
      print('Response Body: ${response.data}');
      return null;
    } catch (e) {
      print('AI Chat error: $e');
      if (e is DioException) {
        print('DioError response: ${e.response?.data}');
      }
      rethrow;
    }
  }
}
