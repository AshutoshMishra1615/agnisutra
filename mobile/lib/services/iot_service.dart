import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../constants.dart';

class IotService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  Future<Map<String, dynamic>?> getLatestData() async {
    try {
      String? token = await _storage.read(key: 'access_token');

      final response = await _dio.get(
        '${AppConstants.baseUrl}/iot/latest',
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
      print('IoT Service Error: $e');
      return null;
    }
  }
}
