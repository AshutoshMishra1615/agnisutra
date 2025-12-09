import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../constants.dart';

class NdviService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  Future<Map<String, dynamic>?> getNdviData(double lat, double lon) async {
    try {
      String? token = await _storage.read(key: 'access_token');

      final response = await _dio.post(
        '${AppConstants.baseUrl}/krishi-saathi/ndvi',
        data: {"lat": lat, "lon": lon},
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
      print('NDVI Service Error: $e');
      return null;
    }
  }
}
