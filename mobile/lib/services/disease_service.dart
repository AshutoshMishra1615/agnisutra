import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../constants.dart';

class DiseaseService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  Future<Map<String, dynamic>?> predictDisease(
    File imageFile,
    String cropName,
    String query,
  ) async {
    try {
      String? token = await _storage.read(key: 'access_token');

      String fileName = imageFile.path.split('/').last;
      FormData formData = FormData.fromMap({
        "file": await MultipartFile.fromFile(
          imageFile.path,
          filename: fileName,
        ),
        "crop_name": cropName,
        "query": query,
      });

      final response = await _dio.post(
        '${AppConstants.baseUrl}/disease/predict',
        data: formData,
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
          },
        ),
      );

      if (response.statusCode == 200) {
        return response.data as Map<String, dynamic>;
      }
      return null;
    } catch (e) {
      print('Disease prediction error: $e');
      rethrow;
    }
  }
}
